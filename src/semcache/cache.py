"""The semantic cache engine (SPEC section 3).

request -> policy -> scope -> normalise -> L1 exact -> (single-flight) -> L2 semantic
        -> guards -> grey-zone verifier -> upstream on a miss -> store + metrics + log

The cache fails open: if the embedder or the store breaks, the request still gets a real
answer from upstream. A cache must never make the app less available than it was without it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from semcache.config import CacheConfig, Settings
from semcache.decision import evaluate_candidate
from semcache.embed import Embedder
from semcache.metrics import Metrics
from semcache.normalize import exact_key, normalize
from semcache.policy import RequestPolicy, request_policy, store_policy
from semcache.scope import Scope, build_scope
from semcache.singleflight import SingleFlight
from semcache.store.base import CacheStore
from semcache.types import CacheEntry, ChatRequest, ChatResponse, Decision
from semcache.upstream import Upstream, UpstreamResult
from semcache.verify import Verifier

log = logging.getLogger("semcache")


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class _Ctx:
    model_id: str
    policy: RequestPolicy
    scope: Scope
    norm: str
    xkey: str


@dataclass
class _Lookup:
    entry: CacheEntry | None = None
    score: float | None = None  # best similarity seen (or the accepted candidate's)
    decision: Decision = Decision.MISS
    guard: str | None = None
    vector: np.ndarray | None = None
    overhead_usd: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class SemanticCache:
    def __init__(
        self,
        *,
        store: CacheStore,
        embedder: Embedder,
        upstream: Upstream,
        verifier: Verifier | None = None,
        config: CacheConfig | None = None,
        metrics: Metrics | None = None,
        chat_model_id: str = "mock-upstream",
        now: Callable[[], datetime] = utcnow,
    ):
        self.store = store
        self.embedder = embedder
        self.upstream = upstream
        self.verifier = verifier
        self.config = config or CacheConfig()
        self.metrics = metrics or Metrics()
        self.chat_model_id = chat_model_id
        self.now = now
        self.singleflight = SingleFlight(self.config.singleflight_timeout_s)

    # --- construction from .env ------------------------------------------------------
    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, config: CacheConfig | None = None
    ) -> SemanticCache:
        """Real Bedrock embedder, upstream and judge; pgvector if DATABASE_URL is set."""
        from semcache.bedrock import get_runtime_client
        from semcache.config import get_settings
        from semcache.embed import TitanEmbedder
        from semcache.store.memory import InMemoryStore
        from semcache.upstream import BedrockUpstream
        from semcache.verify import LLMVerifier

        settings = settings or get_settings()
        client = get_runtime_client(settings)
        embedder = TitanEmbedder(
            client,
            model_id=settings.bedrock_embed_model_id,
            dimensions=settings.embed_dimensions,
        )
        if settings.database_url:
            from semcache.store.pgvector import PgVectorStore

            store = PgVectorStore(settings.database_url, settings.embed_dimensions)
        else:
            store = InMemoryStore()
        return cls(
            store=store,
            embedder=embedder,
            upstream=BedrockUpstream(client),
            verifier=LLMVerifier(client, settings.bedrock_judge_model_id),
            config=config,
            chat_model_id=settings.bedrock_chat_model_id,
        )

    # --- public API ------------------------------------------------------------------
    def get_or_call(self, request: ChatRequest) -> ChatResponse:
        """Synchronous wrapper for scripts. Inside async code, use `aget_or_call`."""
        return asyncio.run(self.aget_or_call(request))

    async def aget_or_call(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        ctx = self._prepare(request)
        if not ctx.policy.use_cache:
            resp = await self._bypass(request, ctx)
        else:
            entry = self._get_exact(ctx)
            if entry is not None:
                resp = self._hit(entry, Decision.EXACT_HIT, 1.0, ctx, overhead=0.0, timings={})
            else:
                resp, shared = await self.singleflight.do(
                    ctx.xkey, lambda: self._lookup_or_call(request, ctx)
                )
                if shared:
                    resp = self._coalesced(resp)
        resp.latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.observe(resp, namespace=request.namespace)
        return resp

    async def astream(self, request: ChatRequest) -> AsyncIterator[dict]:
        """Yield {"delta": text} events, then one {"done": True, "response": ChatResponse}.

        A hit replays the stored text in chunks; a miss streams live and is stored only if
        the stream finished cleanly. Streaming misses are not coalesced (SPEC 9.1).
        """
        start = time.perf_counter()
        ctx = self._prepare(request)
        resp: ChatResponse | None = None
        if ctx.policy.use_cache:
            entry = self._get_exact(ctx)
            if entry is not None:
                resp = self._hit(entry, Decision.EXACT_HIT, 1.0, ctx, overhead=0.0, timings={})
            else:
                look = await self._lookup(request, ctx)
                if look.entry is not None:
                    resp = self._hit(
                        look.entry, look.decision, look.score, ctx, look.overhead_usd, look.timings
                    )
        if resp is not None:
            for chunk in _chunks(resp.text, self.config.stream_chunk_chars):
                yield {"delta": chunk}
        else:
            look = look if ctx.policy.use_cache else _Lookup()
            result: UpstreamResult | None = None
            async for item in self._aiter_upstream(request, ctx.model_id):
                if isinstance(item, UpstreamResult):
                    result = item
                else:
                    yield {"delta": item}
            assert result is not None
            if ctx.policy.use_cache:
                resp = self._miss(request, ctx, look, result)
            else:
                resp = self._bypass_response(ctx, result)
        resp.latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.observe(resp, namespace=request.namespace)
        yield {"done": True, "response": resp}

    def invalidate(self, **selector) -> int:
        return self.store.invalidate(**selector)

    def purge_expired(self) -> int:
        return self.store.purge_expired(self.now())

    def scope_for(self, request: ChatRequest) -> Scope:
        return self._prepare(request).scope

    # --- steps -----------------------------------------------------------------------
    def _prepare(self, request: ChatRequest) -> _Ctx:
        model_id = request.model or self.chat_model_id
        policy = request_policy(request, self.config.namespace(request.namespace))
        scope = build_scope(
            request,
            model_id=model_id,
            embed_model_id=self.embedder.model_id,
            embed_dim=self.embedder.dimensions,
        )
        norm = normalize(request.question)
        return _Ctx(model_id, policy, scope, norm, exact_key(scope.key, norm))

    def _get_exact(self, ctx: _Ctx) -> CacheEntry | None:
        try:
            return self.store.get_exact(ctx.scope.key, ctx.xkey, self.now())
        except Exception:
            log.exception("store.get_exact failed; continuing without the cache")
            return None

    async def _lookup(self, request: ChatRequest, ctx: _Ctx) -> _Lookup:
        look = _Lookup()
        try:
            emb = await asyncio.to_thread(self.embedder.embed, ctx.norm)
        except Exception:
            log.exception("embedding failed; treating the request as a miss")
            look.error = "embed_error"
            return look
        look.vector = emb.vector
        look.overhead_usd += emb.cost_usd
        look.timings["embed"] = emb.latency_ms

        t = time.perf_counter()
        try:
            candidates = self.store.search(ctx.scope.key, emb.vector, self.config.top_k, self.now())
        except Exception:
            log.exception("store.search failed; treating the request as a miss")
            look.error = "store_error"
            return look
        look.timings["search"] = (time.perf_counter() - t) * 1000
        if candidates:
            look.score = candidates[0][1]

        for entry, score in candidates:  # best first
            if self.verifier is not None and self.config.verifier_enabled:
                v = await asyncio.to_thread(
                    evaluate_candidate,
                    request.question,
                    entry.query_text,
                    score,
                    self.config,
                    self.verifier,
                )
            else:
                v = evaluate_candidate(request.question, entry.query_text, score, self.config)
            if v.judged:
                look.overhead_usd += v.judge_cost_usd
                look.timings["judge"] = look.timings.get("judge", 0.0) + v.judge_ms
            if v.accept:
                look.entry, look.score, look.decision = entry, score, v.decision
                look.guard = None
                return look
            if v.reason == "below_t_low":
                break  # candidates are sorted: the rest score even lower
            if v.guard and look.guard is None:
                look.guard = v.guard
        look.decision = Decision.GUARD_REJECT if look.guard else Decision.MISS
        return look

    async def _lookup_or_call(self, request: ChatRequest, ctx: _Ctx) -> ChatResponse:
        look = await self._lookup(request, ctx)
        if look.entry is not None:
            return self._hit(
                look.entry, look.decision, look.score, ctx, look.overhead_usd, look.timings
            )
        result = await self._call_upstream(request, ctx.model_id)
        return self._miss(request, ctx, look, result)

    async def _call_upstream(self, request: ChatRequest, model_id: str) -> UpstreamResult:
        try:
            return await asyncio.to_thread(self.upstream.complete, request, model_id)
        except Exception:
            self.metrics.upstream_errors.inc()
            raise

    async def _bypass(self, request: ChatRequest, ctx: _Ctx) -> ChatResponse:
        result = await self._call_upstream(request, ctx.model_id)
        return self._bypass_response(ctx, result)

    def _bypass_response(self, ctx: _Ctx, result: UpstreamResult) -> ChatResponse:
        return ChatResponse(
            text=result.text,
            decision=Decision.BYPASS,
            scope_key=ctx.scope.key,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            skip_reason=ctx.policy.reason,
            timings={"upstream": result.latency_ms},
        )

    def _hit(
        self,
        entry: CacheEntry,
        decision: Decision,
        score: float | None,
        ctx: _Ctx,
        overhead: float,
        timings: dict[str, float],
    ) -> ChatResponse:
        try:
            self.store.record_hit(entry.id, self.now())
        except Exception:
            log.exception("store.record_hit failed")
        return ChatResponse(
            text=entry.answer_text,
            decision=decision,
            score=score,
            entry_id=entry.id,
            scope_key=ctx.scope.key,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cost_usd=overhead,
            overhead_usd=overhead,
            saved_usd=entry.cost_usd,
            timings=dict(timings),
        )

    def _miss(
        self, request: ChatRequest, ctx: _Ctx, look: _Lookup, result: UpstreamResult
    ) -> ChatResponse:
        stored, skip, entry_id = self._maybe_store(request, ctx, look, result)
        return ChatResponse(
            text=result.text,
            decision=look.decision,
            score=look.score,
            entry_id=entry_id,
            scope_key=ctx.scope.key,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd + look.overhead_usd,
            overhead_usd=look.overhead_usd,
            stored=stored,
            skip_reason=skip,
            guard=look.guard,
            timings={**look.timings, "upstream": result.latency_ms},
        )

    def _maybe_store(
        self, request: ChatRequest, ctx: _Ctx, look: _Lookup, result: UpstreamResult
    ) -> tuple[bool, str | None, str | None]:
        if look.vector is None:
            return False, look.error or "no_embedding", None
        skip = store_policy(request.question, result.text, result.stop_reason)
        if skip:
            return False, skip, None
        now = self.now()
        entry = CacheEntry(
            id=str(uuid.uuid4()),
            scope_key=ctx.scope.key,
            namespace=request.namespace,
            model_id=ctx.model_id,
            exact_key=ctx.xkey,
            query_text=request.question,
            query_norm=ctx.norm,
            embedding=look.vector.tolist(),
            answer_text=result.text,
            stop_reason=result.stop_reason or "",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            ttl_class=ctx.policy.ttl_class,
            created_at=now,
            expires_at=now + ctx.policy.ttl_class.ttl,
        )
        try:
            self.store.put(entry)
        except Exception:
            log.exception("store.put failed; answer returned but not cached")
            return False, "store_error", None
        return True, None, entry.id

    def _coalesced(self, leader: ChatResponse) -> ChatResponse:
        """A waiter's view of the leader's result: it made no calls and paid nothing."""
        decision = leader.decision if leader.decision.is_hit else Decision.EXACT_HIT
        saved = (
            leader.saved_usd if leader.decision.is_hit else leader.cost_usd - leader.overhead_usd
        )
        return leader.model_copy(
            update={
                "decision": decision,
                "cost_usd": 0.0,
                "overhead_usd": 0.0,
                "saved_usd": saved,
                "stored": False,
                "skip_reason": None,
                "guard": None,
                "coalesced": True,
                "timings": {},
            }
        )

    async def _aiter_upstream(self, request: ChatRequest, model_id: str):
        """Bridge the upstream's blocking stream iterator into async code via a thread."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        def pump() -> None:
            try:
                for item in self.upstream.stream(request, model_id):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except BaseException as exc:  # handed to the consumer below
                loop.call_soon_threadsafe(queue.put_nowait, _Failure(exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        worker = loop.run_in_executor(None, pump)
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, _Failure):
                self.metrics.upstream_errors.inc()
                raise item.exc
            yield item
        await worker


@dataclass
class _Failure:
    exc: BaseException


def _chunks(text: str, size: int):
    for i in range(0, len(text), max(1, size)):
        yield text[i : i + size]
