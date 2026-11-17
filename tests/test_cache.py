import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from semcache.cache import SemanticCache
from semcache.config import CacheConfig, NamespaceConfig
from semcache.embed import EmbedResult
from semcache.normalize import normalize
from semcache.store import InMemoryStore
from semcache.types import ChatRequest, Decision
from semcache.upstream import MockUpstream
from semcache.verify import StaticVerifier


class DictEmbedder:
    """Vectors chosen by the test: texts in the same group share a direction."""

    model_id = "mock-embed"
    dimensions = 8

    def __init__(self, groups: dict[str, tuple[float, ...]], fail: bool = False):
        self.groups = {normalize(k): v for k, v in groups.items()}
        self.fail = fail
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding service down")
        raw = self.groups.get(text)
        if raw is None:  # unknown text: a direction of its own
            raw = tuple(np.random.default_rng(abs(hash(text)) % 2**32).normal(size=8))
        v = np.zeros(8, dtype=np.float32)
        v[: len(raw)] = raw
        v /= np.linalg.norm(v)
        return EmbedResult(v, input_tokens=5, latency_ms=20.0, cost_usd=5 * 0.024 / 1e6)


def req(text, **kw):
    return ChatRequest(messages=[{"role": "user", "content": text}], **kw)


def make_cache(groups=None, *, upstream=None, verifier=None, config=None, embedder=None, now=None):
    return SemanticCache(
        store=InMemoryStore(),
        embedder=embedder or DictEmbedder(groups or {}),
        upstream=upstream or MockUpstream(latency_ms=0.0, sleep=False),
        verifier=verifier,
        config=config or CacheConfig(t_high=0.95, t_low=0.85),
        **({"now": now} if now else {}),
    )


# --- L1 / L2 ---------------------------------------------------------------------


def test_miss_then_exact_hit():
    cache = make_cache()
    first = cache.get_or_call(req("What is the repo rate?"))
    assert first.decision is Decision.MISS and first.stored
    second = cache.get_or_call(req("  what is the REPO rate "))
    assert second.decision is Decision.EXACT_HIT
    assert second.text == first.text
    assert cache.upstream.calls == 1


def test_paraphrase_semantic_hit():
    cache = make_cache({"What is the repo rate?": (1, 0), "Current repo rate?": (1, 0.05)})
    cache.get_or_call(req("What is the repo rate?"))
    r = cache.get_or_call(req("Current repo rate?"))
    assert r.decision is Decision.SEMANTIC_HIT
    assert r.score > 0.95
    assert cache.upstream.calls == 1


def test_low_score_is_a_miss():
    cache = make_cache({"a question": (1, 0), "other question": (0, 1)})
    cache.get_or_call(req("a question"))
    assert cache.get_or_call(req("other question")).decision is Decision.MISS
    assert cache.upstream.calls == 2


def test_guard_rejects_near_identical_vectors():
    same = (1, 0)
    cache = make_cache(
        {"Can I return it within 7 days?": same, "Can I return it within 30 days?": same}
    )
    cache.get_or_call(req("Can I return it within 7 days?"))
    r = cache.get_or_call(req("Can I return it within 30 days?"))
    assert r.decision is Decision.GUARD_REJECT
    assert r.guard == "numbers"
    assert r.text == "Answer to: Can I return it within 30 days?"
    assert cache.upstream.calls == 2


def test_top_k_falls_back_past_a_guard_reject():
    cache = make_cache(
        {
            "refund in 7 days?": (1, 0),
            "refund within a week?": (1, 0.2),
            "refund in seven days?": (1, 0.01),
        }
    )
    cache.get_or_call(req("refund in 7 days?"))
    cache.get_or_call(req("refund within a week?"))  # miss (score below t_low? no: stored)
    r = cache.get_or_call(req("refund in seven days?"))
    assert r.decision is Decision.SEMANTIC_HIT
    assert r.text == "Answer to: refund in 7 days?"


def test_cross_namespace_isolation_l1_and_l2():
    cache = make_cache({"What is the repo rate?": (1, 0), "Current repo rate?": (1, 0.01)})
    cache.get_or_call(req("What is the repo rate?", namespace="tenant-a"))
    # identical text (L1) and paraphrase (L2) from another tenant: both must miss
    r1 = cache.get_or_call(req("What is the repo rate?", namespace="tenant-b"))
    r2 = cache.get_or_call(req("Current repo rate?", namespace="tenant-c"))
    assert r1.decision is Decision.MISS and r2.decision is Decision.MISS
    assert cache.upstream.calls == 3


def test_other_system_prompt_or_model_never_gets_old_answers():
    cache = make_cache()
    cache.get_or_call(req("What is KYC?"))
    assert (
        cache.get_or_call(req("What is KYC?", system="Answer in Hindi")).decision is Decision.MISS
    )
    assert cache.get_or_call(req("What is KYC?", model="other")).decision is Decision.MISS


# --- grey zone ---------------------------------------------------------------------


def grey_cache(answer, enabled=True):
    v = StaticVerifier(lambda a, b: answer)
    cfg = CacheConfig(t_high=0.99, t_low=0.80, verifier_enabled=enabled)
    cache = make_cache({"How do I reset my password?": (1, 0), "Forgot password?": (1, 0.3)},
                       verifier=v, config=cfg)  # fmt: skip
    cache.get_or_call(req("How do I reset my password?"))
    return cache, v, cache.get_or_call(req("Forgot password?"))


def test_grey_zone_verifier_yes():
    cache, v, r = grey_cache(True)
    assert r.decision is Decision.VERIFIED_HIT
    assert v.calls == 1
    assert r.overhead_usd > r.saved_usd * 0  # judge cost counted
    assert "judge" in r.timings


def test_grey_zone_verifier_no():
    _, v, r = grey_cache(False)
    assert r.decision is Decision.MISS and v.calls == 1


def test_grey_zone_verifier_off():
    _, v, r = grey_cache(True, enabled=False)
    assert r.decision is Decision.MISS and v.calls == 0


# --- policy, storage rules, TTL ----------------------------------------------------


def test_no_cache_bypasses_and_stores_nothing():
    cache = make_cache()
    r = cache.get_or_call(req("What is KYC?", no_cache=True))
    assert r.decision is Decision.BYPASS and r.skip_reason == "no_cache_flag"
    assert len(cache.store) == 0
    assert cache.embedder.calls == 0


def test_time_sensitive_uses_volatile_ttl_in_opted_in_namespace():
    t = [datetime(2026, 1, 1, tzinfo=UTC)]
    cfg = CacheConfig(namespaces={"markets": NamespaceConfig(allow_volatile=True)})
    cache = make_cache(config=cfg, now=lambda: t[0])
    cache.get_or_call(req("gold price today?", namespace="markets"))
    t[0] += timedelta(minutes=10)
    assert cache.get_or_call(req("gold price today?", namespace="markets")).decision.is_hit
    t[0] += timedelta(minutes=10)  # 20 minutes > 15-minute volatile TTL
    assert (
        cache.get_or_call(req("gold price today?", namespace="markets")).decision is Decision.MISS
    )


def test_pii_answers_are_not_stored():
    cache = make_cache(upstream=MockUpstream(answer=lambda r: "Call 9876543210", sleep=False))
    r = cache.get_or_call(req("support number?"))
    assert not r.stored and r.skip_reason == "pii"
    assert len(cache.store) == 0
    assert b"semcache_pii_skipped_total 1.0" in cache.metrics.render()


def test_truncated_answers_are_not_stored():
    cache = make_cache(upstream=MockUpstream(stop_reason="max_tokens", sleep=False))
    r = cache.get_or_call(req("long question"))
    assert r.skip_reason == "stop_reason:max_tokens" and len(cache.store) == 0


def test_upstream_error_propagates_and_is_counted():
    cache = make_cache(upstream=MockUpstream(fail_with=RuntimeError("boom"), sleep=False))
    with pytest.raises(RuntimeError):
        cache.get_or_call(req("q"))
    assert b"semcache_upstream_errors_total 1.0" in cache.metrics.render()


# --- fail open ---------------------------------------------------------------------


def test_embedding_failure_fails_open():
    cache = make_cache(embedder=DictEmbedder({}, fail=True))
    r = cache.get_or_call(req("What is KYC?"))
    assert r.decision is Decision.MISS and r.text.startswith("Answer to")
    assert not r.stored and r.skip_reason == "embed_error"


def test_store_failure_fails_open():
    class BrokenStore(InMemoryStore):
        def search(self, *a, **k):
            raise ConnectionError("db down")

        def get_exact(self, *a, **k):
            raise ConnectionError("db down")

        def put(self, *a, **k):
            raise ConnectionError("db down")

    cache = make_cache()
    cache.store = BrokenStore()
    r = cache.get_or_call(req("What is KYC?"))
    assert r.decision is Decision.MISS and r.skip_reason == "store_error"


# --- cost accounting ---------------------------------------------------------------


def test_hit_saves_the_stored_call_cost_minus_embedding():
    cache = make_cache({"What is the repo rate?": (1, 0), "Current repo rate?": (1, 0.05)})
    miss = cache.get_or_call(req("What is the repo rate?"))
    hit = cache.get_or_call(req("Current repo rate?"))
    upstream_cost = 40 * 0.35 / 1e6 + 200 * 2.95 / 1e6
    embed_cost = 5 * 0.024 / 1e6
    assert miss.cost_usd == pytest.approx(upstream_cost + embed_cost)
    assert miss.overhead_usd == pytest.approx(embed_cost)
    assert hit.saved_usd == pytest.approx(upstream_cost)
    assert hit.cost_usd == pytest.approx(embed_cost)
    snap = cache.metrics.snapshot()
    assert snap["usd_net_saved"] == pytest.approx(upstream_cost - 2 * embed_cost, abs=1e-9)


# --- single-flight -------------------------------------------------------------------


def test_single_flight_one_upstream_call_for_50_concurrent_requests():
    upstream = MockUpstream(latency_ms=100.0, sleep=True)
    cache = make_cache(upstream=upstream)

    async def burst():
        return await asyncio.gather(*[cache.aget_or_call(req("What is GST?")) for _ in range(50)])

    results = asyncio.run(burst())
    assert upstream.calls == 1
    assert sum(r.coalesced for r in results) == 49
    assert {r.text for r in results} == {"Answer to: What is GST?"}
    assert sum(r.cost_usd for r in results if r.coalesced) == 0.0


def test_single_flight_waiters_see_leader_errors():
    upstream = MockUpstream(latency_ms=50.0, fail_with=RuntimeError("boom"))
    cache = make_cache(upstream=upstream)

    async def burst():
        return await asyncio.gather(
            *[cache.aget_or_call(req("q")) for _ in range(5)], return_exceptions=True
        )

    results = asyncio.run(burst())
    assert all(isinstance(r, RuntimeError) for r in results)
    assert upstream.calls == 1


# --- streaming -----------------------------------------------------------------------


def collect(cache, request):
    async def run():
        return [e async for e in cache.astream(request)]

    events = asyncio.run(run())
    text = "".join(e["delta"] for e in events if "delta" in e)
    return text, events[-1]["response"], len(events) - 1


def test_stream_miss_then_hit_replay():
    cache = make_cache()
    text, resp, n = collect(cache, req("How do I file a GST return?", stream=True))
    assert resp.decision is Decision.MISS and resp.stored
    assert text == "Answer to: How do I file a GST return?" and n > 1
    text2, resp2, n2 = collect(cache, req("How do I file a GST return?", stream=True))
    assert resp2.decision is Decision.EXACT_HIT and text2 == text and n2 >= 1


def test_unclean_stream_is_not_stored():
    cache = make_cache(upstream=MockUpstream(stop_reason=None, sleep=False))
    _, resp, _ = collect(cache, req("q", stream=True))
    assert not resp.stored and resp.skip_reason == "stop_reason:None"


def test_stream_bypass():
    cache = make_cache()
    text, resp, _ = collect(cache, req("q", no_cache=True, stream=True))
    assert resp.decision is Decision.BYPASS and text == "Answer to: q"


# --- observability --------------------------------------------------------------------


def test_one_json_log_line_per_request(caplog):
    cache = make_cache()
    with caplog.at_level(logging.INFO, logger="semcache.request"):
        cache.get_or_call(req("What is KYC?"))
        cache.get_or_call(req("What is KYC?"))
    lines = [json.loads(r.message) for r in caplog.records if r.name == "semcache.request"]
    assert [entry["decision"] for entry in lines] == ["miss", "exact_hit"]
    assert set(lines[0]) >= {"decision", "score", "entry_id", "latency_ms", "cost_usd", "saved_usd"}


def test_prometheus_series_present():
    cache = make_cache()
    cache.get_or_call(req("What is KYC?"))
    cache.get_or_call(req("What is KYC?"))
    text = cache.metrics.render().decode()
    for series in (
        'semcache_requests_total{decision="miss"} 1.0',
        'semcache_requests_total{decision="exact_hit"} 1.0',
        "semcache_request_latency_seconds_bucket",
        'semcache_usd_total{kind="saved"}',
        'semcache_tokens_total{kind="avoided"} 240.0',
        "semcache_entries_stored_total 1.0",
    ):
        assert series in text
    assert cache.metrics.snapshot()["hit_rate"] == 0.5
