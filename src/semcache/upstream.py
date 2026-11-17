"""The LLM the cache sits in front of: Bedrock Converse, or a mock for tests and simulation."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from semcache.bedrock import with_retries
from semcache.budget import LiveBudget
from semcache.pricing import cost_usd
from semcache.types import ChatRequest


@dataclass
class UpstreamResult:
    text: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model_id, self.input_tokens, self.output_tokens)


class Upstream(Protocol):
    def complete(self, request: ChatRequest, model_id: str) -> UpstreamResult: ...

    def stream(self, request: ChatRequest, model_id: str) -> Iterator[str | UpstreamResult]:
        """Yield text chunks, then exactly one UpstreamResult (stop_reason None = unclean)."""


def _converse_messages(request: ChatRequest) -> list[dict]:
    # Converse accepts user/assistant turns only; tool turns never reach here (policy P2
    # bypasses them, and the bypass path sends them as user text so nothing is lost).
    out = []
    for m in request.messages:
        role = "assistant" if m.role == "assistant" else "user"
        out.append({"role": role, "content": [{"text": m.content}]})
    return out


class BedrockUpstream:
    """Amazon Bedrock Converse / ConverseStream."""

    def __init__(self, client, *, budget: LiveBudget | None = None, sleep=time.sleep):
        self.client = client
        self.budget = budget
        self.sleep = sleep

    def _kwargs(self, request: ChatRequest, model_id: str) -> dict:
        kw = {
            "modelId": model_id,
            "messages": _converse_messages(request),
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.system:
            kw["system"] = [{"text": request.system}]
        return kw

    def _check_budget(self, request: ChatRequest, model_id: str) -> None:
        if self.budget:
            chars = sum(len(m.content) for m in request.messages) + len(request.system)
            self.budget.check(cost_usd(model_id, chars // 3 + 1, request.max_tokens))

    def complete(self, request: ChatRequest, model_id: str) -> UpstreamResult:
        self._check_budget(request, model_id)
        start = time.perf_counter()
        resp = with_retries(lambda: self.client.converse(**self._kwargs(request, model_id)))
        latency = (time.perf_counter() - start) * 1000
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        # Only text blocks: reasoning blocks are scratch work, not the answer.
        text = "".join(b["text"] for b in blocks if "text" in b).strip()
        usage = resp.get("usage", {})
        result = UpstreamResult(
            text=text,
            stop_reason=resp.get("stopReason"),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            latency_ms=latency,
            model_id=model_id,
        )
        if self.budget:
            self.budget.add(result.cost_usd)
        return result

    def stream(self, request: ChatRequest, model_id: str) -> Iterator[str | UpstreamResult]:
        self._check_budget(request, model_id)
        start = time.perf_counter()
        resp = with_retries(lambda: self.client.converse_stream(**self._kwargs(request, model_id)))
        parts: list[str] = []
        stop_reason = None
        usage: dict = {}
        for event in resp["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {}).get("text")
                if delta:
                    parts.append(delta)
                    yield delta
            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")
            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
        result = UpstreamResult(
            text="".join(parts).strip(),
            stop_reason=stop_reason,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            latency_ms=(time.perf_counter() - start) * 1000,
            model_id=model_id,
        )
        if self.budget:
            self.budget.add(result.cost_usd)
        yield result


class MockUpstream:
    """Deterministic stand-in for Bedrock.

    `latency_ms` and the token counts can be fixed numbers or callables (the simulation
    samples them from the live calibration). With `sleep=False` the latency is *reported*
    but not waited for, so a 5,000-request simulation runs in seconds.
    """

    def __init__(
        self,
        *,
        answer: Callable[[ChatRequest], str] | None = None,
        latency_ms: float | Callable[[], float] = 800.0,
        input_tokens: int | Callable[[ChatRequest], int] = 40,
        output_tokens: int | Callable[[], int] = 200,
        stop_reason: str = "end_turn",
        sleep: bool = True,
        model_id: str = "mock-upstream",
        fail_with: Exception | None = None,
        seed: int = 0,
    ):
        self.answer = answer or (lambda r: f"Answer to: {r.question}")
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason
        self.sleep = sleep
        self.model_id = model_id
        self.fail_with = fail_with
        self.rng = random.Random(seed)
        self.calls = 0
        self._lock = threading.Lock()

    def _value(self, v, *args):
        return v(*args) if callable(v) else v

    def complete(self, request: ChatRequest, model_id: str) -> UpstreamResult:
        with self._lock:
            self.calls += 1
        if self.fail_with:
            raise self.fail_with
        latency = float(self._value(self.latency_ms))
        if self.sleep:
            time.sleep(latency / 1000)
        return UpstreamResult(
            text=self.answer(request),
            stop_reason=self.stop_reason,
            input_tokens=int(self._value(self.input_tokens, request)),
            output_tokens=int(self._value(self.output_tokens)),
            latency_ms=latency,
            model_id=self.model_id,
        )

    def stream(self, request: ChatRequest, model_id: str) -> Iterator[str | UpstreamResult]:
        result = self.complete(request, model_id)
        words = result.text.split(" ")
        for i, w in enumerate(words):
            yield w if i == 0 else " " + w
        yield result
