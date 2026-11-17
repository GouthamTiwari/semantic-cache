"""Prometheus metrics, a JSON request log, and running totals (SPEC section 13).

Each `Metrics` object owns its own registry, so tests and simulations never share counters.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Histogram, generate_latest
from prometheus_client import Counter as PromCounter

from semcache.types import ChatResponse, Decision

request_logger = logging.getLogger("semcache.request")

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = PromCounter(
            "semcache_requests", "Requests by cache decision", ["decision"], registry=r
        )
        self.latency = Histogram(
            "semcache_request_latency_seconds",
            "End-to-end request latency by decision",
            ["decision"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = PromCounter(
            "semcache_tokens", "LLM tokens spent upstream or avoided by hits", ["kind"], registry=r
        )
        self.usd = PromCounter(
            "semcache_usd",
            "USD spent upstream, saved by hits, or spent on cache overhead (embeddings, judge)",
            ["kind"],
            registry=r,
        )
        self.guard_rejects = PromCounter(
            "semcache_guard_rejects", "Semantic candidates rejected by a guard", ["guard"],
            registry=r,
        )  # fmt: skip
        self.pii_skipped = PromCounter(
            "semcache_pii_skipped", "Answers not stored because they looked like PII", registry=r
        )
        self.upstream_errors = PromCounter(
            "semcache_upstream_errors", "Upstream calls that raised", registry=r
        )
        self.entries_stored = PromCounter(
            "semcache_entries_stored", "Entries written to the store", registry=r
        )
        self._lock = threading.Lock()
        self.totals: Counter = Counter()

    def observe(self, resp: ChatResponse, *, namespace: str) -> None:
        d = resp.decision.value
        self.requests.labels(d).inc()
        self.latency.labels(d).observe(resp.latency_ms / 1000)
        overhead_usd = resp.overhead_usd
        upstream_usd = max(resp.cost_usd - overhead_usd, 0.0)
        if resp.decision.is_hit:
            self.tokens.labels("avoided").inc(resp.input_tokens + resp.output_tokens)
            self.usd.labels("saved").inc(resp.saved_usd)
        else:  # bypass, miss and guard_reject all paid for an upstream call
            self.tokens.labels("spent").inc(resp.input_tokens + resp.output_tokens)
        self.usd.labels("spent").inc(upstream_usd)
        self.usd.labels("overhead").inc(overhead_usd)
        if resp.guard:
            self.guard_rejects.labels(resp.guard).inc()
        if resp.skip_reason == "pii":
            self.pii_skipped.inc()
        if resp.stored:
            self.entries_stored.inc()
        with self._lock:
            self.totals[f"decision:{d}"] += 1
            self.totals["requests"] += 1
            self.totals["usd_saved"] += resp.saved_usd
            self.totals["usd_upstream"] += upstream_usd
            self.totals["usd_overhead"] += overhead_usd
        log_request(resp, namespace=namespace)

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def snapshot(self) -> dict:
        with self._lock:
            t = dict(self.totals)
        hits = sum(v for k, v in t.items() if k in _HIT_KEYS)
        return {
            "requests": t.get("requests", 0),
            "decisions": {k.split(":", 1)[1]: v for k, v in t.items() if k.startswith("decision:")},
            "hit_rate": round(hits / t["requests"], 4) if t.get("requests") else 0.0,
            "usd_saved": round(t.get("usd_saved", 0.0), 9),
            "usd_upstream": round(t.get("usd_upstream", 0.0), 9),
            "usd_overhead": round(t.get("usd_overhead", 0.0), 9),
            "usd_net_saved": round(t.get("usd_saved", 0.0) - t.get("usd_overhead", 0.0), 9),
        }


_HIT_KEYS = {f"decision:{d.value}" for d in Decision if d.is_hit}


def log_request(resp: ChatResponse, *, namespace: str) -> None:
    """One structured JSON line per request."""
    if not request_logger.isEnabledFor(logging.INFO):
        return
    line = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "decision": resp.decision.value,
        "namespace": namespace,
        "score": None if resp.score is None else round(resp.score, 4),
        "entry_id": resp.entry_id,
        "latency_ms": round(resp.latency_ms, 2),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cost_usd": round(resp.cost_usd, 8),
        "saved_usd": round(resp.saved_usd, 8),
        "guard": resp.guard,
        "coalesced": resp.coalesced,
        "stored": resp.stored,
        "skip_reason": resp.skip_reason,
    }
    request_logger.info(json.dumps(line))
