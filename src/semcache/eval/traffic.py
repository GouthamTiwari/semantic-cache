"""Traffic simulation: the cost and latency story (SPEC section 12.4).

A synthetic day of requests (Zipf-popular intents, several phrasings each, some bursts of
identical copies) is replayed against five configurations. The semantic configurations run
through the real `SemanticCache` engine with the real Titan vectors from the memo; only the
LLM is a mock, calibrated from real Nova 2 Lite calls. Time is virtual: component latencies
are added up (calibrated embedding and judge latency, measured search time, sampled LLM
latency) instead of slept, so the run takes seconds and repeats exactly.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from semcache.cache import SemanticCache
from semcache.config import CacheConfig, NamespaceConfig
from semcache.eval.calibrate import SIM_SYSTEM, load_intents
from semcache.metrics import Metrics
from semcache.normalize import normalize
from semcache.policy import request_policy
from semcache.pricing import cost_usd
from semcache.store.memory import InMemoryStore
from semcache.types import ChatRequest, Decision
from semcache.upstream import MockUpstream

SEED = 42
N_REQUESTS = 5000
ZIPF_S = 1.1
BURST_SHARE = 0.05
START = datetime(2026, 11, 17, tzinfo=UTC)
NAMESPACE = "sim"


@dataclass
class Event:
    t: datetime
    text: str
    intent: str
    burst: int = 0  # >0: index of the burst this copy belongs to


def build_stream(intents: list[dict], n: int = N_REQUESTS, seed: int = SEED) -> list[Event]:
    rng = random.Random(seed)
    order = list(range(len(intents)))
    rng.shuffle(order)  # popularity rank is random, not file order
    weights = np.array([1 / (rank + 1) ** ZIPF_S for rank in range(len(order))])
    weights /= weights.sum()
    np_rng = np.random.default_rng(seed)
    day = 24 * 3600
    events: list[Event] = []
    n_burst_target = int(n * BURST_SHARE)
    burst_id = 0
    while sum(1 for e in events if e.burst) < n_burst_target:
        burst_id += 1
        k = int(np_rng.integers(5, 21))
        intent = intents[order[int(np_rng.choice(len(order), p=weights))]]
        text = rng.choice(intent["phrasings"])
        t = START + timedelta(seconds=float(np_rng.uniform(0, day)))
        events += [Event(t, text, intent["intent_id"], burst_id) for _ in range(k)]
    while len(events) < n:
        intent = intents[order[int(np_rng.choice(len(order), p=weights))]]
        text = rng.choice(intent["phrasings"])
        t = START + timedelta(seconds=float(np_rng.uniform(0, day)))
        events.append(Event(t, text, intent["intent_id"]))
    events.sort(key=lambda e: (e.t, e.burst))
    return events


@dataclass
class Tally:
    name: str
    decisions: dict[str, int] = field(default_factory=dict)
    false_hits: int = 0
    upstream_calls: int = 0
    tokens: int = 0
    usd_upstream: float = 0.0
    usd_overhead: float = 0.0
    latency_ms: list[float] = field(default_factory=list)
    hit_latency_ms: list[float] = field(default_factory=list)
    miss_latency_ms: list[float] = field(default_factory=list)
    judge_calls: int = 0

    def add(self, decision: str, latency: float, *, hit: bool) -> None:
        self.decisions[decision] = self.decisions.get(decision, 0) + 1
        self.latency_ms.append(latency)
        (self.hit_latency_ms if hit else self.miss_latency_ms).append(latency)

    def summary(self, baseline_usd: float | None) -> dict:
        n = len(self.latency_ms)
        hits = sum(v for k, v in self.decisions.items() if Decision(k).is_hit)
        total = self.usd_upstream + self.usd_overhead

        def p(xs, q):
            return round(float(np.percentile(xs, q)), 1) if xs else None

        out = {
            "config": self.name,
            "requests": n,
            "decisions": dict(sorted(self.decisions.items())),
            "hit_rate": round(hits / n, 4),
            "exact_hits": self.decisions.get("exact_hit", 0),
            "semantic_hits": self.decisions.get("semantic_hit", 0)
            + self.decisions.get("verified_hit", 0),
            "false_hits": self.false_hits,
            "false_hit_share_of_hits": round(self.false_hits / hits, 4) if hits else 0.0,
            "upstream_calls": self.upstream_calls,
            "judge_calls": self.judge_calls,
            "llm_tokens": self.tokens,
            "usd_llm": round(self.usd_upstream, 6),
            "usd_cache_overhead": round(self.usd_overhead, 6),
            "usd_total": round(total, 6),
            "latency_ms_p50": p(self.latency_ms, 50),
            "latency_ms_p95": p(self.latency_ms, 95),
            "hit_latency_ms_p50": p(self.hit_latency_ms, 50),
            "hit_latency_ms_p95": p(self.hit_latency_ms, 95),
            "miss_latency_ms_p50": p(self.miss_latency_ms, 50),
            "miss_latency_ms_p95": p(self.miss_latency_ms, 95),
        }
        if baseline_usd:
            out["usd_net_saved_vs_no_cache"] = round(baseline_usd - total, 6)
            out["net_saving_pct"] = round((baseline_usd - total) / baseline_usd, 4)
        return out


class _Sampler:
    """Draws (latency, input tokens, output tokens) triplets from the calibration samples."""

    def __init__(self, samples: list[dict], seed: int = SEED):
        self.samples = samples
        self.rng = random.Random(seed)
        self.last: dict = samples[0]

    def draw(self) -> dict:
        self.last = self.rng.choice(self.samples)
        return self.last


def _request(text: str) -> ChatRequest:
    return ChatRequest(
        namespace=NAMESPACE,
        system=SIM_SYSTEM,
        messages=[{"role": "user", "content": text}],
        max_tokens=400,
    )


def simulate_no_cache(events: list[Event], samples: list[dict]) -> Tally:
    tally = Tally("no cache")
    sampler = _Sampler(samples)
    for _ in events:
        s = sampler.draw()
        tally.upstream_calls += 1
        tally.tokens += s["input_tokens"] + s["output_tokens"]
        tally.usd_upstream += cost_usd("mock-upstream", s["input_tokens"], s["output_tokens"])
        tally.add("bypass", s["latency_ms"], hit=False)
    return tally


def simulate_exact(events: list[Event], samples: list[dict]) -> Tally:
    """Exact-match cache on normalised text, same policy and TTL rules, no embeddings."""
    tally = Tally("exact only")
    sampler = _Sampler(samples)
    cache: dict[str, tuple[str, datetime]] = {}
    ns = NamespaceConfig()
    for e in events:
        req = _request(e.text)
        pol = request_policy(req, ns)
        key = normalize(e.text)
        t0 = time.perf_counter()
        hit = cache.get(key) if pol.use_cache else None
        lookup_ms = (time.perf_counter() - t0) * 1000
        if hit and hit[1] > e.t:
            tally.add("exact_hit", lookup_ms, hit=True)
            if hit[0] != e.intent:
                tally.false_hits += 1
            continue
        s = sampler.draw()
        tally.upstream_calls += 1
        tally.tokens += s["input_tokens"] + s["output_tokens"]
        tally.usd_upstream += cost_usd("mock-upstream", s["input_tokens"], s["output_tokens"])
        tally.add("miss" if pol.use_cache else "bypass", lookup_ms + s["latency_ms"], hit=False)
        if pol.use_cache:
            cache[key] = (e.intent, e.t + pol.ttl_class.ttl)
    return tally


def simulate_engine(
    name: str, events: list[Event], samples: list[dict], embedder, config: CacheConfig,
    verifier=None,
) -> Tally:  # fmt: skip
    """Replay through the real engine with a virtual clock and a calibrated mock LLM."""
    tally = Tally(name)
    sampler = _Sampler(samples)
    intent_of = {normalize(e.text): e.intent for e in events}
    clock = {"now": START}
    upstream = MockUpstream(
        answer=lambda r: f"answer:{intent_of[normalize(r.question)]}",
        latency_ms=lambda: sampler.draw()["latency_ms"],
        input_tokens=lambda r: sampler.last["input_tokens"],
        output_tokens=lambda: sampler.last["output_tokens"],
        sleep=False,
    )
    cache = SemanticCache(
        store=InMemoryStore(),
        embedder=embedder,
        upstream=upstream,
        verifier=verifier,
        config=config,
        metrics=Metrics(),
        now=lambda: clock["now"],
    )

    def account(e: Event, resp) -> None:
        hit = resp.decision.is_hit
        # Real engine time (lookups, search, bookkeeping) plus the network latencies that the
        # memo and the mock report instead of waiting for: embedding, judge, LLM.
        t = resp.timings
        latency = (
            resp.latency_ms + t.get("embed", 0.0) + t.get("judge", 0.0) + t.get("upstream", 0.0)
        )
        tally.add(resp.decision.value, latency, hit=hit)
        tally.usd_overhead += resp.overhead_usd
        if "judge" in resp.timings and resp.timings["judge"] > 0:
            tally.judge_calls += 1
        if hit:
            if resp.text != f"answer:{e.intent}":
                tally.false_hits += 1
        else:
            tally.upstream_calls += 1
            tally.tokens += resp.input_tokens + resp.output_tokens
            tally.usd_upstream += resp.cost_usd - resp.overhead_usd

    async def run() -> None:
        i = 0
        while i < len(events):
            e = events[i]
            group = [e]
            if e.burst:
                while i + len(group) < len(events) and events[i + len(group)].burst == e.burst:
                    group.append(events[i + len(group)])
            clock["now"] = e.t
            responses = await asyncio.gather(*[cache.aget_or_call(_request(g.text)) for g in group])
            for g, r in zip(group, responses, strict=True):
                account(g, r)
            i += len(group)

    asyncio.run(run())
    return tally


def run_simulation(embedder, op: dict, samples: list[dict], verifier) -> dict:
    intents = load_intents()
    events = build_stream(intents)
    guards = tuple(op["guards"])
    t_high = op["t_high"]
    t_low_verifier = op["t_low"] if op["verifier_enabled"] else round(t_high - 0.04, 2)
    tallies = [
        simulate_no_cache(events, samples),
        simulate_exact(events, samples),
        simulate_engine(
            "semantic (threshold only)", events, samples, embedder,
            CacheConfig(t_high=t_high, t_low=t_high, guards=(), verifier_enabled=False),
        ),
        simulate_engine(
            "semantic + guards", events, samples, embedder,
            CacheConfig(t_high=t_high, t_low=t_high, guards=guards, verifier_enabled=False),
        ),
        simulate_engine(
            "semantic + guards + verifier", events, samples, embedder,
            CacheConfig(t_high=t_high, t_low=t_low_verifier, guards=guards, verifier_enabled=True),
            verifier,
        ),
    ]  # fmt: skip
    baseline = tallies[0].usd_upstream
    return {
        "stream": {
            "requests": len(events),
            "intents": len(intents),
            "distinct_texts": len({normalize(e.text) for e in events}),
            "burst_copies": sum(1 for e in events if e.burst),
            "zipf_s": ZIPF_S,
            "seed": SEED,
            "virtual_duration_hours": 24,
        },
        "operating_point": op,
        "verifier_grey_zone": [t_low_verifier, t_high],
        "results": [t.summary(baseline) for t in tallies],
    }


def write_simulation_report(res: dict, calibration_summary: dict) -> None:
    from semcache.eval.report import pct, table, usd, write_json, write_text

    write_json("simulation.json", res | {"calibration": calibration_summary})
    s = res["stream"]
    rows = [
        [r["config"], pct(r["hit_rate"]), r["exact_hits"], r["semantic_hits"], r["false_hits"],
         r["upstream_calls"], usd(r["usd_total"], 4), usd(r.get("usd_net_saved_vs_no_cache", 0), 4),
         pct(r.get("net_saving_pct", 0)), r["latency_ms_p50"], r["latency_ms_p95"]]
        for r in res["results"]
    ]  # fmt: skip
    lat_rows = [
        [r["config"], r["hit_latency_ms_p50"], r["hit_latency_ms_p95"], r["miss_latency_ms_p50"],
         r["miss_latency_ms_p95"]]
        for r in res["results"]
    ]  # fmt: skip
    c = calibration_summary
    lines = [
        "# Traffic simulation",
        "",
        f"{s['requests']} requests over a virtual {s['virtual_duration_hours']} h: "
        f"{s['intents']} intents (families of near-miss siblings), {s['distinct_texts']} distinct "
        f"phrasings, Zipf popularity (s = {s['zipf_s']}), {s['burst_copies']} burst copies, "
        f"seed {s['seed']}.",
        "",
        f"The LLM is a mock calibrated from {c['calls']} real `global.amazon.nova-2-lite-v1:0` "
        f"calls (latency p50 {c['upstream_latency_ms_p50']} ms, p95 "
        f"{c['upstream_latency_ms_p95']} ms; mean {c['input_tokens_mean']} input and "
        f"{c['output_tokens_mean']} output tokens). Embeddings are real Titan v2 vectors from "
        f"the memo, with their recorded call latency (p50 {c.get('embed_latency_ms_p50')} ms). "
        "Costs use the price table in `src/semcache/pricing.py`.",
        "",
        table(
            [
                "config",
                "hit rate",
                "exact hits",
                "semantic hits",
                "false hits",
                "LLM calls",
                "total USD",
                "net saved USD",
                "net saved",
                "p50 ms",
                "p95 ms",
            ],
            rows,
        ),  # fmt: skip
        "",
        "Latency by outcome (ms):",
        "",
        table(["config", "hit p50", "hit p95", "miss p50", "miss p95"], lat_rows),
        "",
        "Net saved = cost without a cache minus (LLM cost + embedding cost + judge cost). "
        "A miss in a semantic configuration pays for an embedding and a search on top of the "
        "LLM call; that overhead is included.",
        "",
    ]
    write_text("simulation.md", "\n".join(lines))
