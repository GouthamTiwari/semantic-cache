from semcache.config import CacheConfig
from semcache.embed import HashingEmbedder
from semcache.eval.traffic import build_stream, simulate_engine, simulate_exact, simulate_no_cache

INTENTS = [
    {"intent_id": f"i{k}", "phrasings": [f"how do I do task number {w} quickly {k}" for w in
                                          ("one", "uno", "ein")]}
    for k in range(30)
]  # fmt: skip
SAMPLES = [{"latency_ms": 1000.0, "input_tokens": 80, "output_tokens": 100}]


def test_stream_is_deterministic_with_bursts():
    a = build_stream(INTENTS, n=400, seed=1)
    b = build_stream(INTENTS, n=400, seed=1)
    assert [(e.text, e.t) for e in a] == [(e.text, e.t) for e in b]
    assert len(a) >= 400
    assert sum(1 for e in a if e.burst) >= 20
    assert all(x.t <= y.t for x, y in zip(a, a[1:], strict=False))


def test_baselines_and_engine_accounting():
    events = build_stream(INTENTS, n=300, seed=2)
    none = simulate_no_cache(events, SAMPLES).summary(None)
    base = none["usd_total"]
    exact = simulate_exact(events, SAMPLES).summary(base)
    assert none["hit_rate"] == 0 and none["upstream_calls"] == len(events)
    assert exact["hit_rate"] > 0 and exact["false_hits"] == 0
    assert exact["usd_net_saved_vs_no_cache"] > 0
    sem = simulate_engine(
        "semantic", events, SAMPLES, HashingEmbedder(64),
        CacheConfig(t_high=0.5, t_low=0.5, guards=(), verifier_enabled=False),
    ).summary(base)  # fmt: skip
    # A loose threshold on a non-semantic embedder hits more, and some hits are wrong:
    # exactly what the simulation must be able to detect.
    assert sem["hit_rate"] >= exact["hit_rate"]
    assert sem["false_hits"] > 0
    assert sem["usd_cache_overhead"] > 0
    assert sem["hit_latency_ms_p50"] < sem["miss_latency_ms_p50"]
