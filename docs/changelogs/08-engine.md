# Tasks 08-11: Cache engine, upstream, single-flight, streaming, verifier, metrics
Branch: task/08-engine | Date: 2026-11-17

Built together because all four tasks edit `cache.py`; splitting them would have meant writing the engine three times.

## What changed
- `src/semcache/cache.py`: `SemanticCache` with `get_or_call` / `aget_or_call` / `astream`. Pipeline: policy, scope, normalise, L1 exact, single-flight, L2 top-k search, `evaluate_candidate` per candidate, upstream on a miss, store if allowed. Fails open on embedder or store errors. `from_settings` builds the real Bedrock wiring.
- `src/semcache/decision.py`: `evaluate_candidate` (t_low, guards, t_high, verifier) shared by the engine and the eval.
- `src/semcache/upstream.py`: `BedrockUpstream` (Converse + ConverseStream, budget check) and `MockUpstream` (fixed or sampled latency/tokens, optional real sleep).
- `src/semcache/singleflight.py`: one leader per key; waiters await it with a timeout; errors reach every waiter.
- `src/semcache/verify.py`: `LLMVerifier` (YES/NO judge, fail-safe parsing, memo by unordered normalised pair, disk memo), `StaticVerifier` test double.
- `src/semcache/metrics.py`: Prometheus series, running totals with net savings, one JSON log line per request.
- `src/semcache/embed.py`: embedding latency no longer includes quota-pacing waits.
- `src/semcache/types.py`: `overhead_usd` and per-component `timings` on responses.

## How to verify
- `uv run pytest` → 143 passed
- `uv run pytest -m live` → 3 passed (Titan, Nova 2 Lite sync + stream, judge)

## Tests
- `tests/test_cache.py` (25): L1/L2, guard reject, top-k fallback past a guard reject, cross-namespace isolation at L1 and L2, system prompt and model isolation, grey zone yes/no/off, bypass, volatile TTL with a fake clock, PII and truncation not stored, upstream errors, fail-open embedder and store, cost and net savings, 50 concurrent requests -> 1 upstream call, waiters share leader errors, stream miss/hit/unclean/bypass, JSON log line, Prometheus series.
- `tests/test_upstream_verify.py` (13): Converse request shape, stream parsing, budget block, verdict parsing, pair key, verifier memo.

## Follow-ups
- Paraphrases in flight at the same time are not coalesced (different L1 keys); documented in SPEC 9.1.
