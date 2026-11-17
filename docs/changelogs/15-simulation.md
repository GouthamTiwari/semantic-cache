# Task 15: Calibration + traffic simulation
Branch: task/15-simulation | Date: 2026-11-17

## What changed
- `src/semcache/eval/calibrate.py`: at most 50 real Nova 2 Lite calls on sampled traffic questions; saves latency and token samples; summarises upstream and embedding latency.
- `src/semcache/eval/traffic.py`: a seeded synthetic day (5,000 requests, Zipf s = 1.1 over 240 intents, 5 % burst copies). Five configurations: no cache, exact only, semantic, semantic + guards, semantic + guards + verifier. The semantic ones run through the real engine with real Titan vectors from the memo, a virtual clock (TTL applies) and a mock LLM that samples the calibration. Correctness is checked per hit: the served answer must belong to the request's intent.
- `src/semcache/eval/commands.py`: `semcache calibrate --live`, `semcache simulate`, `semcache eval` (regenerate everything).
- `reports/calibration.json`: 40 live calls (p50 1371 ms, p95 2396 ms, ~83 input / ~106 output tokens).

## How to verify
- `uv run pytest tests/test_traffic.py` → 2 passed
- `uv run semcache simulate` after the embedding memo is complete.

## Cost
- Calibration: 40 Nova 2 Lite calls; project live spend after this task: $0.144.
