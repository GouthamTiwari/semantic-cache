# Task 04: Embedding client with disk memo
Branch: task/04-embedder | Date: 2026-11-17

## What changed
- `src/semcache/bedrock.py`: one place that builds the `bedrock-runtime` client (API key or IAM keys, never both) and retries throttling errors.
- `src/semcache/embed.py`: `TitanEmbedder` (InvokeModel, 256/512/1024 dims, unit vectors, pacing for the 60/min quota), `MemoEmbedder` (SQLite memo keyed by model + dimension + text; can report the original call's cost on a hit for honest eval accounting), `HashingEmbedder` (offline fake for tests and demos only).
- `src/semcache/budget.py`: `LiveBudget` checks the estimated cost before each live call and records the actual cost in `.embed_cache/live_spend.json`.
- `tests/test_live.py`: opt-in live Titan test.

## How to verify
- `uv run pytest tests/test_embed.py` → 10 passed
- `uv run pytest -m live` → 1 passed (two real Titan calls)

## Tests
- Request body shape, unit length, memo hit without a second call and across restarts, dimension in the memo key, pacing with a fake clock, budget cap, auth modes, retry policy. Full suite 51 passed.
