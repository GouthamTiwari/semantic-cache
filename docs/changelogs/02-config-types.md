# Task 02: Config + domain types
Branch: task/02-config-types | Date: 2026-11-17

## What changed
- `src/semcache/types.py`: `ChatRequest` (validates that the last message is a non-empty user message), `ChatResponse`, `CacheEntry`, `Decision`, `TTLClass` with the three TTL durations.
- `src/semcache/config.py`: `Settings` from `.env` (credentials, model IDs, database URL, budget) and `CacheConfig` (thresholds, guards, top-k, verifier switch) with validation; `NamespaceConfig` for per-tenant TTL policy; `SCHEMA_VERSION`.
- `src/semcache/pricing.py`: price table for Titan v2, Nova 2 Lite and the mocks, dated 2026-11-16; unknown models raise.

## How to verify
- `uv run pytest tests/test_config_types.py` → 8 passed

## Tests
- `tests/test_config_types.py`: settings from env, TTLs, decision helper, config validation, request validation, cost maths, unknown price, expiry. Full suite 9 passed.

## Self-review
- No secrets in defaults; `SecretStr` keeps keys out of reprs and logs.
