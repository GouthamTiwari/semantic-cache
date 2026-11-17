# Task 05: CacheStore interface + InMemoryStore
Branch: task/05-memory-store | Date: 2026-11-17

## What changed
- `src/semcache/store/base.py`: the `CacheStore` protocol (put, get_exact, search, record_hit, invalidate, purge_expired, stats) and the "exactly one selector" rule for invalidation.
- `src/semcache/store/memory.py`: numpy store; one vector matrix per scope, so a search is one matrix product and can never see another scope's rows.
- `tests/test_store.py`: contract tests written once and run against every backend; `tests/helpers.py` builds entries.

## How to verify
- `uv run pytest tests/test_store.py` → 9 passed

## Tests
- Exact get inside scope only, replace on same exact key, sorted top-k, empty scope, expiry ignored then purged, invalidate by id/scope/namespace, refuse ambiguous invalidation, hit counts. Full suite 60 passed.
