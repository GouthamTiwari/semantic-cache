# Task 06: PgVectorStore
Branch: task/06-pgvector-store | Date: 2026-11-17

## What changed
- `src/semcache/store/schema.sql`: one table per embedding dimension, unique exact-key index, (scope_key, expires_at) index, namespace index, HNSW cosine index.
- `src/semcache/store/pgvector.py`: the same contract as the in-memory store over psycopg 3; upsert on exact key; scope and expiry filtered in SQL; `hnsw.iterative_scan = relaxed_order` so a busy neighbouring scope cannot starve results.
- `tests/test_store.py`: the contract tests now also run against Postgres (`-m db`), plus index and dimension checks.

## How to verify
- `docker compose up -d` then `uv run pytest -m db` → 11 passed

## Tests
- All 9 contract tests pass on both backends. Plain `uv run pytest` stays Docker-free: 60 passed.

## Follow-ups
- One locked connection is fine for v1; a `psycopg_pool` pool is the next step for heavy concurrency.
