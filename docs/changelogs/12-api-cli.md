# Task 12: FastAPI service + CLI
Branch: task/12-api-cli | Date: 2026-11-17

## What changed
- `src/semcache/api.py`: `create_app(cache)` with `POST /v1/chat` (JSON or Server-Sent Events), `GET /metrics`, `GET /healthz`, `POST /admin/invalidate`, `POST /admin/purge-expired`, `GET /admin/stats`. Optional `ADMIN_TOKEN` checked with a constant-time compare. Upstream failures become HTTP 502 without leaking details.
- `src/semcache/cli.py`: `semcache serve [--mock]` and `semcache purge --expired | --namespace | --scope-key | --id`. `--mock` runs with no AWS (hashing embedder + mock LLM) so anyone can try the API.
- `pyproject.toml`: silence a third-party test-client deprecation warning.

## How to verify
- `uv run pytest tests/test_api.py` → 10 passed
- `uv run semcache serve --mock` then `curl -X POST localhost:8000/v1/chat -H "content-type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is KYC?\"}]}"`

## Tests
- Miss then hit over HTTP, 422 on bad input, 502 on upstream error, SSE stream, metrics and health, admin invalidate and stats, admin token (none/wrong/right), CLI parsing. Full suite 153 passed.
