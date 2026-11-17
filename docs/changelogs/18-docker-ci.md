# Task 18: Docker + CI
Branch: task/18-docker-ci | Date: 2026-11-17

## What changed
- `Dockerfile`: uv + Python 3.12 slim image, dependencies cached in their own layer, runs as a non-root user, health check on `/healthz`.
- `.dockerignore`: keeps `.env`, data, reports and the local caches out of the image.
- `docker-compose.yml`: `docker compose up -d` starts only Postgres; `--profile app` adds the API service wired to the database.
- `.github/workflows/ci.yml`: ruff (lint + format), the unit suite (Bedrock mocked), the pgvector contract tests against a Postgres service container, and dataset validation.

## How to verify
- `docker build -t semcache:dev .` then `docker run --rm -p 8001:8000 semcache:dev uv run --no-sync semcache serve --mock --host 0.0.0.0` → `/healthz` returns ok, `/v1/chat` answers (checked locally).
- CI run on GitHub after push.
