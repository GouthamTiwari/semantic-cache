# Tasks

Each task is built on its own branch `task/NN-slug`, tested, squash-merged into `main` with one conventional commit, and logged in `docs/CHANGELOG.md`. "Deps" lists tasks that must be merged first.

| # | Task | Files | Deps | Acceptance criteria | Why |
|---|---|---|---|---|---|
| 01 | Repo setup | `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `LICENSE`, `README.md` (skeleton), `docker-compose.yml`, `docker/init.sql`, `CLAUDE.md`, `.githooks/` | none | `uv run pytest` runs; `uv run ruff check` clean; hooks active (`git config core.hooksPath` = `.githooks`); `.env` ignored; MIT licence holder Goutham Tiwari | everything else builds on it |
| 02 | Config + domain types | `config.py`, `types.py`, `pricing.py` | 01 | settings load from `.env`; both-auth error; TTL classes; cost of a known call matches hand maths; unknown model raises | one source of truth for settings and shapes |
| 03 | Scope, normalisation, policy | `scope.py`, `normalize.py`, `policy.py` | 02 | each policy rule P1–P6 has a test; scope changes when any field changes; normalisation keeps numbers and negations | the cheapest safety: never look in the wrong place |
| 04 | Embedder + disk memo | `embed.py`, `bedrock.py` | 02 | Titan request body correct (mocked client); memo returns the same vector without a second call; pacing works with a fake clock; budget guard blocks overspend | eval never pays twice |
| 05 | Store interface + in-memory store | `store/base.py`, `store/memory.py` | 02 | exact get, top-k search sorted by score, scope filter, expiry ignored, invalidate by scope/namespace/id, purge | tests and eval need a fast store |
| 06 | PgVector store | `store/pgvector.py`, `store/schema.sql` | 05 | same contract tests as 05 pass against Docker Postgres (`-m db`); HNSW index exists | production storage |
| 07 | Guards | `guards.py`, `tests/test_guard_*.py` | 03 | one test file per guard incl. seed traps; swap vs strict entity mode | catches what embeddings blur |
| 08 | Cache engine (built together with 09-11: they share `cache.py`) | `cache.py`, `decision.py` | 03–05, 07 | L1, L2, thresholds, top-k fallback past a guard reject, cross-namespace isolation test at L1 and L2, fail-open on embed/store errors | the core product |
| 09 | Upstream, single-flight, streaming | `upstream.py`, `singleflight.py`, `cache.py` | 08 | 50 concurrent identical requests → 1 upstream call; hit streams replay; miss stream stored only on clean stop | stampede protection and real-time UX |
| 10 | Verifier | `verify.py` | 08 | YES/NO parsing fail-safe; verdict memo; off switch; grey-zone path in engine | measured, optional extra precision |
| 11 | Metrics, logs, cost | `metrics.py`, `logs.py` | 08 | one JSON line per request; Prometheus text has all series; saved minus embedding cost | the numbers a platform team reads |
| 12 | FastAPI service + CLI | `api.py`, `cli.py` | 09–11 | `/v1/chat` sync + SSE, `/metrics`, `/healthz`, admin invalidate/stats with optional token, `semcache purge --expired` | drop-in service |
| 13 | Pair dataset | `eval/dataset.py`, `eval/generate.py`, `data/pairs/*` | 04 | ~600 valid pairs, ~50 % same, group split with no leakage, review sample written | ground truth for every number |
| 14 | Sweep, ablations, verifier eval | `eval/pairs_eval.py`, `eval/report.py`, `reports/*` | 07, 10, 13 | sweep for 256 + 1024; operating point chosen on dev only; test run once; ablation and verifier tables; SVG chart | the honest result |
| 15 | Calibration + traffic simulation | `eval/calibrate.py`, `eval/traffic.py`, `data/traffic/*`, `reports/*` | 08–11, 14 | ≤ 50 live calls; 5 configs; hit rates, false hits, p50/p95, net USD after embedding cost | the cost and latency story |
| 16 | README | `README.md`, `eval/readme_sync.py` | 14, 15 | pitch, Mermaid diagram, 3-command quickstart, results from `reports/` only, trade-offs, what it does not do | the 90-second recruiter read |
| 17 | RAG adapter example | `examples/regrag_adapter.py` | 08 | runs with a fake answer function; no hard dependency | shows drop-in use |
| 18 | Docker + CI + polish | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | all | CI green (ruff + pytest, Bedrock mocked); audit prints OK; repo topics and description set | proof it runs anywhere |

## v2 (after v1 was frozen at tag v1.0.0)

| # | Task | Files | Acceptance criteria | Why |
|---|---|---|---|---|
| 19 | Relation guard | `guards.py`, `tests/test_guard_relation.py` | v1 simulation false hits rejected; same-relation paraphrases pass | v1's remaining wrong answers were relation words |
| 20 | v2 dataset | `eval/dataset.py`, `eval/generate.py`, `data/pairs/*` | relation pairs in dev; v1 test retired; fresh test on unseen topics; no text shared with dev or retired | a seen test split cannot test v2 |
| 21 | v2 evaluation | `reports/*`, `README.md`, `config.py` | dev-only tuning; fresh test run once; simulation re-run; README v1 → v2 table generated; defaults equal the operating point | measure the change honestly |
