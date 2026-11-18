# semantic-cache

[![CI](https://github.com/GouthamTiwari/semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/GouthamTiwari/semantic-cache/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**An embedding-similarity cache in front of Amazon Bedrock chat calls, tuned to a measured false-hit rate.**

LLM calls take seconds and cost money per token, and users ask the same questions in different words. This cache returns a stored answer when a new question *means the same thing* as one already answered: exact match first, then Titan v2 embeddings in Postgres + pgvector. The hard part is never serving a wrong answer ("refund within 7 days" and "within 30 days" embed almost identically), so the headline metric is the **false-hit rate**: the threshold is tuned on a labelled pair dataset, every match must pass deterministic guards, and savings are reported **after** paying for the embeddings.

## Results

<!-- results:headline -->
- **False-hit rate on the held-out test split: 2.1%** (2 of 93 should-miss pairs; 95 % interval 0.6%-7.5%), at precision 90.9% and recall 25.0%.
- **Same threshold without guards: 20.4% false hits** on the same test pairs.
- **Traffic simulation (5000 requests):** exact match alone saves 83.3% of LLM spend; semantic + guards saves 86.2% after paying for embeddings, but serves 48 wrong answers (1.0% of requests).
- **Latency:** hit p50 0.2 ms vs miss p50 1449.0 ms (p95 73.4 ms vs 2454.2 ms).
- Operating point chosen on the dev split only: Titan v2 at 256 dims, threshold 0.95, all four guards, verifier off.
<!-- /results:headline -->

**How to read this.** On deliberately hard near-miss questions, no similarity threshold alone reaches the 1 % false-hit target; the guards are what make a safe threshold possible, and they cut false hits about tenfold at the same threshold. The price is recall: at this safety level the cache catches a quarter of the reusable paraphrases. On the simulated traffic, most of the savings come from exact repeats; the semantic layer adds a few points, and its wrong answers come almost entirely from relation words the guards do not check ("before" vs "after", "next" vs "last", "A older than B" vs "B older than A"). A relation guard is the obvious next step. It was deliberately not added after seeing these results, so the numbers remain an honest test of the design as it was frozen.

Every number in this README is generated from the committed files in [`reports/`](reports/) by `semcache eval`; nothing is typed by hand. Data: 622 labelled question pairs (finance/regulation, customer support, general knowledge) and a 5,000-request traffic simulation. See [Evaluation](#evaluation) for how they were produced and what they do not show.

## How it works

```mermaid
flowchart TD
    R["POST /v1/chat<br/>namespace, model, system, messages"] --> P{"policy<br/>cache this at all?"}
    P -- "no: no_cache, tool turns,<br/>time-sensitive" --> U["Bedrock Converse<br/>(Nova 2 Lite)"]
    P -- yes --> S["scope key<br/>namespace + model + system prompt + tools<br/>+ temperature + embedding + history"]
    S --> L1{"L1 exact<br/>hash(scope + normalised text)"}
    L1 -- hit --> H["return stored answer"]
    L1 -- miss --> SF["single-flight<br/>identical misses wait for one call"]
    SF --> E["embed (Titan v2)"] --> V["pgvector HNSW top-k<br/>same scope, not expired"]
    V --> T{"similarity"}
    T -- "&ge; t_high" --> G{"guards<br/>numbers, dates,<br/>negation, entities"}
    T -- "grey zone" --> G2{"guards"} --> J{"judge model<br/>same answer?"}
    T -- "&lt; t_low" --> U
    G -- pass --> H
    G -- reject --> U
    J -- yes --> H
    J -- no --> U
    U --> ST["store if clean:<br/>no error, not truncated, no PII; TTL"]
    H --> M["metrics + one JSON log line"]
    ST --> M
```

- **Scope isolation (a security property):** an answer is only reused inside the same namespace, model, system prompt, tool set, temperature band, embedding model and conversation history. Tests prove a tenant never receives another tenant's answer at either cache level.
- **Guards:** numbers ("7" = "seven", "2 lakh" = 200000), dates and financial years, negation polarity, and swapped named entities (RBI vs SEBI) must agree, or the hit is rejected.
- **Grey-zone verifier:** optional judge call for scores just under the threshold; its value is measured, not assumed.
- **Policy:** never caches tool conversations, time-sensitive questions (unless the namespace opts into a 15-minute TTL), errors, refusals, truncated answers, or answers that look like PII.
- **Single-flight:** 50 concurrent identical requests make exactly one LLM call (tested).
- **Streaming:** hits replay as Server-Sent Events; misses stream live and are stored only if the stream finished cleanly.
- **TTL and invalidation:** 7 days / 24 hours / 15 minutes classes, purge job, invalidation by entry, scope or namespace.
- **Fails open:** if embeddings or the database fail, requests still get a live answer.
- **Observability:** Prometheus `/metrics` (decisions, latency histograms, tokens and USD spent/saved/overhead, guard rejections) and one JSON log line per request.

## Quickstart

No AWS account needed for the demo (hashing embedder + mock LLM):

```bash
git clone https://github.com/GouthamTiwari/semantic-cache && cd semantic-cache
uv sync
uv run semcache serve --mock
```

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat -H "content-type: application/json" -d '{"messages":[{"role":"user","content":"What is KYC?"}]}'
```

Send it twice: the second response has `"decision":"exact_hit"` and a non-zero `saved_usd`.

**With real Bedrock:** copy `.env.example` to `.env`, set `AWS_BEARER_TOKEN_BEDROCK` (a Bedrock API key) and `AWS_REGION`, then `uv run semcache serve`. **With Postgres:** `docker compose up -d` and set `DATABASE_URL` as shown in `.env.example`. **Everything in Docker:** `docker compose --profile app up -d --build`.

**As a library**, in front of any answer function (for example a RAG pipeline, see [`examples/rag_adapter.py`](examples/rag_adapter.py)):

```python
from semcache import ChatRequest, SemanticCache

cache = SemanticCache.from_settings()  # Titan + Nova on Bedrock, pgvector if DATABASE_URL is set
r = cache.get_or_call(ChatRequest(namespace="support", messages=[{"role": "user", "content": "How do I reset my password?"}]))
print(r.decision, r.saved_usd, r.text)
```

## Evaluation

### Dataset
622 question pairs labelled `same` (a cached answer is correct) or `different` (a hit would be wrong), in five categories: paraphrases, and four trap categories (near-miss numbers, negation, entity swaps, same topic but a different question). Each trap category also holds "hard same" pairs that keep the trap on both sides ("7 days" vs "seven days"), so guards that block good hits are caught too. 30 seed pairs were written by hand; the rest were generated by Nova 2 Lite over 40 topics and reviewed pair by pair (every change is logged in [`docs/DATASET_REVIEW.md`](docs/DATASET_REVIEW.md)). Pairs are split 70/30 into dev and test **by topic**, so no question appears in both.

### Held-out test result
The operating point was chosen on the dev split only (the lowest threshold whose false-hit rate is at most 1 %), then the test split was run once.

<!-- results:test -->
| config (test split, threshold 0.95) | precision | recall | false-hit rate | false hits |
|---|---|---|---|---|
| with guards (chosen) | 90.9% | 25.0% | 2.1% | 2 |
| without guards | 55.8% | 30.0% | 20.4% | 19 |

By category:

| category | should-miss pairs | false hits | should-hit pairs | recall |
|---|---|---|---|---|
| paraphrase | 0 | 0 | 36 | 16.7% |
| near_miss_number | 22 | 1 | 11 | 72.7% |
| negation | 25 | 1 | 9 | 11.1% |
| entity_swap | 23 | 0 | 12 | 33.3% |
| same_topic_different_question | 23 | 0 | 12 | 8.3% |
<!-- /results:test -->

### Threshold sweep and embedding dimension (dev split)

<!-- results:chart -->
![Threshold sweep](reports/sweep_256.svg)
<!-- /results:chart -->

<!-- results:dims -->
| dims | config | lowest threshold with dev FHR <= 1% | dev recall | dev FHR |
|---|---|---|---|---|
| 256 | threshold only | none | - | - |
| 256 | threshold + guards | 0.95 | 20.1% | 0.8% |
| 1024 | threshold only | none | - | - |
| 1024 | threshold + guards | 0.95 | 20.6% | 0.8% |

Chosen: **256 dims**. Rule fixed before the run: choose 256 only if its dev recall at FHR <= 1% is within 2% of 1024's.
<!-- /results:dims -->

### Guard ablation (dev split)

<!-- results:ablation -->
Dev split, 256 dims, threshold 0.95:

| guards | false hits | removed vs none | false-hit rate | recall | good hits wrongly blocked |
|---|---|---|---|---|---|
| none | 51 | 0 | 20.8% | 25.0% | 0 |
| numbers | 39 | 12 | 15.9% | 24.5% | 1 |
| dates | 49 | 2 | 20.0% | 25.0% | 0 |
| negation | 16 | 35 | 6.5% | 21.1% | 8 |
| entities | 51 | 0 | 20.8% | 24.5% | 1 |
| all | 2 | 49 | 0.8% | 20.1% | 10 |
| all (entities strict) | 2 | 49 | 0.8% | 19.6% | 11 |
<!-- /results:ablation -->

### Grey-zone verifier (dev split)

<!-- results:verifier -->
| grey zone from | verifier | recall | false-hit rate | judge calls | judge cost | judge p50 ms |
|---|---|---|---|---|---|---|
| 0.95 | off | 20.1% | 0.8% | 0 | $0.000000 | 0.0 |
| 0.93 | on | 30.4% | 1.2% | 25 | $0.001442 | 700.7 |
| 0.91 | on | 35.3% | 1.2% | 36 | $0.002080 | 751.5 |
| 0.89 | on | 43.6% | 1.6% | 56 | $0.003227 | 1017.2 |
| 0.87 | on | 48.5% | 1.6% | 66 | $0.003802 | 1428.7 |
| 0.85 | on | 52.4% | 1.6% | 76 | $0.004376 | 1737.3 |

Rule fixed before the run: worth it only if it adds >= 2% dev recall while keeping FHR <= 1%. Result: **not worth it** on this data.
<!-- /results:verifier -->

### Traffic simulation

<!-- results:simulation -->
5000 requests, 240 intents in near-miss families, Zipf popularity, 250 burst copies. LLM mock calibrated from 40 real Nova 2 Lite calls (p50 1371.3 ms); real Titan vectors.

| config | hit rate | exact hits | semantic hits | false hits | LLM calls | total cost | net saving | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| no cache | 0.0% | 0 | 0 | 0 | 5000 | $1.6983 | 0.0% | 1377.7 | 2377.6 |
| exact only | 83.3% | 4166 | 0 | 0 | 834 | $0.2843 | 83.3% | 0.0 | 1485.2 |
| semantic (threshold only) | 86.6% | 3147 | 1184 | 67 | 669 | $0.2279 | 86.6% | 0.3 | 1533.8 |
| semantic + guards | 86.2% | 3243 | 1068 | 48 | 689 | $0.2352 | 86.2% | 0.3 | 1536.6 |
| semantic + guards + verifier | 88.6% | 2754 | 1677 | 92 | 569 | $0.2317 | 86.4% | 0.4 | 2310.2 |
<!-- /results:simulation -->

Net saving = cost with no cache minus (LLM + embedding + judge cost). A miss in a semantic configuration pays for an embedding and a vector search on top of the LLM call; that overhead is included. Latency is accounted rather than slept: recorded Titan and judge call latencies, measured engine time, and LLM latency sampled from 40 real calls ([`reports/calibration.json`](reports/calibration.json)).

### Reproduce
```bash
uv run semcache eval
```
Regenerates every report and the tables above from the on-disk embedding and verdict memos (free and identical on every run). Building the memos from scratch needs Bedrock: `semcache dataset generate`, `semcache embed-memo --live`, `semcache calibrate --live`, then `semcache eval --live`. The whole project's live spend is capped at USD 5 by a budget file and was about USD 0.15.

## Design decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Threshold from a dev-split sweep | similarity scores on short questions are bunched; intuition fails | needs labelled data |
| Deterministic guards before any judge call | free, explainable, catch the most dangerous near-misses | some good paraphrases are blocked (measured above) |
| Entity guard in "swap" mode | strict mode blocks "repo rate?" vs "RBI repo rate?" | one-sided entity additions pass |
| Verifier optional, decided by a rule fixed in advance | an extra LLM call must prove its value | needs judge calls during eval |
| Scope includes system prompt, tools and history | no cross-tenant or cross-prompt leaks | lower hit rate across apps sharing a cache |
| Engine and eval share one decision function | the numbers describe the code that runs | none |
| Fail open | availability beats savings | failures visible only in logs and metrics |
| Single-flight on the exact key | simple and safe | paraphrases in flight at the same moment both pay |
| pgvector, one table per dimension | Postgres is already everywhere; HNSW needs a fixed dimension | one locked connection in v1 (a pool is the next step) |
| Nova 2 Lite as the LLM | the AWS account cannot buy Marketplace models; Claude Haiku is a config change (Converse API) | none for the design |

## What this does not do

- No distributed cache or Redis backend (the `CacheStore` interface allows one later).
- No conversational caching beyond identical history: multi-turn requests only match the exact same earlier turns.
- No coalescing of *different* paraphrases that are in flight at the same moment.
- No fine-tuned embedding model; thresholds are specific to Titan v2 and must be re-swept for another model.
- Guards are lexical. The test split's false hits show their blind spots: an antonym without a negation word ("allowed" vs "prohibited") and a unit change without a number ("in minutes" vs "in seconds").
- The test split is small (93 should-miss pairs), so one false hit moves the rate by about one point; the confidence interval is reported next to the rate.
- No real user traffic: the dataset and the traffic are synthetic (generated, then reviewed), English only.
- No authentication beyond an optional admin token, and no web UI.

## Project layout

```
src/semcache/        engine (cache.py), scope, policy, guards, verifier, single-flight,
                     Bedrock clients, stores (memory, pgvector), metrics, API, CLI
src/semcache/eval/   dataset, generators, sweep and ablations, simulation, reports
data/pairs/          labelled pairs, raw generations, review log
data/traffic/        intent families for the simulation
reports/             every number in this README (JSON + Markdown + SVG)
tests/               unit tests (Bedrock mocked), -m db (Postgres), -m live (real Bedrock)
docs/                spec, task list, changelogs
```

## Development

```bash
uv run pytest            # unit tests, free, no Docker
docker compose up -d && uv run pytest -m db   # store contract tests on Postgres + pgvector
uv run pytest -m live    # a few real Bedrock calls (fractions of a cent)
uv run ruff check && uv run ruff format --check
```

## License

MIT © Goutham Tiwari
