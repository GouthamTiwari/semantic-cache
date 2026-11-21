# Semantic Cache for LLM Apps: specification

Status: approved for build, 2026-11-17. Owner: Goutham Tiwari.

This document turns the project brief into precise behaviour. Every non-obvious decision has a one-line *why*. Where this spec departs from the original brief, the departure is listed in section 12.

---

## 1. Goal

A Python library plus a small FastAPI service that sits in front of Amazon Bedrock chat calls and returns a stored answer when a new question *means the same thing* as one already answered, while keeping the **false-hit rate** (answers returned to questions that should have missed) at or below a stated target. Hit rate and savings only count at a false-hit rate we can defend.

Users:
- an application developer who wraps an LLM call with `SemanticCache.get_or_call(...)`;
- a platform team that runs the HTTP service and reads `/metrics`;
- a reviewer (recruiter, interviewer) who reads the README numbers and wants to know where they came from.

---

## 2. Bedrock models, region and prices

Checked 2026-11-16 against the AWS model catalogue and the public AWS Price List for `ap-south-1`, and confirmed with live calls from the same AWS account (used by the sibling Regulatory RAG project).

| Role | Model ID | Region | Price (USD) |
|---|---|---|---|
| Embeddings | `amazon.titan-embed-text-v2:0` (dims 256, 512 or 1024; `normalize: true`) | ap-south-1 (Mumbai) | 0.024 per 1M input tokens |
| Answering model (upstream) | `global.amazon.nova-2-lite-v1:0` (global cross-region inference profile) | ap-south-1 | 0.35 per 1M input, 2.95 per 1M output |
| Grey-zone judge | same Nova 2 Lite profile | ap-south-1 | same |

- **Why Nova 2 Lite and not Claude Haiku:** the account is billed through AWS India (AISPL), which cannot buy AWS Marketplace models (Claude, Cohere) by card. Nova 2 Lite is the same "small, fast, cheap" class. Claude Haiku is the documented upgrade path; switching is a change to `BEDROCK_CHAT_MODEL_ID` because the code uses the model-agnostic Converse API.
- **APIs:** chat uses `Converse` and `ConverseStream` (same request shape for every model). Embeddings use `InvokeModel` with body `{"inputText", "dimensions", "normalize"}`; the response gives `embedding` and `inputTextTokenCount`.
- **Quotas:** Titan in Mumbai is limited to 60 requests per minute on this account, so the embedder is paced at 1 request per second by default. Every eval text is embedded once and kept in an on-disk memo.
- **Auth, pick one:** a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK` (simplest), or IAM keys in `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`. Setting both is an error (a beginner cannot tell which one is used).
- The price table lives in `src/semcache/pricing.py` with `PRICES_CHECKED_ON`. An unknown model ID raises instead of silently costing $0.

---

## 3. Request flow

```
request (namespace, model, system, messages, params)
  1. policy        cache at all?            no  -> bypass: straight to upstream, nothing stored
  2. scope         build the scope key      (section 4)
  3. normalize     text used for matching   (section 5)
  4. L1 exact      hash(scope + normalized) found and not expired -> exact_hit
  5. L2 semantic   embed, nearest neighbour inside the SAME scope (top-k)
                   score >= t_high              -> candidate
                   t_low <= score < t_high      -> grey zone: candidate only if the verifier says "same"
                   score < t_low                -> miss
  6. guards        numbers, dates, negation, entities must agree   else -> guard_reject (treated as a miss)
  7. single-flight identical misses in flight wait for the first one
  8. upstream      Bedrock Converse (sync or streamed)
  9. store + log   store if the answer is cacheable; one JSON log line per request
```

### 3.1 Inputs (`ChatRequest`)
| Field | Type | Default | Notes |
|---|---|---|---|
| `namespace` | str | `"default"` | tenant or app; part of the scope |
| `model` | str or null | settings chat model | part of the scope |
| `system` | str | `""` | hashed into the scope |
| `messages` | list of `{role, content}` | required | roles `user`, `assistant`, `tool`; last message must be `user` |
| `tools` | list or null | null | tool definitions; hashed into the scope |
| `temperature` | float | 0.0 | bucketed into the scope |
| `max_tokens` | int | 512 | |
| `stream` | bool | false | |
| `no_cache` | bool | false | forces bypass |

### 3.2 Output (`ChatResponse`)
`text`, `decision` (`bypass | exact_hit | semantic_hit | verified_hit | miss | guard_reject`), `score` (best similarity or null), `entry_id` (matched or newly stored), `scope_key`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd` (spent on this request: embedding + judge + upstream), `saved_usd` (for hits: the stored call's cost), `stored` (bool), `skip_reason` (why not stored, if not).

### 3.3 Error cases
- Upstream error: the error propagates to the caller (HTTP 502 in the API); nothing is stored; counted in `semcache_upstream_errors_total`.
- Embedding error: the cache **fails open**: it logs the error and calls upstream as a miss (why: the cache must never make the app less available than it was without it).
- Store error (database down): fail open in the same way.
- Last message not from `user`, or empty messages: HTTP 422 / `ValueError`.

---

## 4. Scope (security property)

An entry is returned only to a request with the **same scope key**:

```
scope_key = sha256( namespace | model_id | sha256(system) | sha256(tools JSON, sorted keys)
                    | temperature_bucket | embed_model_id | embed_dim | history_hash | SCHEMA_VERSION )
```

- `temperature_bucket`: `t0` (0), `low` (≤ 0.5), `mid` (≤ 1.0), `high` (> 1.0). Why: answers sampled at very different temperatures are different products.
- `history_hash`: sha256 of every message before the last user message (empty string for single-turn). See 6.2.
- `SCHEMA_VERSION = 1`: bump it when the stored format or meaning changes, and every old entry becomes unreachable at once.
- The scope key is stored on every row and every lookup filters on it **before** similarity. A test proves that an identical question in namespace B never receives namespace A's answer, at L1 or L2.

---

## 5. Normalisation

Used only for matching; the original text is stored and returned as-is.
1. Unicode NFKC, lowercase, trim, collapse whitespace.
2. Strip leading filler phrases: "please", "hi", "hello", "hey", "can you tell me", "could you tell me", "tell me", "i want to know", "i would like to know", "do you know".
3. Strip trailing punctuation (`?`, `.`, `!`).

The L1 key is `sha256(scope_key + "\n" + normalized_text)`. The embedding is computed from the normalized text. Numbers, negations and names are never stripped (why: they are exactly what the guards compare).

---

## 6. Cache policy

### 6.1 Rules (each has a test)
| # | Rule | When | Result |
|---|---|---|---|
| P1 | `no_cache` flag | request field is true | bypass |
| P2 | tool traffic | any message has role `tool`, or the request carries `tools` and the model returns tool use | bypass |
| P3 | time-sensitive | last user message matches `today, now, tonight, tomorrow, yesterday, latest, right now, this week, live, at the moment, currently, current` (word-boundary match; `current account` is excluded because it is a bank product) | bypass, **unless** the namespace allows the `volatile` TTL class; then cache with 15-minute TTL |
| P4 | upstream refused or failed | exception, empty text, `stopReason` not `end_turn` (for example `max_tokens`, `guardrail_intervened`, `content_filtered`, `tool_use`) | do not store |
| P5 | PII-looking content | question or answer contains an email address, a phone number (Indian 10-digit mobile or `+` international), or a run of 9+ digits (spaces/hyphens allowed inside) | do not store; `semcache_pii_skipped_total` + 1 |
| P6 | stream not finished | client disconnected or stream ended without a clean stop | do not store |

### 6.2 Multi-turn decision
**Decision:** multi-turn requests are cached, but the hash of every earlier turn is part of the scope (section 4). Why: a follow-up like "and for a car loan?" means nothing on its own; with the history in the scope it can only ever match another conversation with the *identical* history, which is safe. The cost is a low hit rate on multi-turn traffic, which we accept. Conversational caching beyond this rule is a non-goal.

### 6.3 TTL classes
| Class | TTL | Use |
|---|---|---|
| `static` | 7 days | stable facts (definitions, how-to) |
| `normal` | 24 hours | default |
| `volatile` | 15 minutes | time-sensitive questions, only in namespaces that opt in |

Namespace config (`namespaces.json` or code): `default_ttl` and `allow_volatile`. Unknown namespaces get `normal`, `allow_volatile = false`.

---

## 7. Guards (deterministic safety net)

Run on (new query, cached query) for every semantic candidate, after the score check. Each is a pure function `(a, b) -> bool (passes)` with its own test file.

| Guard | Rejects when | Detail |
|---|---|---|
| `numbers` | the multisets of numeric values differ | digits (`7`, `1,000`, `2.5`), `%`, and number words one–twenty, thirty … hundred, thousand, lakh, crore are mapped to values; years are left to the `dates` guard |
| `dates` | the sets of years, month names, weekday names or explicit dates differ | years 1900–2099, `dd/mm/yyyy`, `yyyy-mm-dd`, `FY24`, `2023-24` |
| `negation` | the negation polarity differs | cues: not, no, never, cannot, can't, won't, isn't, aren't, don't, doesn't, didn't, without, nor, neither, n't; polarity = cue count mod 2 |
| `entities` | **swap mode (default):** each side has a key term the other lacks. **strict mode:** either side has a key term the other lacks | key terms: ALL-CAPS tokens of 2+ letters (GST, RBI), capitalised words that are not sentence-initial, quoted terms; compared case-insensitively |

Why swap mode is the default: strict mode rejects common paraphrases such as "repo rate?" vs "RBI repo rate?". The ablation reports both so the trade-off is visible.

---

## 8. Grey-zone verifier

For candidates with `t_low <= score < t_high`, ask the judge model:

> System: You decide whether two user questions can be answered by exactly the same answer. Differences in numbers, dates, negation, products, places or organisations mean NO. Different wording of the same question means YES. Reply with one word: YES or NO.
> User: Question A: … Question B: …

- `maxTokens` 5, temperature 0. Anything other than a clear YES is NO (why: fail safe).
- Verdicts are memoised by the unordered pair of normalized texts (in memory and on disk), so the same pair is never judged twice.
- The verifier can be switched off (`verifier_enabled = false`); then the grey zone is a miss.
- Its cost and latency are added to the request's cost and latency.

---

## 9. Concurrency and streaming

### 9.1 Single-flight
Key: the L1 key. The first miss for a key becomes the leader and calls upstream; later identical misses await the leader's result (up to `singleflight_timeout_s = 30`; after that a waiter calls upstream itself, why: a stuck leader must not hang everyone). Waiters get decision `exact_hit` with `coalesced = true` in the log. Test: 50 concurrent identical requests produce exactly 1 upstream call.

Paraphrases in flight at the same time are not coalesced (they have different L1 keys). Streaming misses do not coalesce (each streams live). Both are documented limitations.

### 9.2 Streaming
- Hit: the stored text is replayed in chunks of about 40 characters.
- Miss: the upstream `ConverseStream` deltas are forwarded as they arrive and buffered; the entry is stored only after `messageStop` with `stopReason = end_turn` and the policy checks pass.
- HTTP: Server-Sent Events. Each chunk is `data: {"delta": "..."}`; the last event is `data: {"done": true, ...metadata}`.

---

## 10. Storage

### 10.1 Interface
```python
class CacheStore(Protocol):
    def put(self, entry: CacheEntry) -> None
    def get_exact(self, scope_key: str, exact_key: str, now: datetime) -> CacheEntry | None
    def search(self, scope_key: str, vector: Sequence[float], k: int, now: datetime) -> list[tuple[CacheEntry, float]]
    def record_hit(self, entry_id: str, now: datetime) -> None
    def invalidate(self, *, scope_key=None, namespace=None, entry_id=None) -> int
    def purge_expired(self, now: datetime) -> int
    def stats(self) -> dict
```
Implementations: `InMemoryStore` (numpy, used by tests and the eval) and `PgVectorStore`.

### 10.2 Postgres schema (one table per embedding dimension)
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS semcache_entries_{dim} (
    id            uuid PRIMARY KEY,
    scope_key     text        NOT NULL,
    namespace     text        NOT NULL,
    model_id      text        NOT NULL,
    exact_key     text        NOT NULL,
    query_text    text        NOT NULL,
    query_norm    text        NOT NULL,
    embedding     vector({dim}) NOT NULL,
    answer_text   text        NOT NULL,
    stop_reason   text        NOT NULL,
    input_tokens  integer     NOT NULL,
    output_tokens integer     NOT NULL,
    cost_usd      double precision NOT NULL,
    ttl_class     text        NOT NULL,
    created_at    timestamptz NOT NULL,
    expires_at    timestamptz NOT NULL,
    hit_count     integer     NOT NULL DEFAULT 0,
    last_hit_at   timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS semcache_entries_{dim}_exact ON semcache_entries_{dim} (exact_key);
CREATE INDEX IF NOT EXISTS semcache_entries_{dim}_scope ON semcache_entries_{dim} (scope_key, expires_at);
CREATE INDEX IF NOT EXISTS semcache_entries_{dim}_ns ON semcache_entries_{dim} (namespace);
CREATE INDEX IF NOT EXISTS semcache_entries_{dim}_hnsw ON semcache_entries_{dim}
    USING hnsw (embedding vector_cosine_ops);
```
- Why one table per dimension: a pgvector HNSW index needs a fixed dimension, and the dimension is already part of the scope.
- Search: `ORDER BY embedding <=> $v` with `WHERE scope_key = $s AND expires_at > $now LIMIT k`; similarity = `1 - distance`. The session sets `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) so the scope filter cannot starve the result list.
- Writing an entry whose `exact_key` already exists replaces it (upsert).
- Postgres runs in Docker (`pgvector/pgvector:pg16`) on host port **5434** (5433 is taken by the Regulatory RAG database).

---

## 11. Parameters (starting values, replaced by the sweep)

| Parameter | Start | Why |
|---|---|---|
| `t_high` | 0.90 | high enough that unrelated short questions rarely pass; **replaced by the dev sweep** |
| `t_low` | 0.85 | grey zone of 0.05; **replaced by the dev sweep** |
| embedding dim | 1024 | Titan default; 256 must justify itself with numbers |
| top-k | 3 | check the next candidates when the best one is guard-rejected |
| single-flight timeout | 30 s | longer than a normal Nova answer, short enough not to hang |
| guards | all four, entities in swap mode | |
| verifier | on in the grey zone | only if the eval says it is worth it |
| false-hit target | ≤ 1.0 % on all `different` pairs | from the brief |

The final values are written to `reports/operating_point.json` by `semcache sweep`, and a test checks that `CacheConfig` defaults equal them.

---

## 12. Evaluation

### 12.1 Pair dataset (`data/pairs/pairs.jsonl`)
```json
{"id":"p0001","a":"...","b":"...","label":"same|different","category":"...","split":"dev|test","group":"g017","source":"seed|generated","domain":"finance|support|general"}
```
- Categories: `paraphrase` (all `same`), and four trap categories `near_miss_number`, `negation`, `entity_swap`, `same_topic_different_question`, each holding mostly `different` pairs **plus "hard same" pairs** that keep the trap feature on both sides (for example "7 days" vs "seven days", "not refundable" vs "can't be refunded"). Why: without hard-same pairs the guards' false rejections would be invisible.
- Target ≈ 600 pairs, ≈ 50 % `same`: paraphrase 120; each trap category 120 (75 different + 45 same).
- Growth: the 30 seed pairs are the style guide. `semcache dataset generate` asks Nova 2 Lite for pairs per (topic, category), validates them (schema, no duplicates, `a != b`, length limits), and assigns a `group` per topic.
- **Split rule (no leakage):** split by `group`, never by pair: about 70 % of groups `dev`, 30 % `test`, chosen by a stable hash. A test proves no text appears in both splits.
- Human check: `semcache dataset review-sample` writes a random 10 % sample to `data/pairs/review_sample.csv`; the owner's corrections are recorded in `docs/DATASET_REVIEW.md`. Until that review is done, the README says so.
- Domains: regulatory/finance FAQs, customer support FAQs, general short factual questions. All invented or public; no employer data.
- Public paraphrase corpora: **Quora Question Pairs is not used.** Its terms restrict it to non-commercial use and do not clearly allow redistribution, so committing it is out; downloading at run time adds a dependency for little gain, because our trap categories are the point.

### 12.2 Metrics (exact formulas)
For a configuration and threshold, each pair is predicted `hit` or `miss` (b is the new query, a the cached one):
- TP = hit and label `same`; FP = hit and label `different`; FN = miss and label `same`; TN = miss and `different`.
- **precision** = TP / (TP + FP)
- **recall (hit rate on should-hit pairs)** = TP / (TP + FN)
- **false-hit rate (FHR)** = FP / (FP + TN), i.e. over all `different` pairs; reported overall and per category.
- **guard false-rejection rate** = same pairs above threshold that a guard rejects / same pairs above threshold.

Traffic simulation metrics:
- hit rate = hits / requests (exact and semantic separately);
- false hits = semantic hits whose stored intent ≠ request intent;
- **net savings USD** = Σ over hits of the stored call's cost − Σ over all cache-enabled requests of (embedding cost + judge cost);
- latency p50 / p95 by decision, where latency = calibrated embedding latency + measured search time + judge latency (if used) + mock upstream latency (misses).

### 12.3 Experiments
1. Threshold sweep 0.70 → 0.99, step 0.01, on `dev`: precision, recall, FHR, per category; for "threshold only" and "threshold + guards".
2. Embedding dimension 256 vs 1024 (both swept).
3. Guard ablation at the chosen threshold: none, each guard alone, all guards, entities strict.
4. Verifier: off vs on for the grey zone: recall gained, FHR, judge calls, cost and latency added.
5. **Operating point:** the lowest threshold whose dev FHR ≤ 1 % with all guards (verifier off), which maximises recall. Then one run on `test`, reported as is; no retuning afterwards.
6. Traffic simulation (section 12.4) at the operating point.

### 12.4 Traffic simulation
- `data/traffic/intents.jsonl`: intent families. Each family is a topic with 2–3 sibling intents that are near misses of each other (different number, entity or polarity), and each intent has 4–5 phrasings. Generated with Nova 2 Lite, then validated.
- Stream: a fixed seed, about 5,000 requests; intent popularity follows a Zipf law (s = 1.1); the phrasing is drawn uniformly; 5 % of requests arrive in bursts of 5–20 identical copies (to exercise single-flight).
- Upstream: a mock with latency and token counts sampled from `reports/calibration.json`, which `semcache calibrate --live` measures from ≤ 50 real Nova 2 Lite calls (and ≤ 50 Titan calls for embedding latency).
- Configurations: no cache, exact only, semantic (threshold only), semantic + guards, semantic + guards + verifier.

### 12.5 Numbers rule
Every number in the README comes from a committed file in `reports/` (`*.json` + `*.md` + `*.svg`) that `semcache eval` regenerates from the embedding memo. `semcache readme-sync` copies the report tables into the README between markers, so nothing is typed by hand.

### 12.6 Budget
Live calls are opt-in (`--live`, `pytest -m live`) and the whole project is capped at USD 5. Every real call adds its estimated cost to `.embed_cache/live_spend.json`; a call that would exceed `LIVE_BUDGET_USD` raises before it is made.

---

## 13. Observability

One JSON log line per request (logger `semcache.request`):
```json
{"ts":"...","decision":"semantic_hit","namespace":"default","score":0.9431,"entry_id":"...","latency_ms":41.2,
 "input_tokens":38,"output_tokens":212,"cost_usd":0.0000004,"saved_usd":0.000639,"guard":null,"coalesced":false,"stored":false,"skip_reason":null}
```
`/metrics` (Prometheus text): `semcache_requests_total{decision}`, `semcache_request_latency_seconds{decision}` (histogram), `semcache_tokens_total{kind="spent|avoided"}`, `semcache_usd_total{kind="spent|saved|overhead"}`, `semcache_guard_rejects_total{guard}`, `semcache_pii_skipped_total`, `semcache_upstream_errors_total`, `semcache_entries_stored_total`.

---

## 14. HTTP API

| Method | Path | Body / result |
|---|---|---|
| POST | `/v1/chat` | `ChatRequest` → `ChatResponse`, or SSE when `stream: true` |
| GET | `/metrics` | Prometheus text |
| GET | `/healthz` | `{"status":"ok","store":"memory|pgvector"}` |
| POST | `/admin/invalidate` | exactly one of `scope_key`, `namespace`, `entry_id` → `{"deleted": n}` |
| GET | `/admin/stats` | entry counts, hit counts, decision counts, USD saved |

If `ADMIN_TOKEN` is set, admin endpoints require header `X-Admin-Token`. (Full auth is a non-goal; this stops accidental use.)

CLI: `semcache serve`, `semcache purge --expired`, `semcache dataset generate|validate|review-sample`, `semcache embed-memo`, `semcache calibrate --live`, `semcache sweep`, `semcache eval`, `semcache simulate`, `semcache readme-sync`.

---

## 15. Departures from the brief

1. **Nova 2 Lite instead of a Claude Haiku model** (section 2): account billing limitation. Same class of model; Claude is a one-line config change.
2. **Postgres on port 5434**, not 5433: 5433 is already used by the Regulatory RAG database on this machine.
3. **Git identity uses the GitHub noreply address** (`227244284+GouthamTiwari@users.noreply.github.com`) for every commit, so no personal email appears in the history.
4. **Trap categories also contain "hard same" pairs** (section 12.1), so guard false rejections are measured, not hidden.
5. **Policy lives in `policy.py`**, separate from `normalize.py` (one job per module).
6. The simulation's latency is **accounted, not slept**: component latencies come from calibration and real search timings, so a 5,000-request run takes seconds and is repeatable.

---

## 16. Non-goals for v1

Distributed cache clusters; Redis or other vector backends (the `CacheStore` interface allows them later); conversational caching beyond the multi-turn rule; fine-tuned embedding models; coalescing paraphrases in flight; a web UI; real authentication; caching tool-use conversations.

## 17. Notes added during the build

- 2026-11-17: Titan's per-region quota (60 requests/minute) made one region too slow for all eval embeddings in one sitting. The 256-dim pair vectors were embedded in us-east-1; everything the simulation's latency depends on (1024-dim vectors, intents) was embedded in ap-south-1. A spot check gave cosine 1.000000 between the two regions for the same text. Recorded in `reports/pairs_dev.json`.
- 2026-11-17: `current` added to the time-sensitive words (except "current account"), found by the RAG adapter demo.
- 2026-11-18 (v2): fifth guard `relation` (time order and amount bounds strict; sequence, extremes, change and bare higher/lower conflict-only; flipped "than" comparisons with antonyms normalised). Negation also counts "false/untrue that", opposite-prefix words (impossible, prohibited, ...) and any "non-" word. New category `relation`; v1 test split retired; fresh test split on 16 unseen topics. The comparison check was rewritten after the first v2 **dev** run exposed flipped comparisons in other phrasings; the test split was run only after that, once. `CacheConfig` defaults equal `reports/operating_point.json`, enforced by a test.
