# Task 13: Pair dataset builder + review tooling
Branch: task/13-dataset | Date: 2026-11-17

## What changed
- `src/semcache/eval/dataset.py`: `Pair` schema, load/save, validation (duplicate ids and pairs, a == b, paraphrase must be `same`, no text in both splits, no group in both splits), stable group-based split (70/30, seeds always dev), stats, 10 % review sample CSV and `apply_review`.
- `src/semcache/eval/generate.py`: Nova 2 Lite generators for labelled pairs (40 topics x 15 pairs) and traffic intent families (40 topics x 2 families x 3 intents x 5 phrasings).
- `src/semcache/eval/runtime.py`: memo-backed embedder and verifier for eval runs; offline mode raises on a memo miss instead of silently calling AWS.
- `src/semcache/eval/commands.py`: `semcache dataset generate|validate|review-sample|apply-review`, `semcache intents generate`, `semcache embed-memo`.
- `data/pairs/pairs.jsonl` (622 pairs, 284 same / 338 different, dev 449 / test 173), `generated_raw.jsonl`, `review_fixes.json`, `review_sample.csv`; `data/traffic/intents.jsonl` (240 intents, 1,200 phrasings, used by Task 15).
- `docs/DATASET_REVIEW.md`: full review pass (16 changes), human spot check pending.

## How to verify
- `uv run semcache dataset validate` → "dataset valid"
- `uv run pytest tests/test_dataset.py` → 6 passed

## Cost
- Generation: $0.063 (pairs) + $0.067 (intents), recorded in `.embed_cache/live_spend.json`.
