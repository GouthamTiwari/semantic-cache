# Task 19: Relation guard (v2)
Branch: task/19-relation-guard | Date: 2026-11-18

## Why
The v1 traffic simulation served 48 wrong answers; almost all came from relation words the four v1 guards do not check ("before 55" vs "after 55", "next" vs "last" Friday) and one flipped comparison ("football older than cricket" vs the reverse). v1 is frozen at tag `v1.0.0` with its reports in `reports/v1/`.

## What changed
- `src/semcache/guards.py`: `relation_guard`.
  - Strict axes (any difference rejects): time order (before/prior to/until vs after/since/beyond, "following" + date) and amount bounds (more than/above/over + amount vs less than/below/up to).
  - Conflict-only axes (both sides must disagree): sequence (next vs last), extremes (maximum vs minimum), change (increase vs cut). Words like "last date" appear in harmless paraphrases, so a one-sided word does not reject.
  - Flipped comparisons: "A older than B" vs "B older than A" rejects; antonyms are normalised, so "B younger than A" still matches.
  - Negation guard: new cues "false that" / "untrue that" ("Is it false that ...?" from the simulation).
- `src/semcache/eval/pairs_eval.py`: the eval now runs five guards.
- `tests/test_guard_relation.py`: the simulation's false hits must be rejected; paraphrases with the same relation must pass.

## How to verify
- `uv run pytest tests/test_guard_relation.py` → 21 passed; full suite 192 passed.
- On the v1 dev pairs (which contain no relation traps) the operating point is unchanged (0.95, FHR 0.8 %) and recall drops by one pair; the effect is measured on new relation data in Task 20/21.
