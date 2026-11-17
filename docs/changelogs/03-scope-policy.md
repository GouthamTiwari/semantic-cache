# Task 03: Scope, normalisation, policy
Branch: task/03-scope-policy | Date: 2026-11-17

## What changed
- `src/semcache/normalize.py`: matching text (NFKC, lowercase, whitespace, leading filler, trailing punctuation) and the L1 exact key.
- `src/semcache/scope.py`: scope key from namespace, model, system-prompt hash, tools hash, temperature bucket, embedding model + dimension, history hash and schema version.
- `src/semcache/policy.py`: request rules P1-P3 (no_cache, tool messages, time-sensitive with optional volatile TTL) and store rules P4-P6 (empty, unclean stop reason, PII).

## How to verify
- `uv run pytest tests/test_scope_normalize_policy.py` → 32 passed

## Tests
- Every policy rule has a test; scope changes with each of its 9 fields; "know"/"delivery" do not trip the time-sensitive rule; ordinary numbers are not PII. Full suite 41 passed.

## Self-review
- Multi-turn decision from SPEC 6.2 is implemented through `history_hash` in the scope.
