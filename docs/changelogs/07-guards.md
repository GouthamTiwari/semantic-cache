# Task 07: Guards
Branch: task/07-guards | Date: 2026-11-17

## What changed
- `src/semcache/guards.py`: four pure functions plus `run_guards`.
  - `numbers_guard`: multiset of numeric values; digits and number words ("7" = "seven", "2 lakh" = "two lakh" = 200000); years excluded.
  - `dates_guard`: years, year ranges, FY labels, numeric dates, month and weekday names ("may" counts as a month only next to a number).
  - `negation_guard`: negation polarity (count of cues mod 2).
  - `entities_guard`: ALL-CAPS tokens, capitalised non-initial words, camel-case names, quoted terms; `swap` (default) and `strict` modes.
- `tests/test_guard_{numbers,dates,negation,entities}.py`: one file per guard.

## How to verify
- `uv run pytest tests/test_guard_*.py` → 45 passed

## Tests
- Full suite 105 passed. On the 30 seed pairs the guards reject 13 of 20 `different` pairs and none of the 10 `same` pairs; lowercase swaps (restaurants vs hotels) and same-topic questions pass the guards by design and are left to the threshold and verifier.
