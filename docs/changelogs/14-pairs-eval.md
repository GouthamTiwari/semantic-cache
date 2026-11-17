# Task 14: Sweep, ablations, verifier eval, test run
Branch: task/14-pairs-eval | Date: 2026-11-17

## What changed
- `src/semcache/eval/pairs_eval.py`: scores every pair from the embedding memo and runs the production `evaluate_candidate` on it.
  - Experiment 1: sweep 0.70-0.99 with and without guards, per category.
  - Experiment 2: 256 vs 1024 dims, with a rule fixed before seeing results (256 only if its recall at the target is within 2 points).
  - Experiment 3: guard ablation at the chosen threshold (none, each alone, all, all with strict entities), including same pairs wrongly rejected.
  - Experiment 4: grey-zone verifier for five grey-zone widths, with a "worth it" rule (>= 2 points of recall at FHR <= 1 %).
  - Experiment 5: `run_test` on the held-out split at the frozen operating point.
- `src/semcache/eval/report.py`: JSON/Markdown writers and a dependency-free SVG line chart.
- `src/semcache/eval/commands.py`: `semcache sweep [--live]` and `semcache test [--live]`; `test` refuses to run with a changed operating point unless `--new-operating-point` is passed, so the test split cannot be used for tuning by accident.

## How to verify
- `uv run pytest tests/test_pairs_eval.py` → 4 passed (synthetic data with controlled similarities)
- Real results are produced in the results commit after the embedding memo is complete.
