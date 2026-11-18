# Task 16: README with generated results + the real evaluation run
Branch: task/16-readme | Date: 2026-11-18

## What changed
- `README.md`: pitch, Mermaid architecture diagram, quickstart, results, evaluation method, design trade-offs, "what this does not do". Every result block is generated.
- `src/semcache/eval/readme_sync.py` + `semcache readme-sync`: rebuilds the README blocks from `reports/*.json`; `tests/test_readme_sync.py` fails if the README and the reports disagree.
- `reports/`: the real run. Dev sweep for 256 and 1024 dims, guard ablation, verifier experiment, operating point, held-out test (run once), traffic simulation, SVG charts.
- `src/semcache/eval/pairs_eval.py`: Wilson 95 % interval for the test false-hit rate (93 should-miss pairs is a small sample). Operating point unchanged.
- `src/semcache/budget.py`: found during the run. Two embedding jobs writing the spend file at once crashed one of them; writes now use a lock file and an atomic replace, with retries for Windows file-sharing errors. New two-process test.
- `src/semcache/eval/commands.py`: the simulator no longer prints 15,000 JSON request lines.
- `src/semcache/store/pgvector.py`: 5-second connect timeout. With Docker stopped, the Postgres tests hung instead of failing.

## Results (from reports/)
- Dev: no threshold alone meets FHR <= 1 %; with guards, 0.95 does (0.8 %). 256 dims chosen by the pre-set rule. Verifier not worth it at the 1 % target (recall up, FHR 1.2-1.6 %).
- Test (once): FHR 2.1 % (2/93, 95 % CI 0.6-7.5 %) vs 20.4 % without guards. Both false hits are outside the guards' scope (antonym without a negation word; unit change without a number).
- Simulation: exact match saves 83.3 %, semantic + guards 86.2 % with 48 wrong answers in 5,000 requests, almost all "before/after"-type relation pairs. Not fixed after the fact; listed as the next step.

## Cost
- Project live spend: about $0.16 of the $5 cap.
