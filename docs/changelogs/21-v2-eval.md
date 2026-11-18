# Task 21: v2 evaluation
Branch: task/21-v2-eval | Date: 2026-11-18

## What happened, in order
1. First v2 dev run (relation guard as merged in Task 19): the safe threshold rose to 0.99 with 7 % recall. The 8 dev false hits at 0.95 were flipped comparisons phrased differently from "A older than B" ("Does RTGS take longer than IMPS?"), bare "higher" vs "lower", and opposite-prefix words without "not" ("possible" vs "impossible", "safe" vs "unsafe").
2. Fixes, on dev data only: comparison check rewritten (split at "than", detect swapped sides, antonyms normalised); bare higher/lower conflict when no "than"; negation counts opposite-prefix words and "non-" words.
3. Second dev run: threshold 0.94, 1024 dims (rule: 256 only if within 2 points of recall), verifier worth it (grey zone 0.84, recall 57.5 % at FHR 0.7 %).
4. Fresh test split, run once: FHR 1.3 % (2/153, CI 0.4-4.6 %), precision 97.4 %, recall 62.2 %; 25.5 % FHR without guards. Both false hits named in the README limits.
5. Simulation (same stream as v1): guards only 1 wrong answer (v1: 48); with the verifier 17 wrong answers, +4 points of hits, verified hits wait seconds for the judge. README states the trade-off and how to turn the verifier off.

## What changed
- `src/semcache/guards.py`: rewritten comparison check, bare level words, opposite-prefix negation.
- `src/semcache/config.py`: defaults = v2 operating point; `tests/test_config_types.py` enforces it.
- `src/semcache/eval/readme_sync.py`: v1 → v2 table (`reports/v1/` vs `reports/`), verifier trade-off line.
- `reports/`: v2 results; `reports/v1/`: frozen v1 results.
- `README.md`, `docs/SPEC.md`, `docs/TASKS.md`: v2 described.

## Cost
- Project live spend: about $0.23 of the $5 cap.
