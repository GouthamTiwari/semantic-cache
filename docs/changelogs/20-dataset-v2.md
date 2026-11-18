# Task 20: v2 dataset (relation pairs + fresh test split)
Branch: task/20-dataset-v2 | Date: 2026-11-18

## What changed
- `src/semcache/eval/dataset.py`: new category `relation`; new split value `retired` for the v1 test split; validation now requires that the test split shares no question with dev **or** the retired split.
- `src/semcache/eval/generate.py`: `generate_v2` (relation pairs for the 28 dev topics; a full six-category test set on 16 new topics). The v1 generator is unchanged.
- `src/semcache/eval/commands.py`: `semcache dataset generate-v2`.
- `data/pairs/pairs.jsonl`: 971 pairs = dev 526 (incl. 77 relation pairs), retired 173, test 272 (153 should-miss, 119 should-hit). Raw output in `generated_raw_v2.jsonl`; all 32 review changes in `review_fixes_v2.json`; `review_sample.csv` regenerated (94 pairs) for the human spot check.
- `docs/DATASET_REVIEW.md`: pass 2 (v2) documented, including a known prompt bias.
- `tests/test_dataset.py`: new category and retired split; a test question reused from the retired split is a validation error.

## Cost
- 44 + 22 (an interrupted first attempt) Nova 2 Lite calls, about $0.05.
