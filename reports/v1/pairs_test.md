# Pair evaluation: held-out test split

Run once at the operating point frozen on dev (hash `5cceaee13cea`). Nothing was tuned after this run.

Test pairs: 173 (80 same, 93 different). False-hit rate 95 % interval (Wilson): 0.6% to 7.5%.

| config | precision | recall | false-hit rate | false hits |
|---|---|---|---|---|
| chosen operating point | 90.9% | 25.0% | 2.1% | 2 |
| same threshold, no guards | 55.8% | 30.0% | 20.4% | 19 |

By category:

| category | different pairs | false hits | false-hit rate | same pairs | recall |
|---|---|---|---|---|---|
| paraphrase | 0 | 0 | n/a | 36 | 16.7% |
| near_miss_number | 22 | 1 | 4.5% | 11 | 72.7% |
| negation | 25 | 1 | 4.0% | 9 | 11.1% |
| entity_swap | 23 | 0 | 0.0% | 12 | 33.3% |
| same_topic_different_question | 23 | 0 | 0.0% | 12 | 8.3% |

```json
{
  "embed_dim": 256,
  "t_high": 0.95,
  "t_low": 0.95,
  "verifier_enabled": false,
  "guards": [
    "numbers",
    "dates",
    "negation",
    "entities"
  ],
  "entity_mode": "swap",
  "fhr_target": 0.01,
  "chosen_on": "dev split only"
}
```
