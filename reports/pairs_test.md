# Pair evaluation: held-out test split

Run once at the operating point frozen on dev (hash `e55b98277738`). Nothing was tuned after this run.

Test pairs: 272 (119 same, 153 different). False-hit rate 95 % interval (Wilson): 0.4% to 4.6%.

| config | precision | recall | false-hit rate | false hits |
|---|---|---|---|---|
| chosen operating point | 97.4% | 62.2% | 1.3% | 2 |
| same threshold, no guards | 45.8% | 27.7% | 25.5% | 39 |

By category:

| category | different pairs | false hits | false-hit rate | same pairs | recall |
|---|---|---|---|---|---|
| paraphrase | 0 | 0 | n/a | 48 | 62.5% |
| near_miss_number | 31 | 0 | 0.0% | 12 | 75.0% |
| negation | 32 | 0 | 0.0% | 10 | 40.0% |
| entity_swap | 30 | 0 | 0.0% | 16 | 31.2% |
| same_topic_different_question | 32 | 0 | 0.0% | 16 | 75.0% |
| relation | 28 | 2 | 7.1% | 17 | 82.3% |

```json
{
  "embed_dim": 1024,
  "t_high": 0.94,
  "t_low": 0.84,
  "verifier_enabled": true,
  "guards": [
    "numbers",
    "dates",
    "negation",
    "entities",
    "relation"
  ],
  "entity_mode": "swap",
  "fhr_target": 0.01,
  "chosen_on": "dev split only"
}
```
