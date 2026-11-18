# Traffic simulation

5000 requests over a virtual 24 h: 240 intents (families of near-miss siblings), 805 distinct phrasings, Zipf popularity (s = 1.1), 250 burst copies, seed 42.

The LLM is a mock calibrated from 40 real `global.amazon.nova-2-lite-v1:0` calls (latency p50 1371.3 ms, p95 2396.2 ms; mean 82.7 input and 105.7 output tokens). Embeddings are real Titan v2 vectors from the memo, with their recorded call latency (p50 73.8 ms). Costs use the price table in `src/semcache/pricing.py`.

| config | hit rate | exact hits | semantic hits | false hits | LLM calls | total USD | net saved USD | net saved | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| no cache | 0.0% | 0 | 0 | 0 | 5000 | $1.6983 | $0.0000 | 0.0% | 1377.7 | 2377.6 |
| exact only | 83.3% | 4166 | 0 | 0 | 834 | $0.2843 | $1.4140 | 83.3% | 0.0 | 1485.2 |
| semantic (threshold only) | 87.0% | 3052 | 1299 | 103 | 649 | $0.2213 | $1.4769 | 87.0% | 0.1 | 1534.8 |
| semantic + guards | 86.3% | 3177 | 1139 | 1 | 684 | $0.2335 | $1.4648 | 86.2% | 0.1 | 1553.3 |
| semantic + guards + verifier | 90.1% | 2536 | 1969 | 17 | 495 | $0.2244 | $1.4739 | 86.8% | 0.2 | 3275.5 |

Latency by outcome (ms):

| config | hit p50 | hit p95 | miss p50 | miss p95 |
|---|---|---|---|---|
| no cache | None | None | 1377.7 | 2377.6 |
| exact only | 0.0 | 0.0 | 1364.9 | 2377.6 |
| semantic (threshold only) | 0.1 | 205.4 | 1455.0 | 2458.3 |
| semantic + guards | 0.1 | 205.5 | 1462.8 | 2515.6 |
| semantic + guards + verifier | 0.1 | 3282.4 | 1457.3 | 2819.9 |

Net saved = cost without a cache minus (LLM cost + embedding cost + judge cost). A miss in a semantic configuration pays for an embedding and a search on top of the LLM call; that overhead is included.
