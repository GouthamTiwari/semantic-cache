# Traffic simulation

5000 requests over a virtual 24 h: 240 intents (families of near-miss siblings), 805 distinct phrasings, Zipf popularity (s = 1.1), 250 burst copies, seed 42.

The LLM is a mock calibrated from 40 real `global.amazon.nova-2-lite-v1:0` calls (latency p50 1371.3 ms, p95 2396.2 ms; mean 82.7 input and 105.7 output tokens). Embeddings are real Titan v2 vectors from the memo, with their recorded call latency (p50 288.7 ms). Costs use the price table in `src/semcache/pricing.py`.

| config | hit rate | exact hits | semantic hits | false hits | LLM calls | total USD | net saved USD | net saved | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| no cache | 0.0% | 0 | 0 | 0 | 5000 | $1.6983 | $0.0000 | 0.0% | 1377.7 | 2377.6 |
| exact only | 83.3% | 4166 | 0 | 0 | 834 | $0.2843 | $1.4140 | 83.3% | 0.0 | 1485.2 |
| semantic (threshold only) | 86.6% | 3147 | 1184 | 67 | 669 | $0.2279 | $1.4703 | 86.6% | 0.3 | 1533.8 |
| semantic + guards | 86.2% | 3243 | 1068 | 48 | 689 | $0.2352 | $1.4631 | 86.2% | 0.3 | 1536.6 |
| semantic + guards + verifier | 88.6% | 2754 | 1677 | 92 | 569 | $0.2317 | $1.4666 | 86.4% | 0.4 | 2310.2 |

Latency by outcome (ms):

| config | hit p50 | hit p95 | miss p50 | miss p95 |
|---|---|---|---|---|
| no cache | None | None | 1377.7 | 2377.6 |
| exact only | 0.0 | 0.0 | 1364.9 | 2377.6 |
| semantic (threshold only) | 0.2 | 73.8 | 1445.4 | 2456.4 |
| semantic + guards | 0.2 | 73.4 | 1449.0 | 2454.2 |
| semantic + guards + verifier | 0.3 | 1268.2 | 1445.9 | 2466.3 |

Net saved = cost without a cache minus (LLM cost + embedding cost + judge cost). A miss in a semantic configuration pays for an embedding and a search on top of the LLM call; that overhead is included.
