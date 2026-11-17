# Task 17: RAG adapter example
Branch: task/17-adapter | Date: 2026-11-17

## What changed
- `src/semcache/upstream.py`: `FunctionUpstream` wraps any `fn(question)` (text, or text + token counts) so the cache can sit in front of a whole RAG pipeline; a hit skips retrieval and generation.
- `examples/rag_adapter.py`: runnable demo (`--live` for real Titan embeddings) with a slow stand-in RAG function; no dependency on the Regulatory RAG repo.
- `src/semcache/__init__.py`: public exports (`SemanticCache`, `CacheConfig`, `ChatRequest`, ...).
- `src/semcache/policy.py`: found by the demo: "What is the *current* repo rate?" was not treated as time-sensitive. `current` now counts, except in "current account" (a bank product). SPEC 6.1 updated; three engine tests that used "Current repo rate?" as a paraphrase now use "Tell me the repo rate".

## How to verify
- `uv run python examples/rag_adapter.py` → bypass, miss, exact_hit, semantic_hit, guard_reject
- `uv run pytest` → 168 passed
