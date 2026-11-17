"""Put the semantic cache in front of a RAG answer function (no HTTP layer needed).

Run with no AWS (hashing embedder, demo only):  uv run python examples/rag_adapter.py
Run with real Titan embeddings:                 uv run python examples/rag_adapter.py --live

In a real project, replace `answer_with_rag` with your pipeline, for example the
Regulatory RAG project's answer function. On a cache hit, retrieval and generation are
skipped entirely.
"""

from __future__ import annotations

import sys
import time

from semcache import CacheConfig, ChatRequest, SemanticCache
from semcache.embed import HashingEmbedder
from semcache.store import InMemoryStore
from semcache.upstream import FunctionUpstream


def answer_with_rag(question: str) -> tuple[str, int, int]:
    """Stand-in for a real RAG pipeline: retrieve, rerank, generate (slow and paid)."""
    time.sleep(1.0)
    answer = f"(answer from the RAG pipeline for: {question})"
    return answer, 1800, 250  # prompt tokens incl. retrieved chunks, output tokens


def build_cache(live: bool) -> SemanticCache:
    if live:
        from semcache.bedrock import get_runtime_client
        from semcache.embed import TitanEmbedder

        embedder = TitanEmbedder(get_runtime_client(), dimensions=1024)
        config = CacheConfig(verifier_enabled=False)  # defaults = the evaluated operating point
    else:
        embedder = HashingEmbedder(256)  # not semantic: demo thresholds only
        config = CacheConfig(t_high=0.80, t_low=0.80, verifier_enabled=False)
    return SemanticCache(
        store=InMemoryStore(),
        embedder=embedder,
        upstream=FunctionUpstream(answer_with_rag, model_id="global.amazon.nova-2-lite-v1:0"),
        config=config,
    )


def main() -> None:
    cache = build_cache("--live" in sys.argv)
    questions = [
        "What is the current repo rate?",  # time-sensitive: bypassed by policy
        "What is the penalty for late filing of GSTR-3B?",
        "What is the penalty for late filing of GSTR-3B?",  # exact hit
        "What's the penalty for filing GSTR-3B late?",  # paraphrase: semantic hit (live)
        "What is the penalty for filing GSTR-1 late?",  # different form: must not hit
    ]
    for q in questions:
        r = cache.get_or_call(
            ChatRequest(namespace="regrag", messages=[{"role": "user", "content": q}])
        )
        print(f"{r.decision.value:<13} {r.latency_ms:7.1f} ms  saved ${r.saved_usd:.6f}  {q}")


if __name__ == "__main__":
    main()
