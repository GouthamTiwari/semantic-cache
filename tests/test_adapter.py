from semcache import CacheConfig, ChatRequest, Decision, SemanticCache
from semcache.embed import HashingEmbedder
from semcache.store import InMemoryStore
from semcache.upstream import FunctionUpstream


def test_function_upstream_caches_any_answer_function():
    calls = []

    def rag(question):
        calls.append(question)
        return f"rag answer: {question}", 1800, 250

    cache = SemanticCache(
        store=InMemoryStore(),
        embedder=HashingEmbedder(64),
        upstream=FunctionUpstream(rag, model_id="global.amazon.nova-2-lite-v1:0"),
        config=CacheConfig(t_high=0.95, t_low=0.95, verifier_enabled=False),
    )
    req = ChatRequest(messages=[{"role": "user", "content": "GSTR-3B late fee?"}])
    first = cache.get_or_call(req)
    second = cache.get_or_call(req)
    assert first.decision is Decision.MISS and second.decision is Decision.EXACT_HIT
    assert len(calls) == 1
    assert second.saved_usd > 0  # token counts flow through, so savings are known


def test_function_upstream_accepts_plain_text():
    up = FunctionUpstream(lambda q: "plain")
    r = up.complete(ChatRequest(messages=[{"role": "user", "content": "x?"}]), "m")
    assert r.text == "plain" and r.input_tokens == 0 and r.stop_reason == "end_turn"
