"""Real Bedrock calls. Opt-in: `uv run pytest -m live` (costs a fraction of a cent)."""

import numpy as np
import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    from semcache.bedrock import get_runtime_client
    from semcache.config import Settings

    return get_runtime_client(Settings())


def test_titan_embedding_live(client):
    from semcache.embed import TitanEmbedder

    emb = TitanEmbedder(client, dimensions=256)
    a = emb.embed("What is the repo rate?")
    b = emb.embed("Can you tell me what the repo rate is?")
    assert a.vector.shape == (256,)
    assert a.input_tokens > 0
    assert float(np.dot(a.vector, b.vector)) > 0.7


def test_nova_converse_and_stream_live(client):
    from semcache.config import NOVA_2_LITE
    from semcache.types import ChatRequest
    from semcache.upstream import BedrockUpstream, UpstreamResult

    up = BedrockUpstream(client)
    r = ChatRequest(messages=[{"role": "user", "content": "Say hello in one word."}], max_tokens=20)
    result = up.complete(r, NOVA_2_LITE)
    assert result.text and result.input_tokens > 0 and result.stop_reason == "end_turn"
    items = list(up.stream(r, NOVA_2_LITE))
    assert isinstance(items[-1], UpstreamResult) and items[-1].text


def test_judge_live(client):
    from semcache.config import NOVA_2_LITE
    from semcache.verify import LLMVerifier

    v = LLMVerifier(client, NOVA_2_LITE)
    assert v.verify("How do I reset my password?", "I forgot my password, how do I change it?").same
    assert not v.verify("Return within 7 days?", "Return within 30 days?").same
