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
