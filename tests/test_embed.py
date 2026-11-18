import io
import json

import numpy as np
import pytest

from semcache.bedrock import BedrockAuthError, auth_mode, with_retries
from semcache.budget import BudgetExceededError, LiveBudget
from semcache.config import Settings
from semcache.embed import HashingEmbedder, MemoEmbedder, TitanEmbedder, _Pacer


class FakeBedrock:
    """Records invoke_model calls and returns a fixed Titan-shaped response."""

    def __init__(self, dim=4):
        self.calls = []
        self.dim = dim

    def invoke_model(self, **kw):
        self.calls.append(kw)
        body = {"embedding": [3.0, 4.0] + [0.0] * (self.dim - 2), "inputTextTokenCount": 7}
        return {"body": io.BytesIO(json.dumps(body).encode())}


def test_titan_request_body_and_result():
    fake = FakeBedrock(dim=256)
    emb = TitanEmbedder(fake, dimensions=256)
    r = emb.embed("What is the repo rate?")
    call = fake.calls[0]
    assert call["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(call["body"]) == {
        "inputText": "What is the repo rate?",
        "dimensions": 256,
        "normalize": True,
    }
    assert r.input_tokens == 7
    assert np.isclose(np.linalg.norm(r.vector), 1.0)  # always unit length
    assert r.cost_usd == pytest.approx(7 * 0.024 / 1e6)


def test_titan_rejects_unsupported_dimension():
    with pytest.raises(ValueError):
        TitanEmbedder(FakeBedrock(), dimensions=300)


def test_memo_avoids_second_call(tmp_path):
    fake = FakeBedrock()
    memo = MemoEmbedder(TitanEmbedder(fake, dimensions=256), tmp_path / "m.sqlite")
    first = memo.embed("hello")
    second = memo.embed("hello")
    assert len(fake.calls) == 1
    assert not first.cached and second.cached
    assert second.cost_usd == 0.0  # a memo hit is free by default
    assert np.allclose(first.vector, second.vector)
    memo.close()
    # survives a restart
    memo2 = MemoEmbedder(TitanEmbedder(fake, dimensions=256), tmp_path / "m.sqlite")
    assert memo2.embed("hello").cached
    assert len(fake.calls) == 1


def test_memo_keys_include_dimension(tmp_path):
    fake = FakeBedrock(dim=1024)
    a = MemoEmbedder(TitanEmbedder(fake, dimensions=256), tmp_path / "m.sqlite")
    b = MemoEmbedder(TitanEmbedder(fake, dimensions=1024), tmp_path / "m.sqlite")
    a.embed("x")
    b.embed("x")
    assert len(fake.calls) == 2


def test_memo_can_report_original_cost(tmp_path):
    memo = MemoEmbedder(
        TitanEmbedder(FakeBedrock(), dimensions=256), tmp_path / "m.sqlite", report_cost_on_hit=True
    )
    memo.embed("x")
    assert memo.embed("x").cost_usd == pytest.approx(7 * 0.024 / 1e6)


def test_pacer_spaces_calls_with_fake_clock():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    p = _Pacer(2.0, sleep, clock=lambda: now[0])
    for _ in range(3):
        p.wait()
    assert slept == [0.5, 0.5]


def test_budget_blocks_overspend(tmp_path):
    b = LiveBudget(tmp_path / "spend.json", limit_usd=0.01)
    b.check(0.005)
    b.add(0.009)
    assert b.spent() == pytest.approx(0.009)
    with pytest.raises(BudgetExceededError):
        b.check(0.002)
    # the embedder checks before calling
    emb = TitanEmbedder(FakeBedrock(), dimensions=256, budget=LiveBudget(tmp_path / "z.json", 0))
    with pytest.raises(BudgetExceededError):
        emb.embed("x")


def test_auth_mode():
    assert auth_mode(Settings(_env_file=None, aws_bearer_token_bedrock="k")) == "api_key"
    assert (
        auth_mode(Settings(_env_file=None, aws_access_key_id="a", aws_secret_access_key="b"))
        == "iam"
    )
    with pytest.raises(BedrockAuthError):
        auth_mode(Settings(_env_file=None, aws_bearer_token_bedrock="k", aws_access_key_id="a"))
    with pytest.raises(BedrockAuthError):
        auth_mode(Settings(_env_file=None))


def test_retries_on_throttling_only():
    from botocore.exceptions import ClientError

    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
        return "ok"

    assert with_retries(flaky, sleep=lambda s: None) == "ok"
    assert len(attempts) == 3

    def bad():
        raise ClientError({"Error": {"Code": "ValidationException"}}, "InvokeModel")

    with pytest.raises(ClientError):
        with_retries(bad, sleep=lambda s: None)


def test_hashing_embedder_similarity_order():
    e = HashingEmbedder(256)
    a = e.embed("how do I reset my password").vector
    b = e.embed("how can I reset my password").vector
    c = e.embed("what is the GST rate on gold").vector
    assert float(a @ b) > float(a @ c)


def _add_many(path, n):
    from semcache.budget import LiveBudget

    b = LiveBudget(path, limit_usd=100)
    for _ in range(n):
        b.add(0.001)


def test_budget_survives_concurrent_processes(tmp_path):
    # Two processes adding at once once corrupted the spend file (a reader saw it half
    # written). With the lock file and atomic replace, no update is lost.
    import multiprocessing as mp

    path = tmp_path / "spend.json"
    procs = [mp.Process(target=_add_many, args=(path, 50)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert LiveBudget(path, 100).spent() == pytest.approx(0.1)
