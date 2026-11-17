import pytest

from semcache.budget import BudgetExceededError, LiveBudget
from semcache.types import ChatRequest
from semcache.upstream import BedrockUpstream, UpstreamResult
from semcache.verify import LLMVerifier, pair_key, parse_verdict


class FakeConverse:
    def __init__(self, text="YES", stop="end_turn"):
        self.text, self.stop = text, stop
        self.calls = []

    def converse(self, **kw):
        self.calls.append(kw)
        return {
            "output": {"message": {"content": [{"reasoningContent": {}}, {"text": self.text}]}},
            "stopReason": self.stop,
            "usage": {"inputTokens": 30, "outputTokens": 2},
        }

    def converse_stream(self, **kw):
        self.calls.append(kw)
        events = [{"messageStart": {"role": "assistant"}}]
        events += [{"contentBlockDelta": {"delta": {"text": t}}} for t in ("Hel", "lo")]
        events += [{"messageStop": {"stopReason": self.stop}}]
        events += [{"metadata": {"usage": {"inputTokens": 12, "outputTokens": 3}}}]
        return {"stream": iter(events)}


def req(**kw):
    return ChatRequest(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "What is KYC?"},
        ],
        **kw,
    )


def test_bedrock_complete_request_shape_and_result():
    fake = FakeConverse(text="Know your customer.")
    up = BedrockUpstream(fake)
    r = up.complete(req(system="Be brief.", max_tokens=100), "global.amazon.nova-2-lite-v1:0")
    kw = fake.calls[0]
    assert kw["modelId"] == "global.amazon.nova-2-lite-v1:0"
    assert kw["system"] == [{"text": "Be brief."}]
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    assert kw["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.0}
    assert r.text == "Know your customer." and r.stop_reason == "end_turn"
    assert r.cost_usd == pytest.approx(30 * 0.35 / 1e6 + 2 * 2.95 / 1e6)


def test_bedrock_stream_yields_deltas_then_result():
    up = BedrockUpstream(FakeConverse())
    items = list(up.stream(req(), "global.amazon.nova-2-lite-v1:0"))
    assert items[:2] == ["Hel", "lo"]
    final = items[-1]
    assert isinstance(final, UpstreamResult)
    assert final.text == "Hello" and final.output_tokens == 3 and final.stop_reason == "end_turn"


def test_bedrock_budget_blocks_before_call(tmp_path):
    fake = FakeConverse()
    up = BedrockUpstream(fake, budget=LiveBudget(tmp_path / "b.json", 0.0))
    with pytest.raises(BudgetExceededError):
        up.complete(req(), "global.amazon.nova-2-lite-v1:0")
    assert fake.calls == []


@pytest.mark.parametrize(
    "raw, same",
    [("YES", True), ("Yes.", True), ("yes, they match", True), ("NO", False), ("", False),
     ("Maybe", False), ("**YES**", True), ("I think yes", False)],
)  # fmt: skip
def test_parse_verdict_is_fail_safe(raw, same):
    assert parse_verdict(raw) is same


def test_pair_key_is_unordered_and_normalised():
    assert pair_key("What is KYC?", "kyc meaning") == pair_key("KYC meaning", "what is kyc")


def test_llm_verifier_memoises_on_disk(tmp_path):
    fake = FakeConverse("YES")
    v = LLMVerifier(fake, "global.amazon.nova-2-lite-v1:0", memo_path=tmp_path / "v.sqlite")
    first = v.verify("What is KYC?", "KYC meaning?")
    second = v.verify("kyc meaning", "what is kyc")
    assert first.same and second.same and second.cached
    assert len(fake.calls) == 1
    assert fake.calls[0]["inferenceConfig"] == {"maxTokens": 5, "temperature": 0.0}
    v2 = LLMVerifier(fake, "global.amazon.nova-2-lite-v1:0", memo_path=tmp_path / "v.sqlite",
                     report_cost_on_hit=True)  # fmt: skip
    again = v2.verify("What is KYC?", "KYC meaning?")
    assert again.cached and again.cost_usd > 0
    assert len(fake.calls) == 1
