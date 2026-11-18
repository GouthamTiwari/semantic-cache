from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from semcache.config import CacheConfig, NamespaceConfig, Settings
from semcache.pricing import UnknownModelPriceError, cost_usd
from semcache.types import ChatRequest, Decision, TTLClass


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("EMBED_DIMENSIONS", "256")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    s = Settings(_env_file=None)
    assert s.embed_dimensions == 256
    assert s.database_url == "postgresql://x"
    assert s.aws_region == "ap-south-1"


def test_ttl_classes():
    assert TTLClass.STATIC.ttl == timedelta(days=7)
    assert TTLClass.NORMAL.ttl == timedelta(hours=24)
    assert TTLClass.VOLATILE.ttl == timedelta(minutes=15)


def test_decision_is_hit():
    assert (
        Decision.EXACT_HIT.is_hit and Decision.SEMANTIC_HIT.is_hit and Decision.VERIFIED_HIT.is_hit
    )
    assert not Decision.MISS.is_hit and not Decision.GUARD_REJECT.is_hit
    assert not Decision.BYPASS.is_hit


def test_cache_config_validates():
    with pytest.raises(ValueError):
        CacheConfig(t_low=0.95, t_high=0.90)
    with pytest.raises(ValueError):
        CacheConfig(entity_mode="loose")
    cfg = CacheConfig(namespaces={"news": NamespaceConfig(allow_volatile=True)})
    assert cfg.namespace("news").allow_volatile
    assert cfg.namespace("unknown") == NamespaceConfig()


def test_request_last_message_must_be_user():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "assistant", "content": "hi"}])
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "   "}])
    req = ChatRequest(
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert req.question == "c"
    assert len(req.history) == 2


def test_cost_matches_hand_maths():
    # Nova 2 Lite: 1,000 input tokens at $0.35/M + 500 output at $2.95/M
    expected = 1000 * 0.35 / 1e6 + 500 * 2.95 / 1e6
    assert cost_usd("global.amazon.nova-2-lite-v1:0", 1000, 500) == pytest.approx(expected)
    assert cost_usd("amazon.titan-embed-text-v2:0", 1_000_000) == pytest.approx(0.024)


def test_unknown_model_price_raises():
    with pytest.raises(UnknownModelPriceError):
        cost_usd("no-such-model", 10, 10)
    with pytest.raises(ValueError):
        cost_usd("mock-upstream", -1, 0)


def test_entry_expiry():
    from semcache.types import CacheEntry

    now = datetime(2026, 1, 1, tzinfo=UTC)
    e = CacheEntry(
        id="1", scope_key="s", namespace="n", model_id="m", exact_key="k", query_text="q",
        query_norm="q", embedding=[1.0], answer_text="a", stop_reason="end_turn",
        input_tokens=1, output_tokens=1, cost_usd=0.0, ttl_class=TTLClass.NORMAL,
        created_at=now, expires_at=now + timedelta(hours=1),
    )  # fmt: skip
    assert not e.is_expired(now)
    assert e.is_expired(now + timedelta(hours=1))


def test_defaults_match_the_evaluated_operating_point():
    # The numbers in the README describe CacheConfig(); this keeps them from drifting apart.
    import json
    from pathlib import Path

    op = json.loads(Path("reports/operating_point.json").read_text(encoding="utf-8"))
    cfg = CacheConfig()
    assert (cfg.t_high, cfg.t_low) == (op["t_high"], op["t_low"])
    assert list(cfg.guards) == op["guards"]
    assert cfg.verifier_enabled == op["verifier_enabled"]
    assert cfg.entity_mode == op["entity_mode"]
    assert Settings(_env_file=None).embed_dimensions == op["embed_dim"]
