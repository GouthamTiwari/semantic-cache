import dataclasses

import pytest

from semcache.config import NamespaceConfig
from semcache.normalize import exact_key, normalize
from semcache.policy import contains_pii, request_policy, store_policy
from semcache.scope import build_scope, temperature_bucket
from semcache.types import ChatRequest, TTLClass


def req(text="What is the repo rate?", **kw):
    return ChatRequest(messages=[{"role": "user", "content": text}], **kw)


# --- normalisation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  What   is the Repo Rate?? ", "what is the repo rate"),
        ("Please, can you tell me what KYC means?", "what kyc means"),
        ("Hi! How do I reset my password.", "how do i reset my password"),
        ("hello?", "hello"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_normalize_keeps_numbers_and_negation():
    assert normalize("Is it NOT allowed within 7 days?") == "is it not allowed within 7 days"


def test_exact_key_depends_on_scope():
    assert exact_key("s1", "q") != exact_key("s2", "q")
    assert exact_key("s1", "q") == exact_key("s1", "q")


# --- scope -----------------------------------------------------------------------


def scope_of(r, model="m", emb="e", dim=1024):
    return build_scope(r, model_id=model, embed_model_id=emb, embed_dim=dim)


def test_scope_changes_with_every_field():
    base = scope_of(req())
    variants = [
        scope_of(req(namespace="tenant-b")),
        scope_of(req(), model="other-model"),
        scope_of(req(system="You are a pirate.")),
        scope_of(req(tools=[{"name": "lookup"}])),
        scope_of(req(temperature=0.9)),
        scope_of(req(), emb="other-embed"),
        scope_of(req(), dim=256),
        scope_of(
            ChatRequest(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "What is the repo rate?"},
                ]
            )
        ),
        dataclasses.replace(base, schema_version=base.schema_version + 1),
    ]
    keys = {v.key for v in variants}
    assert base.key not in keys
    assert len(keys) == len(variants)


def test_scope_ignores_the_question_text():
    assert scope_of(req("a")).key == scope_of(req("b")).key


def test_tools_hash_is_order_independent_for_keys():
    a = scope_of(req(tools=[{"name": "x", "description": "y"}]))
    b = scope_of(req(tools=[{"description": "y", "name": "x"}]))
    assert a.key == b.key


@pytest.mark.parametrize("t, bucket", [(0, "t0"), (0.3, "low"), (0.7, "mid"), (1.5, "high")])
def test_temperature_bucket(t, bucket):
    assert temperature_bucket(t) == bucket


# --- request policy (P1-P3) ------------------------------------------------------


def test_p1_no_cache_flag_bypasses():
    p = request_policy(req(no_cache=True), NamespaceConfig())
    assert not p.use_cache and p.reason == "no_cache_flag"


def test_p2_tool_messages_bypass():
    r = ChatRequest(
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "tool", "content": "{temp: 30}"},
            {"role": "user", "content": "so should I carry an umbrella?"},
        ]
    )
    p = request_policy(r, NamespaceConfig())
    assert not p.use_cache and p.reason == "tool_messages"


@pytest.mark.parametrize(
    "q", ["What is the gold price today?", "latest repo rate", "Is the site down right now?"]
)
def test_p3_time_sensitive_bypasses_by_default(q):
    p = request_policy(req(q), NamespaceConfig())
    assert not p.use_cache and p.reason == "time_sensitive"


def test_p3_time_sensitive_uses_volatile_ttl_when_allowed():
    p = request_policy(req("latest repo rate"), NamespaceConfig(allow_volatile=True))
    assert p.use_cache and p.ttl_class is TTLClass.VOLATILE


def test_p3_does_not_match_inside_words():
    # "know" contains "now", "delivery" contains "live": no false positives
    assert request_policy(req("Do you know the delivery charges?"), NamespaceConfig()).use_cache


def test_namespace_default_ttl():
    p = request_policy(req(), NamespaceConfig(default_ttl=TTLClass.STATIC))
    assert p.use_cache and p.ttl_class is TTLClass.STATIC


# --- store policy (P4-P6) --------------------------------------------------------


def test_p4_errors_refusals_and_truncation_are_not_stored():
    assert store_policy("q", "", "end_turn") == "empty_answer"
    assert store_policy("q", "partial", "max_tokens") == "stop_reason:max_tokens"
    assert store_policy("q", "x", "guardrail_intervened") == "stop_reason:guardrail_intervened"
    assert store_policy("q", "x", "tool_use") == "stop_reason:tool_use"


def test_p6_unfinished_stream_is_not_stored():
    assert store_policy("q", "half an answ", None) == "stop_reason:None"


def test_clean_answer_is_stored():
    assert store_policy("What is KYC?", "Know Your Customer.", "end_turn") is None


@pytest.mark.parametrize(
    "text",
    [
        "mail me at someone@example.com",
        "call 9876543210",
        "call +91 9876543210",
        "account 1234 5678 9012",
        "ref 123-456-789-0",
    ],
)
def test_p5_pii_detected(text):
    assert contains_pii(text)
    assert store_policy(text, "ok", "end_turn") == "pii"
    assert store_policy("q", text, "end_turn") == "pii"


@pytest.mark.parametrize(
    "text", ["GST is 18% on most services", "Return within 30 days", "In 2023, the rate was 6.5%"]
)
def test_p5_normal_numbers_are_not_pii(text):
    assert not contains_pii(text)
