import json

import pytest
from fastapi.testclient import TestClient

from semcache.api import create_app
from semcache.cache import SemanticCache
from semcache.config import CacheConfig
from semcache.embed import HashingEmbedder
from semcache.store import InMemoryStore
from semcache.upstream import MockUpstream


def make_client(token="", upstream=None):
    cache = SemanticCache(
        store=InMemoryStore(),
        embedder=HashingEmbedder(64),
        upstream=upstream or MockUpstream(latency_ms=0.0, sleep=False),
        config=CacheConfig(t_high=0.95, t_low=0.95),
    )
    return TestClient(create_app(cache, admin_token=token)), cache


def chat(client, text, **kw):
    body = {"messages": [{"role": "user", "content": text}], **kw}
    return client.post("/v1/chat", json=body)


def test_chat_miss_then_hit():
    client, _ = make_client()
    r1 = chat(client, "What is KYC?").json()
    r2 = chat(client, "what is kyc").json()
    assert r1["decision"] == "miss" and r1["stored"]
    assert r2["decision"] == "exact_hit" and r2["text"] == r1["text"]
    assert r2["saved_usd"] > 0


def test_chat_validation_error():
    client, _ = make_client()
    r = client.post("/v1/chat", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 422


def test_upstream_error_is_502():
    client, _ = make_client(upstream=MockUpstream(fail_with=RuntimeError("x"), sleep=False))
    assert chat(client, "q").status_code == 502


def test_streaming_sse():
    client, _ = make_client()
    with client.stream("POST", "/v1/chat", json={
        "messages": [{"role": "user", "content": "How do I file GST?"}], "stream": True
    }) as r:  # fmt: skip
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    text = "".join(e["delta"] for e in events if "delta" in e)
    assert text == "Answer to: How do I file GST?"
    assert events[-1]["done"] and events[-1]["decision"] == "miss"


def test_metrics_and_health():
    client, _ = make_client()
    chat(client, "What is KYC?")
    m = client.get("/metrics")
    assert m.status_code == 200 and "semcache_requests_total" in m.text
    assert client.get("/healthz").json() == {"status": "ok", "store": "memory"}


def test_admin_invalidate_and_stats():
    client, cache = make_client()
    r = chat(client, "What is KYC?", namespace="t1").json()
    assert client.get("/admin/stats").json()["store"]["entries"] == 1
    assert client.post("/admin/invalidate", json={}).status_code == 400
    d = client.post("/admin/invalidate", json={"namespace": "t1"}).json()
    assert d == {"deleted": 1}
    assert chat(client, "What is KYC?", namespace="t1").json()["decision"] == "miss"
    assert client.post("/admin/invalidate", json={"entry_id": r["entry_id"]}).json() == {
        "deleted": 0
    }


@pytest.mark.parametrize("header, status", [(None, 401), ("wrong", 401), ("s3cret", 200)])
def test_admin_token(header, status):
    client, _ = make_client(token="s3cret")
    headers = {"X-Admin-Token": header} if header else {}
    assert client.get("/admin/stats", headers=headers).status_code == status
    # the public endpoints never need the token
    assert client.get("/healthz").status_code == 200


def test_cli_parser():
    from semcache.cli import build_parser

    args = build_parser().parse_args(["serve", "--mock", "--port", "9000"])
    assert args.mock and args.port == 9000
    args = build_parser().parse_args(["purge", "--expired"])
    assert args.expired
