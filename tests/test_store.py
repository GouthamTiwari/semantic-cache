"""Contract tests: every CacheStore backend must pass the same tests."""

from datetime import timedelta

import pytest

from helpers import NOW, make_entry, unit
from semcache.store import InMemoryStore


@pytest.fixture(params=["memory"])
def store(request):
    if request.param == "memory":
        yield InMemoryStore()


def test_put_and_get_exact(store):
    e = make_entry(text="hello")
    store.put(e)
    got = store.get_exact("s1", e.exact_key, NOW)
    assert got is not None and got.id == e.id and got.answer_text == "a"


def test_get_exact_respects_scope(store):
    e = make_entry(scope="s1", text="hello")
    store.put(e)
    assert store.get_exact("s2", e.exact_key, NOW) is None


def test_put_same_exact_key_replaces(store):
    store.put(make_entry(text="q", answer="old"))
    store.put(make_entry(text="q", answer="new"))
    got = store.get_exact("s1", "s1:q", NOW)
    assert got.answer_text == "new"
    assert store.stats()["entries"] == 1


def test_search_sorted_top_k_inside_scope(store):
    store.put(make_entry(text="a", vector=unit(1, 0)))
    store.put(make_entry(text="b", vector=unit(1, 1)))
    store.put(make_entry(text="c", vector=unit(0, 1)))
    store.put(make_entry(scope="other", text="d", vector=unit(1, 0)))
    res = store.search("s1", unit(1, 0.1), k=2, now=NOW)
    assert [e.query_text for e, _ in res] == ["a", "b"]
    assert res[0][1] > res[1][1]
    assert res[0][1] == pytest.approx(0.995, abs=0.01)


def test_search_empty_scope(store):
    assert store.search("nothing", unit(1), k=3, now=NOW) == []


def test_expired_entries_are_ignored_then_purged(store):
    old = make_entry(text="old", expires_in=timedelta(minutes=-1))
    fresh = make_entry(text="fresh", vector=unit(0.9, 0.1))
    store.put(old)
    store.put(fresh)
    assert store.get_exact("s1", old.exact_key, NOW) is None
    assert [e.query_text for e, _ in store.search("s1", unit(1), k=5, now=NOW)] == ["fresh"]
    assert store.purge_expired(NOW) == 1
    assert store.stats()["entries"] == 1


def test_invalidate_by_each_selector(store):
    a = make_entry(scope="s1", namespace="n1", text="a")
    b = make_entry(scope="s2", namespace="n1", text="b")
    c = make_entry(scope="s3", namespace="n2", text="c")
    for e in (a, b, c):
        store.put(e)
    assert store.invalidate(entry_id=c.id) == 1
    assert store.invalidate(scope_key="s1") == 1
    assert store.invalidate(namespace="n1") == 1
    assert store.stats()["entries"] == 0


def test_invalidate_needs_exactly_one_selector(store):
    with pytest.raises(ValueError):
        store.invalidate()
    with pytest.raises(ValueError):
        store.invalidate(scope_key="s1", namespace="n1")


def test_record_hit(store):
    e = make_entry()
    store.put(e)
    store.record_hit(e.id, NOW)
    store.record_hit(e.id, NOW)
    assert store.get_exact("s1", e.exact_key, NOW).hit_count == 2
    assert store.stats()["total_hits"] == 2
