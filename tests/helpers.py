"""Small builders shared by tests."""

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np

from semcache.types import CacheEntry, TTLClass

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def unit(*values, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[: len(values)] = values
    return (v / np.linalg.norm(v)).tolist()


def make_entry(
    *,
    scope="s1",
    namespace="ns",
    text="q",
    vector=None,
    answer="a",
    exact=None,
    expires_in=timedelta(hours=1),
    dim=8,
    now=NOW,
):
    return CacheEntry(
        id=str(uuid.uuid4()),
        scope_key=scope,
        namespace=namespace,
        model_id="mock-upstream",
        exact_key=exact or f"{scope}:{text}",
        query_text=text,
        query_norm=text,
        embedding=vector or unit(1.0, dim=dim),
        answer_text=answer,
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.001,
        ttl_class=TTLClass.NORMAL,
        created_at=now,
        expires_at=now + expires_in,
    )
