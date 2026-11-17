"""Embedding clients: Titan v2 on Bedrock, an on-disk memo, and an offline fake.

The memo matters for cost and honesty: the eval embeds every text once, and every later
run reads the same vectors from disk, so results are repeatable and free to regenerate.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from semcache.bedrock import with_retries
from semcache.budget import LiveBudget
from semcache.pricing import cost_usd

MAX_INPUT_CHARS = 30_000  # Titan v2 takes up to 8K tokens; stay well below


@dataclass
class EmbedResult:
    vector: np.ndarray  # float32, L2-normalised, so cosine similarity = dot product
    input_tokens: int
    latency_ms: float  # time of the real call (a memo hit reports the original call's time)
    cached: bool = False  # True = served from the memo, no call made
    cost_usd: float = 0.0  # cost of this lookup: 0 when served from the memo


class Embedder(Protocol):
    model_id: str
    dimensions: int

    def embed(self, text: str) -> EmbedResult: ...


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


class _Pacer:
    """Spaces calls to at most `per_second` per second, across threads (Titan quota)."""

    def __init__(self, per_second: float, sleep: Callable[[float], None], clock=time.monotonic):
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self.sleep, self.clock = sleep, clock
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = self.clock()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        if slot > now:
            self.sleep(slot - now)


class TitanEmbedder:
    """Amazon Titan Text Embeddings v2 via InvokeModel."""

    def __init__(
        self,
        client,
        *,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        requests_per_second: float = 0.0,
        budget: LiveBudget | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if dimensions not in (256, 512, 1024):
            raise ValueError("Titan v2 supports 256, 512 or 1024 dimensions")
        self.client = client  # passed in so tests can hand in a fake
        self.model_id = model_id
        self.dimensions = dimensions
        self.budget = budget
        self.sleep = sleep
        self.pacer = _Pacer(requests_per_second, sleep)

    def embed(self, text: str) -> EmbedResult:
        text = text[:MAX_INPUT_CHARS]
        if self.budget:
            self.budget.check(cost_usd(self.model_id, max(1, len(text) // 3)))
        body = json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True})

        def call():
            self.pacer.wait()
            return self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

        start = time.perf_counter()
        response = with_retries(call, sleep=self.sleep)
        latency_ms = (time.perf_counter() - start) * 1000
        payload = json.loads(response["body"].read())
        tokens = int(payload.get("inputTextTokenCount", 0))
        cost = cost_usd(self.model_id, tokens)
        if self.budget:
            self.budget.add(cost)
        vector = _unit(np.asarray(payload["embedding"], dtype=np.float32))
        return EmbedResult(vector, tokens, latency_ms, cached=False, cost_usd=cost)


class MemoEmbedder:
    """Wraps any embedder with an on-disk SQLite memo keyed by (model, dimension, text).

    `report_cost_on_hit=True` makes a memo hit report the cost and latency of the original
    call. The eval uses that, so accounting reflects production (where every lookup pays
    for one embedding) while the run itself is free.
    """

    def __init__(self, inner: Embedder, path: str | Path, *, report_cost_on_hit: bool = False):
        self.inner = inner
        self.model_id = inner.model_id
        self.dimensions = inner.dimensions
        self.report_cost_on_hit = report_cost_on_hit
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, model TEXT, dim INTEGER, "
            "text TEXT, vector BLOB, tokens INTEGER, latency_ms REAL)"
        )
        self._db.commit()
        self.calls = 0  # real calls made through this memo (for tests and logs)

    def _key(self, text: str) -> str:
        raw = f"{self.model_id}\x1f{self.dimensions}\x1f{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str) -> EmbedResult | None:
        with self._lock:
            row = self._db.execute(
                "SELECT vector, tokens, latency_ms FROM memo WHERE key = ?", (self._key(text),)
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype=np.float32).copy()
        tokens, latency = int(row[1]), float(row[2])
        cost = cost_usd(self.model_id, tokens) if self.report_cost_on_hit else 0.0
        return EmbedResult(vector, tokens, latency, cached=True, cost_usd=cost)

    def embed(self, text: str) -> EmbedResult:
        hit = self.get(text)
        if hit is not None:
            return hit
        result = self.inner.embed(text)
        self.calls += 1
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO memo VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self._key(text),
                    self.model_id,
                    self.dimensions,
                    text,
                    result.vector.astype(np.float32).tobytes(),
                    result.input_tokens,
                    result.latency_ms,
                ),
            )
            self._db.commit()
        return result

    def close(self) -> None:
        self._db.close()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Offline, deterministic stand-in for tests and demos (no AWS needed).

    Hashes words and character trigrams into a fixed-size vector, so texts that share
    words score high. It is NOT a semantic model: never use it for the reported results.
    """

    model_id = "mock-embed"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, text: str) -> EmbedResult:
        v = np.zeros(self.dimensions, dtype=np.float32)
        words = _TOKEN_RE.findall(text.lower())
        feats = words + [w[i : i + 3] for w in words for i in range(max(1, len(w) - 2))]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dimensions] += 1.0 if (h >> 63) & 1 else -1.0
        tokens = max(1, len(words))
        return EmbedResult(_unit(v), tokens, 0.0, cost_usd=cost_usd(self.model_id, tokens))
