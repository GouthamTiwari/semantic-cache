"""In-process store backed by numpy. Used by tests, the eval and single-process demos."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from semcache.store.base import one_selector
from semcache.types import CacheEntry


class InMemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}  # id -> entry
        self._by_exact: dict[str, str] = {}  # exact_key -> id
        # Per scope: entry ids and a matrix of their vectors, so a search is one matrix product.
        self._scope_ids: dict[str, list[str]] = {}
        self._scope_vecs: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    # --- writes ------------------------------------------------------------------
    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            old_id = self._by_exact.get(entry.exact_key)
            if old_id is not None:
                self._remove(old_id)
            self._entries[entry.id] = entry
            self._by_exact[entry.exact_key] = entry.id
            vec = np.asarray(entry.embedding, dtype=np.float32)[None, :]
            ids = self._scope_ids.setdefault(entry.scope_key, [])
            ids.append(entry.id)
            mat = self._scope_vecs.get(entry.scope_key)
            self._scope_vecs[entry.scope_key] = vec if mat is None else np.vstack([mat, vec])

    def _remove(self, entry_id: str) -> None:
        entry = self._entries.pop(entry_id, None)
        if entry is None:
            return
        self._by_exact.pop(entry.exact_key, None)
        ids = self._scope_ids[entry.scope_key]
        i = ids.index(entry_id)
        ids.pop(i)
        self._scope_vecs[entry.scope_key] = np.delete(self._scope_vecs[entry.scope_key], i, axis=0)
        if not ids:
            del self._scope_ids[entry.scope_key]
            del self._scope_vecs[entry.scope_key]

    def record_hit(self, entry_id: str, now: datetime) -> None:
        with self._lock:
            e = self._entries.get(entry_id)
            if e is not None:
                e.hit_count += 1
                e.last_hit_at = now

    def invalidate(self, *, scope_key=None, namespace=None, entry_id=None) -> int:
        one_selector(scope_key, namespace, entry_id)
        with self._lock:
            victims = [
                e.id
                for e in self._entries.values()
                if (scope_key and e.scope_key == scope_key)
                or (namespace and e.namespace == namespace)
                or (entry_id and e.id == entry_id)
            ]
            for v in victims:
                self._remove(v)
            return len(victims)

    def purge_expired(self, now: datetime) -> int:
        with self._lock:
            victims = [e.id for e in self._entries.values() if e.is_expired(now)]
            for v in victims:
                self._remove(v)
            return len(victims)

    # --- reads -------------------------------------------------------------------
    def get_exact(self, scope_key: str, exact_key: str, now: datetime) -> CacheEntry | None:
        with self._lock:
            entry_id = self._by_exact.get(exact_key)
            if entry_id is None:
                return None
            e = self._entries[entry_id]
            if e.scope_key != scope_key or e.is_expired(now):
                return None
            return e

    def search(
        self, scope_key: str, vector: Sequence[float], k: int, now: datetime
    ) -> list[tuple[CacheEntry, float]]:
        with self._lock:
            mat = self._scope_vecs.get(scope_key)
            if mat is None:
                return []
            ids = list(self._scope_ids[scope_key])
            # Vectors are unit length, so the dot product is the cosine similarity.
            scores = mat @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)
        out: list[tuple[CacheEntry, float]] = []
        for i in order:
            e = self._entries.get(ids[i])
            if e is None or e.is_expired(now):
                continue  # expired rows are ignored at query time; purge deletes them later
            out.append((e, float(scores[i])))
            if len(out) == k:
                break
        return out

    def stats(self) -> dict:
        with self._lock:
            entries = list(self._entries.values())
        return {
            "store": self.name,
            "entries": len(entries),
            "scopes": len({e.scope_key for e in entries}),
            "namespaces": sorted({e.namespace for e in entries}),
            "total_hits": sum(e.hit_count for e in entries),
        }

    def __len__(self) -> int:
        return len(self._entries)
