"""The storage contract every backend implements (SPEC section 10.1)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from semcache.types import CacheEntry


class CacheStore(Protocol):
    name: str

    def put(self, entry: CacheEntry) -> None:
        """Insert, or replace an entry with the same exact_key."""

    def get_exact(self, scope_key: str, exact_key: str, now: datetime) -> CacheEntry | None:
        """The unexpired entry with this exact key inside this scope, if any."""

    def search(
        self, scope_key: str, vector: Sequence[float], k: int, now: datetime
    ) -> list[tuple[CacheEntry, float]]:
        """Up to k unexpired entries in this scope, best cosine similarity first."""

    def record_hit(self, entry_id: str, now: datetime) -> None: ...

    def invalidate(
        self,
        *,
        scope_key: str | None = None,
        namespace: str | None = None,
        entry_id: str | None = None,
    ) -> int:
        """Delete by exactly one selector; returns how many entries were removed."""

    def purge_expired(self, now: datetime) -> int: ...

    def stats(self) -> dict: ...


def one_selector(scope_key, namespace, entry_id) -> None:
    given = [x for x in (scope_key, namespace, entry_id) if x]
    if len(given) != 1:
        # Why strict: an empty or ambiguous selector must never mean "delete everything".
        raise ValueError("give exactly one of scope_key, namespace, entry_id")
