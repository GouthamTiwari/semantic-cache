"""Postgres + pgvector store (SPEC section 10.2).

One connection guarded by a lock keeps v1 simple and correct; a connection pool is the
obvious next step for high concurrency.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from datetime import datetime
from importlib import resources

import numpy as np

from semcache.store.base import one_selector
from semcache.types import CacheEntry, TTLClass

_COLUMNS = (
    "id, scope_key, namespace, model_id, exact_key, query_text, query_norm, embedding, "
    "answer_text, stop_reason, input_tokens, output_tokens, cost_usd, ttl_class, created_at, "
    "expires_at, hit_count, last_hit_at, extra"
)


def _to_list(value) -> list[float]:
    # pgvector returns either a numpy array or its own Vector type, depending on version.
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32).tolist()


class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str, dim: int, *, ef_search: int = 64):
        import psycopg
        from pgvector.psycopg import register_vector

        if not 1 <= dim <= 2000:  # pgvector's HNSW limit for the `vector` type
            raise ValueError("unsupported dimension")
        self.dim = int(dim)  # int() so the f-string table name can only ever hold digits
        self.table = f"semcache_entries_{self.dim}"
        self._lock = threading.Lock()
        # Fail fast when the database is down, so the cache can fail open instead of hanging.
        self._conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self.migrate()
        self._conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        try:
            # pgvector >= 0.8: keep scanning the index until enough rows pass the scope
            # filter, so a busy neighbouring scope cannot starve our results.
            self._conn.execute("SET hnsw.iterative_scan = relaxed_order")
        except Exception:  # older pgvector: filtering still correct, may return fewer rows
            pass

    def migrate(self) -> None:
        sql = resources.files("semcache.store").joinpath("schema.sql").read_text("utf-8")
        with self._lock:
            self._conn.execute(sql.replace("{dim}", str(self.dim)))

    @staticmethod
    def _row_to_entry(row) -> CacheEntry:
        return CacheEntry(
            id=str(row[0]),
            scope_key=row[1],
            namespace=row[2],
            model_id=row[3],
            exact_key=row[4],
            query_text=row[5],
            query_norm=row[6],
            embedding=_to_list(row[7]),
            answer_text=row[8],
            stop_reason=row[9],
            input_tokens=row[10],
            output_tokens=row[11],
            cost_usd=row[12],
            ttl_class=TTLClass(row[13]),
            created_at=row[14],
            expires_at=row[15],
            hit_count=row[16],
            last_hit_at=row[17],
            extra=row[18] or {},
        )

    def put(self, e: CacheEntry) -> None:
        vec = np.asarray(e.embedding, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"embedding has {vec.shape[0]} dims, table expects {self.dim}")
        with self._lock:
            self._conn.execute(
                f"""INSERT INTO {self.table} ({_COLUMNS}) VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (exact_key) DO UPDATE SET
                  id = EXCLUDED.id, scope_key = EXCLUDED.scope_key,
                  namespace = EXCLUDED.namespace, model_id = EXCLUDED.model_id,
                  query_text = EXCLUDED.query_text, query_norm = EXCLUDED.query_norm,
                  embedding = EXCLUDED.embedding, answer_text = EXCLUDED.answer_text,
                  stop_reason = EXCLUDED.stop_reason, input_tokens = EXCLUDED.input_tokens,
                  output_tokens = EXCLUDED.output_tokens, cost_usd = EXCLUDED.cost_usd,
                  ttl_class = EXCLUDED.ttl_class, created_at = EXCLUDED.created_at,
                  expires_at = EXCLUDED.expires_at, hit_count = 0, last_hit_at = NULL,
                  extra = EXCLUDED.extra""",
                (
                    e.id, e.scope_key, e.namespace, e.model_id, e.exact_key, e.query_text,
                    e.query_norm, vec, e.answer_text, e.stop_reason, e.input_tokens,
                    e.output_tokens, e.cost_usd, e.ttl_class.value, e.created_at, e.expires_at,
                    e.hit_count, e.last_hit_at, json.dumps(e.extra),
                ),
            )  # fmt: skip

    def get_exact(self, scope_key: str, exact_key: str, now: datetime) -> CacheEntry | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM {self.table} "
                "WHERE exact_key = %s AND scope_key = %s AND expires_at > %s",
                (exact_key, scope_key, now),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def search(
        self, scope_key: str, vector: Sequence[float], k: int, now: datetime
    ) -> list[tuple[CacheEntry, float]]:
        vec = np.asarray(vector, dtype=np.float32)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS score FROM {self.table} "
                "WHERE scope_key = %s AND expires_at > %s "
                "ORDER BY embedding <=> %s LIMIT %s",
                (vec, scope_key, now, vec, int(k)),
            ).fetchall()
        out = [(self._row_to_entry(r[:-1]), float(r[-1])) for r in rows]
        out.sort(key=lambda pair: -pair[1])  # relaxed_order may return near-sorted rows
        return out

    def record_hit(self, entry_id: str, now: datetime) -> None:
        with self._lock:
            self._conn.execute(
                f"UPDATE {self.table} SET hit_count = hit_count + 1, last_hit_at = %s "
                "WHERE id = %s",
                (now, entry_id),
            )

    def invalidate(self, *, scope_key=None, namespace=None, entry_id=None) -> int:
        one_selector(scope_key, namespace, entry_id)
        column, value = (
            ("scope_key", scope_key)
            if scope_key
            else ("namespace", namespace)
            if namespace
            else ("id", entry_id)
        )
        with self._lock:
            cur = self._conn.execute(f"DELETE FROM {self.table} WHERE {column} = %s", (value,))
            return cur.rowcount

    def purge_expired(self, now: datetime) -> int:
        with self._lock:
            cur = self._conn.execute(f"DELETE FROM {self.table} WHERE expires_at <= %s", (now,))
            return cur.rowcount

    def stats(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                f"SELECT count(*), count(DISTINCT scope_key), coalesce(sum(hit_count), 0), "
                f"coalesce(array_agg(DISTINCT namespace), '{{}}') FROM {self.table}"
            ).fetchone()
        return {
            "store": self.name,
            "entries": int(row[0]),
            "scopes": int(row[1]),
            "namespaces": sorted(row[3] or []),
            "total_hits": int(row[2]),
        }

    def truncate(self) -> None:
        """Delete every row (tests only)."""
        with self._lock:
            self._conn.execute(f"TRUNCATE {self.table}")

    def close(self) -> None:
        self._conn.close()
