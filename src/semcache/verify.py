"""Grey-zone verifier: ask a small model whether two questions need the same answer (SPEC 8).

Only used for scores between t_low and t_high. Verdicts are memoised by the unordered pair
of normalised questions, so the same pair is never judged twice.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from semcache.bedrock import with_retries
from semcache.budget import LiveBudget
from semcache.normalize import normalize, sha256
from semcache.pricing import cost_usd

JUDGE_SYSTEM = (
    "You decide whether two user questions can be answered by exactly the same answer. "
    "Differences in numbers, dates, negation, products, places or organisations mean NO. "
    "Different wording of the same question means YES. Reply with one word: YES or NO."
)


@dataclass
class Verdict:
    same: bool
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    raw: str = ""


class Verifier(Protocol):
    def verify(self, a: str, b: str) -> Verdict: ...


def parse_verdict(text: str) -> bool:
    """Only a clear YES counts as same; anything else is NO (fail safe)."""
    words = text.strip().strip(".!\"'`*").split()
    return bool(words) and words[0].strip(".,!:;\"'`*").upper() == "YES"


def pair_key(a: str, b: str) -> str:
    x, y = sorted([normalize(a), normalize(b)])
    return sha256(x + "\x1f" + y)


class LLMVerifier:
    def __init__(
        self,
        client,
        model_id: str,
        *,
        memo_path: str | Path | None = None,
        budget: LiveBudget | None = None,
        report_cost_on_hit: bool = False,
    ):
        self.client = client
        self.model_id = model_id
        self.budget = budget
        self.report_cost_on_hit = report_cost_on_hit
        self.calls = 0
        self._lock = threading.Lock()
        self._mem: dict[str, tuple[bool, float, float, str]] = {}
        self._db = None
        if memo_path:
            Path(memo_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(memo_path), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS verdicts (key TEXT, model TEXT, same INTEGER, "
                "cost REAL, latency_ms REAL, raw TEXT, PRIMARY KEY (key, model))"
            )
            self._db.commit()

    def _lookup(self, key: str):
        if key in self._mem:
            return self._mem[key]
        if self._db is not None:
            with self._lock:
                row = self._db.execute(
                    "SELECT same, cost, latency_ms, raw FROM verdicts WHERE key=? AND model=?",
                    (key, self.model_id),
                ).fetchone()
            if row:
                val = (bool(row[0]), float(row[1]), float(row[2]), row[3])
                self._mem[key] = val
                return val
        return None

    def verify(self, a: str, b: str) -> Verdict:
        key = pair_key(a, b)
        hit = self._lookup(key)
        if hit is not None:
            same, cost, latency, raw = hit
            if self.report_cost_on_hit:
                return Verdict(same, cost, latency, cached=True, raw=raw)
            return Verdict(same, 0.0, 0.0, cached=True, raw=raw)

        user = f"Question A: {a}\nQuestion B: {b}"
        if self.budget:
            self.budget.check(cost_usd(self.model_id, len(JUDGE_SYSTEM + user) // 3 + 1, 5))
        start = time.perf_counter()
        resp = with_retries(
            lambda: self.client.converse(
                modelId=self.model_id,
                system=[{"text": JUDGE_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0.0},
            )
        )
        latency = (time.perf_counter() - start) * 1000
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        raw = "".join(blk["text"] for blk in blocks if "text" in blk)
        usage = resp.get("usage", {})
        cost = cost_usd(
            self.model_id, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
        )
        if self.budget:
            self.budget.add(cost)
        same = parse_verdict(raw)
        self.calls += 1
        self._mem[key] = (same, cost, latency, raw)
        if self._db is not None:
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?, ?, ?)",
                    (key, self.model_id, int(same), cost, latency, raw),
                )
                self._db.commit()
        return Verdict(same, cost, latency, cached=False, raw=raw)


class StaticVerifier:
    """Test double: answers from a function, counts calls, memoises like the real one."""

    def __init__(self, decide, *, cost: float = 0.00001, latency_ms: float = 300.0):
        self.decide = decide
        self.cost = cost
        self.latency_ms = latency_ms
        self.calls = 0
        self._mem: dict[str, bool] = {}

    def verify(self, a: str, b: str) -> Verdict:
        key = pair_key(a, b)
        if key in self._mem:
            return Verdict(self._mem[key], cached=True)
        self.calls += 1
        same = bool(self.decide(a, b))
        self._mem[key] = same
        return Verdict(same, self.cost, self.latency_ms)
