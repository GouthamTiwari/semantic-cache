"""A hard cap on live Bedrock spend across all eval runs (SPEC section 12.6).

Every real call first checks that its *estimated* cost fits under the cap, then records its
actual cost. The running total lives in a small JSON file so the cap holds across runs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class BudgetExceededError(RuntimeError):
    pass


class LiveBudget:
    def __init__(self, path: str | Path, limit_usd: float):
        self.path = Path(path)
        self.limit_usd = limit_usd
        self._lock = threading.Lock()

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        return float(json.loads(self.path.read_text(encoding="utf-8")).get("spent_usd", 0.0))

    def check(self, estimated_usd: float) -> None:
        """Raise before a call whose estimated cost would push spend over the cap."""
        spent = self.spent()
        if spent + estimated_usd > self.limit_usd:
            raise BudgetExceededError(
                f"Live budget would be exceeded: spent ${spent:.4f} + estimated "
                f"${estimated_usd:.4f} > cap ${self.limit_usd:.2f} (LIVE_BUDGET_USD)."
            )

    def add(self, actual_usd: float, calls: int = 1) -> None:
        with self._lock:
            data = {"spent_usd": 0.0, "calls": 0}
            if self.path.exists():
                data.update(json.loads(self.path.read_text(encoding="utf-8")))
            data["spent_usd"] = round(data["spent_usd"] + actual_usd, 8)
            data["calls"] = int(data["calls"]) + calls
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
