"""A hard cap on live Bedrock spend across all eval runs (SPEC section 12.6).

Every real call first checks that its *estimated* cost fits under the cap, then records its
actual cost. The running total lives in a small JSON file so the cap holds across runs.

Several processes may share the file (for example two embedding jobs in different regions),
so updates take a lock file and writes are atomic: a reader never sees a half-written file.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path


class BudgetExceededError(RuntimeError):
    pass


class LiveBudget:
    STALE_LOCK_S = 30.0  # a lock older than this was left by a crashed process

    def __init__(self, path: str | Path, limit_usd: float):
        self.path = Path(path)
        self.limit_usd = limit_usd
        self._lock = threading.Lock()
        self._lockfile = self.path.with_name(self.path.name + ".lock")

    def _read(self) -> dict:
        data = {"spent_usd": 0.0, "calls": 0}
        for attempt in range(20):
            try:
                if self.path.exists():
                    data.update(json.loads(self.path.read_text(encoding="utf-8")))
                return data
            except (json.JSONDecodeError, PermissionError):
                # A writer without the lock (an older version) may be mid-write.
                if attempt == 19:
                    raise
                time.sleep(0.05)
        return data

    def spent(self) -> float:
        return float(self._read()["spent_usd"])

    def check(self, estimated_usd: float) -> None:
        """Raise before a call whose estimated cost would push spend over the cap."""
        spent = self.spent()
        if spent + estimated_usd > self.limit_usd:
            raise BudgetExceededError(
                f"Live budget would be exceeded: spent ${spent:.4f} + estimated "
                f"${estimated_usd:.4f} > cap ${self.limit_usd:.2f} (LIVE_BUDGET_USD)."
            )

    @contextlib.contextmanager
    def _process_lock(self, timeout_s: float = 10.0):
        """Cross-process lock: whoever creates the lock file first owns it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(self._lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except (FileExistsError, PermissionError):
                # Windows raises PermissionError while another process is deleting the file.
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    if time.time() - self._lockfile.stat().st_mtime > self.STALE_LOCK_S:
                        self._lockfile.unlink()
                        continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not lock {self.path}") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            for _ in range(20):
                try:
                    self._lockfile.unlink()
                    break
                except FileNotFoundError:
                    break
                except PermissionError:
                    time.sleep(0.01)

    def add(self, actual_usd: float, calls: int = 1) -> None:
        with self._lock, self._process_lock():
            data = self._read()
            data["spent_usd"] = round(data["spent_usd"] + actual_usd, 8)
            data["calls"] = int(data["calls"]) + calls
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            for attempt in range(20):
                try:
                    os.replace(tmp, self.path)  # atomic: readers see the old or the new file
                    break
                except PermissionError:
                    # Windows refuses the rename while another process has the file open.
                    if attempt == 19:
                        raise
                    time.sleep(0.05)
