"""Collapse identical concurrent work into one call (cache stampede protection, SPEC 9.1).

If 50 identical requests arrive while the first is still waiting on Bedrock, only the first
(the "leader") does the work; the other 49 await its result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class SingleFlight:
    def __init__(self, timeout_s: float = 30.0):
        self.timeout_s = timeout_s
        self._inflight: dict[str, asyncio.Future] = {}

    async def do(self, key: str, work: Callable[[], Awaitable[Any]]) -> tuple[Any, bool]:
        """Run `work` once per key at a time. Returns (result, was_shared)."""
        existing = self._inflight.get(key)
        if existing is not None:
            try:
                # shield: a waiter timing out must not cancel the leader's work
                result = await asyncio.wait_for(asyncio.shield(existing), self.timeout_s)
                return result, True
            except TimeoutError:
                # A stuck leader must not hang everyone: do the work ourselves.
                return await work(), False

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            result = await work()
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved, so no "never retrieved" warning
            raise
        else:
            future.set_result(result)
            return result, False
        finally:
            self._inflight.pop(key, None)

    def in_flight(self) -> int:
        return len(self._inflight)
