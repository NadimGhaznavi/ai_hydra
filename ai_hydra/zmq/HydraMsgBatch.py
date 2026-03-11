# ai_hydra/zmq/HydraMsgBatch.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0
#

import asyncio
import time

from ai_hydra.constants.DHydra import DHydraMQDef


class HydraMsgBatch:
    def __init__(self) -> None:
        self._msgs: list[dict] = []
        self._batch_num: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._timer: float | None = None

    async def acquire_lock(self) -> None:
        await self._lock.acquire()

    def append(self, payload: dict) -> None:
        assert self._lock.locked(), "ERROR: append() requires a batch lock"
        if not self._msgs:
            self._timer = time.monotonic()
        self._msgs.append(payload)

    def batch_num(self) -> int:
        return self._batch_num

    def batch_size(self) -> int:
        return len(self._msgs)

    def pop_batch(self) -> list[dict]:
        assert self._lock.locked(), "ERROR: pop_batch() requires a batch lock"
        msgs = self._msgs
        self._msgs = []
        self._timer = None
        self._batch_num += 1
        return msgs

    def has_timed_out(self, now: float) -> bool:
        if self._timer is None:
            return False
        return now - self._timer >= DHydraMQDef.MAX_BATCH_TIME

    def is_empty(self) -> bool:
        return not self._msgs

    def release_lock(self) -> None:
        self._lock.release()
