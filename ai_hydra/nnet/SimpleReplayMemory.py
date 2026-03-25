# ai_hydra/nnet/SimpleReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

### ----- Version II -----

from __future__ import annotations

from collections import deque
from random import Random
from typing import Deque, TypeVar

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DEvent import EV_STATUS

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import EventMsg, HydraEventMQ


MAX_MEM_SIZE = DMemory.MAX_MEM_SIZE  # Max Linear transitions
MIN_FRAMES = DMemory.MIN_FRAMES  # Min transitions before returning samples

T = TypeVar("T")


class SimpleReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
        pub_func,
    ):
        self.log = HydraLog(
            client_id=DModule.SIMPLE_REPLAY_MEMORY,
            log_level=log_level,
            to_console=True,
        )
        self._rng = rng
        self.event = HydraEventMQ(
            client_id=DModule.SIMPLE_REPLAY_MEMORY, pub_func=pub_func
        )

        # Linear model settings
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)
        self._memory_not_full = True
        self._memory_cold = True
        self._has_logged_startup = False

    async def _log_startup(self):
        self.log.info("Initialized")

        msg = f"Setting maximum number of stored transitions to {MAX_MEM_SIZE}"
        self.log.info(msg)
        await self.event.publish(EventMsg(level=EV_STATUS.INFO, message=msg))

        msg = f"Setting minimum number of transitions to {MIN_FRAMES}"
        self.log.info(msg)
        await self.event.publish(EventMsg(level=EV_STATUS.INFO, message=msg))

    async def append(self, t: Transition) -> None:
        """Add a transition into memory"""

        if not self._has_logged_startup:
            await self._log_startup()
            self._has_logged_startup = True

        self._memory.append(t)
        if len(self._memory) >= MAX_MEM_SIZE and self._memory_not_full:
            msg = f"Memory has been filled to capacity: {MAX_MEM_SIZE} transitions"
            self.log.info(msg)
            await self.event.publish(
                EventMsg(level=EV_STATUS.INFO, message=msg)
            )
            self._memory_not_full = False

    async def sample_transitions(
        self, batch_size: int
    ) -> list[Transition] | None:
        """Sample random transitions for a linear model."""

        if len(self._memory) < max(MIN_FRAMES, batch_size):
            return None

        if self._memory_cold:
            msg = "Memory has warmed up"
            self.log.debug(msg)
            await self.event.publish(
                EventMsg(level=DHydraLog.INFO, message=msg)
            )
            self._memory_cold = False

        return self._rng.sample(list(self._memory), batch_size)
