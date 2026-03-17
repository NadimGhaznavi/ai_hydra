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

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_MEM_SIZE = DMemory.MAX_MEM_SIZE  # Max Linear transitions
MIN_FRAMES = DMemory.MIN_FRAMES  # Min transitions before returning samples

T = TypeVar("T")


class SimpleReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
    ):
        self.log = HydraLog(
            client_id=DModule.SIMPLE_REPLAY_MEMORY,
            log_level=log_level,
            to_console=True,
        )
        self._rng = rng

        # Linear model settings
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)
        self._memory_not_full = True
        self._memory_cold = True

        self.log.info("Initialized")
        self.log.info(
            f"Setting maximum number of stored transitions to {MAX_MEM_SIZE}"
        )
        self.log.info(f"Setting minimum number of transitions to {MIN_FRAMES}")

    def append(self, t: Transition) -> None:
        """Add a transition into memory"""

        self._memory.append(t)
        if len(self._memory) >= MAX_MEM_SIZE and self._memory_not_full:
            self.log.info(
                f"Memory has been filled to capacity: {MAX_MEM_SIZE} transitions"
            )
            self._memory_not_full = False

    def sample_transitions(self, batch_size: int) -> list[Transition] | None:
        """Sample random transitions for a linear model."""

        if len(self._memory) < max(MIN_FRAMES, batch_size):
            return None

        if self._memory_cold:
            self.log.debug("Memory has warmed up")
            self._memory_cold = False

        return self._rng.sample(list(self._memory), batch_size)
