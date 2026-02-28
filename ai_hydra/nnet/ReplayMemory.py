# ai_hydra/nnet/ReplayMemory.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Deque, List

from ai_hydra.nnet.Transition import Transition


class ReplayMemory:
    def __init__(self, capacity: int, rng: Random):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf: Deque[Transition] = deque(maxlen=capacity)
        self._rng = rng

    def __len__(self) -> int:
        return len(self._buf)

    def add(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size > len(self._buf):
            raise ValueError("batch_size > buffer size")
        # random.sample works on sequences; convert once for simplicity
        return self._rng.sample(list(self._buf), k=batch_size)
