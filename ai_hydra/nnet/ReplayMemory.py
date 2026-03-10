# ai_hydra/nnet/ReplayMemory.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from random import Random
from typing import Deque, List
import pickle

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DNNet import DRNN

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog


class ReplayMemory:

    def __init__(self, rng: Random, log_level: DHydraLog, use_rnn=False):
        # Our SQLite DB manager
        self.log = HydraLog(
            client_id="ReplayMemory", log_level=log_level, to_console=True
        )

        self._rng = rng
        self._use_rnn = use_rnn
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)

    def append(self, transition: Transition):
        self._memory.append(transition)

    def sample(self, batch_size=DMemory.BATCH_SIZE):

        if self._use_rnn:
            return self._sample_sequences(batch_size)
        else:
            if len(self._memory) < DMemory.MIN_FRAMES:
                return None
            return self._rng.sample(list(self._memory), k=batch_size)

    def _sample_sequences(self, batch_size):
        """Sample sequences of transitions (for RNNs)."""
        # We'll need to adjust how we sample to maintain sequential order
        indices = self._rng.sample(range(len(self.buffer) - 1), batch_size)
        sequences = []

        for idx in indices:
            # Extract a sequence (a set of transitions from the buffer)
            # Here you could define your sequence length, say 10 transitions
            sequence = self.buffer[idx : idx + DRNN.REPLAY_MEM_SEQ_SIZE]
            sequences.append(sequence)

        return sequences
