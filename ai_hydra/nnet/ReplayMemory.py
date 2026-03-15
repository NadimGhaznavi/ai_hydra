# ai_hydra/nnet/ReplayMemory.py
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
from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DNNet import DRNN

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_CHUNKS = DMemory.MAX_CHUNKS
MIN_CHUNKS = DMemory.MIN_CHUNKS
SEQ_LENGTH = DRNN.SEQ_LENGTH

MAX_MEM_SIZE = DMemory.MAX_MEM_SIZE  # Max Linear transitions
MIN_FRAMES = DMemory.MIN_FRAMES  # Min transitions before returning samples

T = TypeVar("T")


class ReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
        rnn: bool = False,
        seq_length=None,
    ):
        self.log = HydraLog(
            client_id="ReplayMemory",
            log_level=log_level,
            to_console=True,
        )
        self._rng = rng

        # Linear model settings
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)

        ### RNN model settings
        self._rnn = rnn
        self._chunks: list[list[Transition]] = []
        self._cur_game: list[Transition] = []
        self._max_chunks = MAX_CHUNKS
        self._seq_len = seq_length
        self._memory_cold = True
        self._memory_not_full = True
        if self._rnn:
            if seq_length is None:
                raise ValueError("Sequence length must be set")
            self.log.info("Initialized for RNN model training")
            self.log.info(f"Setting sequence length to {seq_length}")
            self.log.info(
                f"Setting maximum number of stored sequence to {MAX_CHUNKS}"
            )
        else:
            self.log.info("Initialized for Linear model training")
            self.log.info(
                f"Setting maximum number of stored transitions to {MAX_MEM_SIZE}"
            )

    def append(self, t: Transition) -> None:
        """Add a transition into memory"""

        ### IMPORTANT NOTE
        ###
        ### In RNN mode, lookahead must be fixed for the whole episode.
        ### If lookahead were allowed to change per step, an episode could be
        ### split across LH and NLH buffers, breaking temporal continuity of
        ### the training sequences.
        if not self._rnn:
            self._memory.append(t)
            if len(self._memory) >= MAX_MEM_SIZE:
                if self._memory_not_full:
                    self.log.info(
                        f"Memory has been filled to capacity: {MAX_MEM_SIZE} transitions"
                    )
                    self._memory_not_full = False
            return

        self._cur_game.append(t)
        if t.done:
            self._finalize_game()

    def _finalize_game(self) -> None:
        game = self._cur_game
        self._cur_game = []

        if len(game) >= self._seq_len:
            self._chunks.extend(self._chunk_from_end(game))
            if len(self._chunks) > self._max_chunks:
                if self._memory_not_full:
                    self.log.info(
                        f"Memory has been filled to capacity: {self._max_chunks}"
                    )
                    self._memory_not_full = False
                overflow = len(self._chunks) - self._max_chunks
                del self._chunks[:overflow]

    def _chunk_from_end(
        self, game: list[Transition]
    ) -> list[list[Transition]]:
        n = len(game)
        rem = n % self._seq_len
        start = rem if rem != 0 else 0
        chunks: list[list[Transition]] = []
        for i in range(start, n, self._seq_len):
            chunk = game[i : i + self._seq_len]
            if len(chunk) == self._seq_len:
                chunks.append(chunk)
        return chunks

    def num_chunks(self) -> int:
        return len(self._chunks)

    def sample_chunks(
        self,
        batch_size: int,
    ) -> list[list[Transition]] | None:

        # This only checks whether a full batch is possible in aggregate.
        # The requested LH/NLH ratio is best-effort and is resolved in
        # _sample_mixed().
        if len(self._chunks) < max(batch_size, MIN_CHUNKS):
            return None

        if self._memory_cold:
            self.log.debug("Memory has warmed up")
            self._memory_cold = False

        return self._rng.sample(self._chunks, batch_size)

    def sample_transitions(self, batch_size: int) -> list[Transition] | None:
        """Sample random transitions for a linear model."""

        # This only checks whether a full batch is possible in aggregate.
        # The requested LH/NLH ratio is best-effort and is resolved in
        # _sample_mixed().
        if len(self._memory) < max(MIN_FRAMES, batch_size):
            return None

        if self._memory_cold:
            self.log.debug("Memory has warmed up")
            self._memory_cold = False

        return self._rng.sample(self._memory, batch_size)
