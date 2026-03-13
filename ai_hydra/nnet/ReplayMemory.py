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
from typing import Deque, Sequence, TypeVar

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DNNet import DRNN, DLinear

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_CHUNKS = DMemory.MAX_CHUNKS
MIN_CHUNKS = DMemory.MIN_CHUNKS
SEQ_LENGTH = DRNN.SEQ_LENGTH

T = TypeVar("T")


class ReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
        rnn: bool = False,
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
        self._seq_len = SEQ_LENGTH
        if self._rnn:
            self.log.debug("Initialized for RNN model training")
        else:
            self.log.debug("Initialized for Linear model training")

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
                overflow = len(self._chunks) - self._max_chunks
                del self._chunks[:overflow]
        print(f"Number of chunks {len(self._chunks)}")

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

        return self._rng.sample(self._chunks, batch_size)

    def sample_transitions(
        self,
        batch_size: int = DMemory.BATCH_SIZE,
    ) -> list[Transition] | None:
        """Sample random transitions for a linear model."""

        # This only checks whether a full batch is possible in aggregate.
        # The requested LH/NLH ratio is best-effort and is resolved in
        # _sample_mixed().
        if len(self._memory) < max(DMemory.MIN_FRAMES, batch_size):
            return None

        return self._rng.sample(self._memory, batch_size)
