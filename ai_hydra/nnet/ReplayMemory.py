# ai_hydra/nnet/ReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from collections import deque
from random import Random
from typing import Deque

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog
from ai_hydra.constants.DNNet import DRNN, DRNN2

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_CHUNKS = DMemory.MAX_CHUNKS
MIN_CHUNKS = DMemory.MIN_CHUNKS


class ReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
        rnn2: bool = False,
        seq_len: int = DRNN2.SEQ_LENGTH,
    ):
        self.log = HydraLog(
            client_id="ReplayMemory",
            log_level=log_level,
            to_console=True,
        )
        self._rng = rng
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)

        # RNN 2 settings
        self._rnn2 = rnn2
        self._seq_len = seq_len
        self._chunks: list[list[Transition]] = []
        self._cur_game: list[Transition] = []
        self._max_chunks = MAX_CHUNKS
        self._mem_loaded_flag = True
        self.log.debug("Initialized...")

    def append(self, t: Transition) -> None:
        """Add a transition."""
        if not self._rnn2:
            self._memory.append(t)
            return

        self._cur_game.append(t)

        if t.done:
            self._finalize_game()

    def _finalize_game(self) -> None:
        game = self._cur_game
        self._cur_game = []

        if len(game) < self._seq_len:
            return

        self._chunks.extend(self._chunk_from_end(game))

        if len(self._chunks) > self._max_chunks:
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

    def sample_chunks(self, batch_size: int) -> list[list[Transition]] | None:
        if len(self._chunks) < max(batch_size, MIN_CHUNKS):
            return None

        if self._mem_loaded_flag:
            self.log.debug("Warm up complete")
            self._mem_loaded_flag = False

        return self._rng.sample(self._chunks, batch_size)

    def num_chunks(self) -> int:
        return len(self._chunks)

    def sample_transitions(
        self, batch_size: int = DMemory.BATCH_SIZE
    ) -> list[Transition] | None:
        """Sample random transitions for a linear model."""
        if len(self._memory) < max(DMemory.MIN_FRAMES, batch_size):
            return None

        memory = list(self._memory)
        return self._rng.sample(memory, k=batch_size)

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int = DRNN.REPLAY_MEM_SEQ_SIZE,
    ) -> list[list[Transition]] | None:
        """Sample contiguous transition sequences for an RNN."""
        memory = list(self._memory)

        if len(memory) < max(DMemory.MIN_FRAMES, seq_len):
            return None

        max_start = len(memory) - seq_len + 1
        if max_start <= 0:
            return None

        valid_starts: list[int] = []
        for start in range(max_start):
            window = memory[start : start + seq_len]

            # Reject windows that cross an episode boundary before the end
            if any(t.done for t in window[:-1]):
                continue

            valid_starts.append(start)

        if not valid_starts:
            return None

        actual_batch_size = min(batch_size, len(valid_starts))
        starts = self._rng.sample(valid_starts, k=actual_batch_size)

        # self.log.debug(
        #    f"sample_sequences(): memory={len(memory)}, valid_starts={len(valid_starts)}, "
        #    f"batch_size={actual_batch_size}, seq_len={seq_len}"
        # )
        return [memory[start : start + seq_len] for start in starts]
