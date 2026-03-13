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
from ai_hydra.constants.DNNet import DRNN

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_CHUNKS = DMemory.MAX_CHUNKS
MIN_CHUNKS = DMemory.MIN_CHUNKS
SEQ_LENGTH = DRNN.SEQ_LENGTH
SAMPLE_P_VALUE = 0.25  #### MUST Be between 0 and 1


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

        ### Linear model settings
        # No lookahead
        self._memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)
        # Lookahead
        self._lh_memory: Deque[Transition] = deque(maxlen=DMemory.MAX_MEM_SIZE)

        ### RNN model settings
        self._rnn = rnn
        self._chunks: list[list[Transition]] = []
        self._lh_chunks: list[list[Transition]] = []
        self._cur_game: list[Transition] = []
        self._lh_cur_game: list[Transition] = []
        self._max_chunks = MAX_CHUNKS
        self._seq_len = SEQ_LENGTH
        if self._rnn:
            self.log.debug("Initialized for RNN model training")
        else:
            self.log.debug("Initialized for Linear model training")

    def append(self, t: Transition, lookahead: bool) -> None:
        """Add a transition into the lookahead or no lookahead memory"""

        ### IMPORTANT NOTE
        ###
        ### In RNN mode, lookahead must be fixed for the whole episode.
        ### If lookahead were allowed to change per step, an episode could be
        ### split across LH and NLH buffers, breaking temporal continuity of
        ### the training sequences.

        if not self._rnn:
            if lookahead:
                self._lh_memory.append(t)
            else:
                self._memory.append(t)
            return

        if lookahead:
            self._lh_cur_game.append(t)
        else:
            self._cur_game.append(t)

        if t.done:
            self._finalize_game()

    def _finalize_game(self) -> None:
        game = self._cur_game
        lh_game = self._lh_cur_game
        self._cur_game = []
        self._lh_cur_game = []

        if len(game) >= self._seq_len:
            self._chunks.extend(self._chunk_from_end(game))
            if len(self._chunks) > self._max_chunks:
                overflow = len(self._chunks) - self._max_chunks
                del self._chunks[:overflow]

        if len(lh_game) >= self._seq_len:
            self._lh_chunks.extend(self._chunk_from_end(lh_game))
            if len(self._lh_chunks) > self._max_chunks:
                overflow = len(self._lh_chunks) - self._max_chunks
                del self._lh_chunks[:overflow]

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

    def num_lh_chunks(self) -> int:
        return len(self._lh_chunks)

    def sample_chunks(
        self,
        batch_size: int,
        sample_lh_p: float = SAMPLE_P_VALUE,
    ) -> list[list[Transition]] | None:

        total_chunks = len(self._chunks) + len(self._lh_chunks)
        if total_chunks < max(batch_size, MIN_CHUNKS):
            return None

        return self._sample_mixed(
            primary=self._lh_chunks,
            secondary=self._chunks,
            batch_size=batch_size,
            primary_p=sample_lh_p,
        )

    def _sample_mixed(
        self,
        primary: Sequence[T],
        secondary: Sequence[T],
        batch_size: int,
        primary_p: float,
    ) -> list[T] | None:
        n_primary = round(batch_size * primary_p)
        n_secondary = batch_size - n_primary

        # Clamp to available sizes
        n_primary = min(n_primary, len(primary))
        n_secondary = min(n_secondary, len(secondary))

        total = n_primary + n_secondary
        if total < batch_size:
            shortfall = batch_size - total

            # Top up from whichever side still has room
            primary_room = len(primary) - n_primary
            if primary_room > 0:
                add = min(shortfall, primary_room)
                n_primary += add
                shortfall -= add

            if shortfall > 0:
                secondary_room = len(secondary) - n_secondary
                if secondary_room > 0:
                    add = min(shortfall, secondary_room)
                    n_secondary += add
                    shortfall -= add

        if n_primary + n_secondary < batch_size:
            return None

        batch = []
        if n_primary:
            batch.extend(self._rng.sample(primary, n_primary))
        if n_secondary:
            batch.extend(self._rng.sample(secondary, n_secondary))

        # Optional:
        # self._rng.shuffle(batch)

        return batch

    def sample_transitions(
        self,
        batch_size: int = DMemory.BATCH_SIZE,
        sample_lh_p: float = SAMPLE_P_VALUE,
    ) -> list[Transition] | None:
        """Sample random transitions for a linear model."""
        total_mem = len(self._memory) + len(self._lh_memory)
        if total_mem < max(DMemory.MIN_FRAMES, batch_size):
            return None

        return self._sample_mixed(
            primary=self._lh_memory,
            secondary=self._memory,
            batch_size=batch_size,
            primary_p=sample_lh_p,
        )
