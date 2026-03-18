# ai_hydra/nnet/ReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from __future__ import annotations

import traceback
from random import Random
from dataclasses import dataclass, field

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog

MAX_FRAMES = DMemory.MAX_MEM_SIZE  # Max Linear transitions
MAX_BUCKETS = 20
THRESHOLDS_REQUIRED = 5

# Gear => (promotion_score, seq_length, batch_size)
GEARBOX = {
    1: (10, 4, 128),
    2: (20, 8, 64),
    3: (30, 16, 32),
    4: (40, 32, 16),
    5: (50, 64, 8),
}


@dataclass(slots=True, frozen=True)
class Episode:
    frames: tuple[Transition, ...] = field(default_factory=list)
    size: int


class ReplayMemory:
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
    ):
        self.log = HydraLog(
            client_id="ReplayMemory",
            log_level=log_level,
            to_console=True,
        )
        self._rng = rng

        self._cur_game: list[Transition] = []
        self._games: list[Episode] = []
        self._stored_frames = 0

        self._cur_gear = 1
        _, self._cur_seq_length, self._batch_size = GEARBOX[self._cur_gear]

        self._threshold_achieved_count = 0

        self._has_logged_pruning = False

        self.log.info("Initialized")
        self.log.info(f"Setting max_frames to {MAX_FRAMES}")
        self.log.info(f"Setting max_buckets to {MAX_BUCKETS}")

    def append(self, t: Transition, final_score=None) -> None:
        """Add a transition into memory"""

        self._cur_game.append(t)

        if t.done:
            if final_score is None:
                raise ValueError(
                    "final_score must be provided when t.done is True"
                )
            self._finalize_game()
            self._prune_if_needed()
            self._check_gear(final_score)

    def _check_gear(self, final_score: int) -> None:
        if self._cur_gear == max(GEARBOX):
            return

        threshold, _, _ = GEARBOX[self._cur_gear]
        if final_score >= threshold:
            self._threshold_achieved_count += 1

            if self._threshold_achieved_count >= THRESHOLDS_REQUIRED:
                self._cur_gear += 1
                self._threshold_achieved_count = 0

                _, self._cur_seq_length, self._batch_size = GEARBOX[
                    self._cur_gear
                ]
                self.log.info(
                    f"Shifting into higher gear: {self._cur_gear} "
                    f"(seq_length={self._cur_seq_length}, batch_size={self._batch_size})"
                )

    def _chunk_range_for_bucket(
        self, ep_size: int, seq_length: int, bucket_idx: int
    ) -> tuple[int, int] | None:
        pass

    def _finalize_game(self) -> None:
        frames = tuple(self._cur_game)
        self._cur_game = []

        ep_size = len(frames)
        self._stored_frames += ep_size

        self._games.append(
            Episode(
                frames=frames,
                size=ep_size,
            )
        )

    def _num_chunks_for_episode(self, ep_size: int, seq_length: int) -> int:
        return ep_size // seq_length

    def _prune_if_needed(self) -> None:
        while self._stored_frames > MAX_FRAMES:
            episode = self._games.pop(0)
            self._stored_frames -= episode.size

            if not self._has_logged_pruning:
                self.log.info("Memory is full, pruning initiated")
                self._has_logged_pruning = True
