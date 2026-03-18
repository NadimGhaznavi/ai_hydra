# ai_hydra/nnet/ReplayMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from __future__ import annotations

from random import Random
from dataclasses import dataclass, field

from ai_hydra.constants.DReplayMemory import DMemory
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DEvent import EV_TYPE

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ, EventMsg

MAX_FRAMES = DMemory.MAX_MEM_SIZE  # Max Linear transitions
MAX_BUCKETS = 20
THRESHOLDS_REQUIRED = 5

# Gear => (promotion_score, seq_length, batch_size)
GEARBOX = {
    1: (10, 4, 128),
    2: (25, 8, 64),
    3: (40, 16, 32),
    4: (55, 32, 16),
    5: (70, 64, 8),
}


@dataclass(slots=True, frozen=True)
class GearMeta:
    """
    This stores, for one episode and under one gear:

    - Ssequence length
    - Number of chunks
    - Slice ranges for each bucket/chunk
    """

    seq_length: int
    num_chunks: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(slots=True, frozen=True)
class Episode:
    """
    This stores the frames for one game and additional metadata.
    """

    frames: tuple[Transition, ...]
    size: int
    gear_meta: dict[int, GearMeta] = field(default_factory=dict)


class ATHReplayMemory:
    def __init__(self, rng: Random, log_level: DHydraLog, pub_func):
        self.log = HydraLog(
            client_id=DModule.ATH_REPLAY_MEMORY,
            log_level=log_level,
            to_console=True,
        )
        self.event = HydraEventMQ(
            client_id=DModule.ATH_REPLAY_MEMORY, pub_func=pub_func
        )
        self._rng = rng

        self._episodes_by_bucket: dict[int, list[Episode]] = {
            idx: [] for idx in range(MAX_BUCKETS)
        }
        self._warmed_buckets: list[int] = []

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

    async def append(
        self, t: Transition, final_score: int | None = None
    ) -> None:
        self._cur_game.append(t)

        if t.done:
            if final_score is None:
                raise ValueError(
                    "final_score must be provided when appending a terminal transition"
                )

            self._finalize_game()
            await self._prune_if_needed()

            await self._check_gear(final_score)
            self._rebuild_bucket_index()

    def _build_episode_gear_meta(
        self,
        ep_size: int,
    ) -> dict[int, GearMeta]:
        """
        Build metadata for all gears for one episode.
        """
        gear_meta: dict[int, GearMeta] = {}

        for gear, (_, seq_length, _) in GEARBOX.items():
            gear_meta[gear] = self._build_gear_meta_for_size(
                ep_size=ep_size,
                seq_length=seq_length,
            )

        return gear_meta

    def _build_gear_meta_for_size(
        self,
        ep_size: int,
        seq_length: int,
    ) -> GearMeta:
        """
        Build gear-specific chunk metadata for an episode size.

        Chunks are aligned from the end, matching the old ReplayMemory behavior.
        """
        num_chunks = ep_size // seq_length
        rem = ep_size % seq_length

        ranges: list[tuple[int, int]] = []

        for chunk_idx in range(num_chunks):
            start = rem + (chunk_idx * seq_length)
            end = start + seq_length
            ranges.append((start, end))

        return GearMeta(
            seq_length=seq_length,
            num_chunks=num_chunks,
            ranges=tuple(ranges),
        )

    async def _check_gear(self, final_score: int) -> bool:
        if self._cur_gear == max(GEARBOX):
            return False

        promotion_score, _, _ = GEARBOX[self._cur_gear]

        if final_score < promotion_score:
            return False

        self._threshold_achieved_count += 1

        if self._threshold_achieved_count < THRESHOLDS_REQUIRED:
            return False

        self._cur_gear += 1
        self._threshold_achieved_count = 0
        _, self._cur_seq_length, self._batch_size = GEARBOX[self._cur_gear]

        msg1 = f"Shifting into higher gear: {self._cur_gear}"
        msg2 = f"(seq_length={self._cur_seq_length}, batch_size={self._batch_size})"

        self.log.info(msg1)
        self.log.info(msg2)
        await self.event.publish(
            EventMsg(
                level=DHydraLog.INFO,
                message=msg1,
                ev_type=EV_TYPE.SHIFTING,
                payload={
                    DField.GEAR: self._cur_gear,
                    DField.SEQ_LENGTH: self._cur_seq_length,
                    DField.BATCH_SIZE: self._batch_size,
                },
            )
        )

        return True

    def _chunk_range_for_bucket(
        self, ep_size: int, seq_length: int, bucket_idx: int
    ) -> tuple[int, int] | None:
        pass

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_seq_length(self) -> int:
        return self._cur_seq_length

    def get_current_gear(self) -> int:
        return self._cur_gear

    def get_warmed_buckets(self) -> list[int]:
        return list(self._warmed_buckets)

    def _finalize_game(self) -> None:
        frames = tuple(self._cur_game)
        self._cur_game = []

        ep_size = len(frames)
        self._stored_frames += ep_size

        gear_meta = self._build_episode_gear_meta(ep_size)

        self._games.append(
            Episode(
                frames=frames,
                size=ep_size,
                gear_meta=gear_meta,
            )
        )

    def _num_chunks_for_episode(self, ep_size: int, seq_length: int) -> int:
        return ep_size // seq_length

    async def _prune_if_needed(self) -> None:
        while self._stored_frames > MAX_FRAMES:
            episode = self._games.pop(0)
            self._stored_frames -= episode.size

            if not self._has_logged_pruning:
                msg = "Memory is full, pruning initiated"
                self.log.info(msg)
                await self.event.publish(
                    EventMsg(level=DHydraLog.INFO, message=msg)
                )
                self._has_logged_pruning = True

    def _rebuild_bucket_index(self) -> None:
        """
        Rebuild current-gear bucket eligibility index.
        """
        self._episodes_by_bucket = {idx: [] for idx in range(MAX_BUCKETS)}

        for ep in self._games:
            meta = ep.gear_meta[self._cur_gear]
            usable_buckets = min(meta.num_chunks, MAX_BUCKETS)

            for bucket_idx in range(usable_buckets):
                self._episodes_by_bucket[bucket_idx].append(ep)

        self._warmed_buckets = [
            idx
            for idx, episodes in self._episodes_by_bucket.items()
            if episodes
        ]

    def sample_chunks(self) -> list[list[Transition]] | None:
        """
        Sample chunks across warmed buckets for the current gear.
        """
        if not self._warmed_buckets:
            return None

        total_chunks = 0
        for bucket_idx in self._warmed_buckets:
            total_chunks += len(self._episodes_by_bucket[bucket_idx])

        if total_chunks < self._batch_size:
            return None

        samples: list[list[Transition]] = []

        base = self._batch_size // len(self._warmed_buckets)
        remainder = self._batch_size % len(self._warmed_buckets)

        requested: dict[int, int] = {idx: base for idx in self._warmed_buckets}

        warmed_desc = sorted(self._warmed_buckets, reverse=True)
        for i in range(remainder):
            requested[warmed_desc[i % len(warmed_desc)]] += 1

        for bucket_idx in self._warmed_buckets:
            eligible = self._episodes_by_bucket[bucket_idx]
            if not eligible:
                continue

            meta_samples = requested[bucket_idx]

            for _ in range(meta_samples):
                ep = self._rng.choice(eligible)
                meta = ep.gear_meta[self._cur_gear]
                start, end = meta.ranges[bucket_idx]
                samples.append(list(ep.frames[start:end]))

        return samples
