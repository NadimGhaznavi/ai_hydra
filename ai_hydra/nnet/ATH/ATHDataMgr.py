# ai_hydra/nnet/ATH/ATHDataMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from random import Random

from ai_hydra.constants.DReplayMemory import DMemDef, DMemField
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DEvent import EV_TYPE, EV_STATUS

from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ, EventMsg
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.ATH.ATHDataStore import ATHDataStore
from ai_hydra.nnet.ATH.ATHCommon import (
    GearMeta,
    get_gear_data,
    get_valid_gears,
)


class ATHDataMgr:

    def __init__(
        self,
        log_level: DHydraLog,
        rng: Random,
        pub_func,
        data_store: ATHDataStore,
        max_frames: int,
        max_gear: int,
        max_training_frames: int,
    ):
        self._rng = rng
        self.store = data_store
        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_DATA_MGR,
            log_level=log_level,
            to_console=True,
        )
        # Remote logging: ZeroMQ "Events" PUB/SUB topic
        self.event = HydraEventMQ(
            client_id=DModule.ATH_DATA_MGR, pub_func=pub_func
        )

        self._cur_gear = None
        self._samples_served = 0
        self._has_logged_startup = False
        self._has_logged_pruning = False

        self._max_frames = max_frames
        self._max_gear = max_gear
        self._max_training_frames = max_training_frames

    async def append(
        self, t: Transition, final_score: int | None = None
    ) -> None:

        if not self._has_logged_startup:
            await self._log_startup()
            self._has_logged_startup = True

        self.store.append(t)

        if t.done:
            if final_score is None:
                raise ValueError(
                    "final_score must be provided when appending a terminal transition"
                )

            await self._finalize_game()

    def _build_episode_gear_meta(
        self,
        ep_size: int,
    ) -> dict[int, GearMeta]:
        """
        Build metadata for all gears for one episode.
        """
        gear_meta: dict[int, GearMeta] = {}

        for gear in get_valid_gears(max_gear=self._max_gear):
            seq_length, _ = get_gear_data(
                gear=gear,
                max_gear=self._max_gear,
                max_training_frames=self._max_training_frames,
            )
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

    async def _finalize_game(self) -> None:
        ep_size = self.store.get_cur_ep_size()
        gear_meta = self._build_episode_gear_meta(ep_size)
        self.store.finalize_game(gear_meta)

        await self._prune_if_needed()
        self._rebuild_bucket_index()

    async def _log_startup(self):
        self.log.info(f"Initialized. Setting max_frames to {self._max_frames}")

    async def _prune_if_needed(self) -> None:

        while self.store.get_stored_frame_count() > self._max_frames:
            self.store.pop_first_game()

            if not self._has_logged_pruning:
                msg = f"Memory is full ({self._max_frames}), pruning initiated"
                self.log.info(msg)

                await self.event.publish(
                    EventMsg(level=EV_STATUS.INFO, message=msg)
                )
                self._has_logged_pruning = True

    def _rebuild_bucket_index(self):
        self.store.reset_bucket_indexes()
        self.store.update_episodes_by_bucket()
        self.store.update_warmed_buckets_list()

    async def sample_chunks(self) -> list[list[Transition]] | None:
        """
        Sample chunks across warmed buckets for the current gear.
        """
        warmed_buckets = self.store.get_warmed_buckets()

        if not warmed_buckets:
            return None

        total_chunks = 0
        for bucket_idx in warmed_buckets:
            episodes = self.store.get_eps_by_bucket_idx(bucket_idx=bucket_idx)
            total_chunks += len(episodes)

        _, batch_size = get_gear_data(
            gear=self._cur_gear,
            max_gear=self._max_gear,
            max_training_frames=self._max_training_frames,
        )

        if total_chunks < batch_size:
            return None

        samples: list[list[Transition]] = []

        base = batch_size // len(warmed_buckets)
        remainder = batch_size % len(warmed_buckets)

        requested: dict[int, int] = {idx: base for idx in warmed_buckets}

        warmed_desc = sorted(warmed_buckets, reverse=True)
        for i in range(remainder):
            requested[warmed_desc[i % len(warmed_desc)]] += 1

        for bucket_idx in warmed_buckets:
            eligible = self.store.get_eps_by_bucket_idx(bucket_idx=bucket_idx)
            if not eligible:
                continue

            meta_samples = requested[bucket_idx]

            for _ in range(meta_samples):
                ep = self._rng.choice(eligible)
                meta = ep.gear_meta[self._cur_gear]
                start, end = meta.ranges[bucket_idx]
                samples.append(list(ep.frames[start:end]))

        self._samples_served += 1

        if self._samples_served % 10 == 0:
            bucket_counts = self.store.get_bucket_counts()
            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.INFO,
                    ev_type=EV_TYPE.BUCKETS_STATUS,
                    payload={
                        EV_TYPE.BUCKET_COUNTS: bucket_counts,
                    },
                )
            )
        return samples

    def set_gear(self, gear: int):
        if self._cur_gear == gear:
            return

        self._cur_gear = gear
        self.store.set_gear(gear)
        self._rebuild_bucket_index()
