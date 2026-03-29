# ai_hydra/nnet/ATH/ATHDataMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from random import Random

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
        self._max_frames = max_frames
        self._max_gear = max_gear
        self._max_training_frames = max_training_frames

        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_DATA_MGR,
            log_level=log_level,
            to_console=True,
        )

        # Remote logging: ZeroMQ "Events" PUB/SUB topic
        self.event = HydraEventMQ(
            client_id=DModule.ATH_DATA_MGR,
            pub_func=pub_func,
        )

        self._cur_gear: int | None = None
        self._samples_served = 0
        self._has_logged_startup = False
        self._has_logged_pruning = False

    async def append(
        self,
        t: Transition,
        final_score: int | None = None,
    ) -> None:
        if not self._has_logged_startup:
            await self._log_startup()
            self._has_logged_startup = True

        self.store.append(t)

        if not t.done:
            return

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
        Build per-gear chunk metadata for one finalized episode.
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

        Chunks are aligned from the end to preserve the newest / terminal
        part of the episode, matching the original ATH behavior.
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

        self.store.finalize_game(gear_meta=gear_meta)

        await self._prune_if_needed()
        self._rebuild_bucket_index()

    async def _log_startup(self) -> None:
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

    def _rebuild_bucket_index(self) -> None:
        self.store.reset_bucket_indexes()
        self.store.update_episodes_by_bucket()
        self.store.update_warmed_buckets_list()

    def _get_eligible_eps_for_bucket(self, bucket_idx: int):
        """
        Return only episodes that actually have a valid chunk for this
        bucket index at the current gear.
        """
        episodes = self.store.get_eps_by_bucket_idx(bucket_idx=bucket_idx)
        eligible = []

        for ep in episodes:
            meta = ep.gear_meta[self._cur_gear]
            if bucket_idx < meta.num_chunks:
                eligible.append(ep)

        return eligible

    async def sample_chunks(self) -> list[list[Transition]] | None:
        """
        Sample chunks across warmed buckets for the current gear.

        Important invariant:
        bucket_idx is only safe to use as a chunk index if the episode
        actually has that chunk at the current gear.
        """
        if self._cur_gear is None:
            return None

        warmed_buckets = self.store.get_warmed_buckets()
        if not warmed_buckets:
            return None

        _, batch_size = get_gear_data(
            gear=self._cur_gear,
            max_gear=self._max_gear,
            max_training_frames=self._max_training_frames,
        )

        eligible_by_bucket: dict[int, list] = {}
        total_chunks = 0

        for bucket_idx in warmed_buckets:
            eligible = self._get_eligible_eps_for_bucket(bucket_idx)
            if not eligible:
                continue
            eligible_by_bucket[bucket_idx] = eligible
            total_chunks += len(eligible)

        if not eligible_by_bucket:
            return None

        if total_chunks < batch_size:
            return None

        samples: list[list[Transition]] = []

        active_buckets = sorted(eligible_by_bucket.keys())
        base = batch_size // len(active_buckets)
        remainder = batch_size % len(active_buckets)

        requested: dict[int, int] = {idx: base for idx in active_buckets}

        # Bias remainder toward deeper buckets first
        active_desc = sorted(active_buckets, reverse=True)
        for i in range(remainder):
            requested[active_desc[i % len(active_desc)]] += 1

        for bucket_idx in active_buckets:
            eligible = eligible_by_bucket[bucket_idx]
            num_samples = requested[bucket_idx]

            for _ in range(num_samples):
                ep = self._rng.choice(eligible)
                meta = ep.gear_meta[self._cur_gear]

                # Defensive check: this should always hold because of the
                # eligibility filter above.
                if bucket_idx >= meta.num_chunks:
                    continue

                start, end = meta.ranges[bucket_idx]
                chunk = list(ep.frames[start:end])

                if chunk:
                    samples.append(chunk)

        if len(samples) < batch_size:
            return None

        self._samples_served += 1

        if self._samples_served % 10 == 0:
            bucket_counts = self.store.get_bucket_counts()

            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.INFO,
                    ev_type=EV_TYPE.BUCKETS_STATUS,
                    payload={
                        EV_TYPE.BUCKET_COUNTS: bucket_counts,
                        EV_TYPE.CUR_GEAR: self._cur_gear,
                    },
                )
            )

        return samples

    def set_gear(self, gear: int) -> None:
        if self._cur_gear == gear:
            return

        self._cur_gear = gear
        self.store.set_gear(gear)
        self._rebuild_bucket_index()
