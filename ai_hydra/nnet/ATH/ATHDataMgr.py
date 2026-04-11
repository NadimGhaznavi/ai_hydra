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
    Episode,
    get_gear_data,
    get_valid_gears,
    assert_valid_gear,
)

# Number of episodes before anything is returned from sample_chunks()
MIN_EPISODES = 1


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
        """
        Initialize ATHDataMgr.

        Responsibilities:
        - manage episode finalization
        - prune canonical replay storage
        - rebuild derived bucket indexes
        - sample fixed-length replay chunks for the active gear

        Parameters:
            log_level (DHydraLog):
                Logging verbosity for local logger output.
            rng (Random):
                Random number generator used for replay sampling.
            pub_func:
                Publish function used by HydraEventMQ for emitting events.
            data_store (ATHDataStore):
                Canonical ATH replay storage and derived bucket index holder.
            max_frames (int):
                Maximum number of stored frames allowed before pruning.
            max_gear (int):
                Highest valid ATH gear.
            max_training_frames (int):
                Upper bound used to derive (seq_length, batch_size) per gear.

        Invariants:
            - data_store must be a valid ATHDataStore instance
            - max_frames must be >= 1
            - max_gear must be >= 1
            - max_training_frames must be >= 4
            - _cur_gear is unset until explicitly provided by set_gear()
        """
        if not isinstance(rng, Random):
            raise TypeError(f"rng must be Random, got {type(rng).__name__}")

        if not isinstance(data_store, ATHDataStore):
            raise TypeError(
                f"data_store must be ATHDataStore, got {type(data_store).__name__}"
            )

        if not isinstance(max_frames, int):
            raise TypeError(
                f"max_frames must be int, got {type(max_frames).__name__}"
            )
        if not isinstance(max_gear, int):
            raise TypeError(
                f"max_gear must be int, got {type(max_gear).__name__}"
            )
        if not isinstance(max_training_frames, int):
            raise TypeError(
                "max_training_frames must be int, got "
                f"{type(max_training_frames).__name__}"
            )

        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        if max_gear < 1:
            raise ValueError(f"max_gear must be >= 1, got {max_gear}")
        if max_training_frames < 4:
            raise ValueError(
                f"max_training_frames must be >= 4, got {max_training_frames}"
            )

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

        # Runtime state
        self._cur_gear: int | None = None
        self._samples_served: int = 0
        self._has_logged_startup: bool = False
        self._has_logged_pruning: bool = False
        self._num_episodes = 0

    async def append(
        self,
        t: Transition,
        final_score: int | None = None,
    ) -> None:
        """
        Append a transition and finalize the episode if terminal.

        This method:
        - logs startup once on first use
        - appends the transition to the in-progress episode
        - finalizes the episode if the transition is terminal

        Parameters:
            t (Transition):
                Transition to append.
            final_score (int | None):
                Final episode score. Required if t.done is True.

        Invariants:
            - store.append() is always called exactly once per transition
            - finalize_game() is only triggered on terminal transitions
        """

        if not isinstance(t, Transition):
            raise TypeError(f"t must be Transition, got {type(t).__name__}")

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

        if not isinstance(final_score, int):
            raise TypeError(
                f"final_score must be int, got {type(final_score).__name__}"
            )

        if final_score > 0:
            await self._finalize_game()

        else:
            self.store.discard_cur_game()

    def _build_episode_gear_meta(
        self,
        ep_size: int,
    ) -> dict[int, GearMeta]:
        """
        Build per-gear chunk metadata for one finalized episode.

        Parameters:
            ep_size (int):
                Number of transitions in the finalized episode.

        Returns:
            dict[int, GearMeta]:
                Mapping of 1-based gear -> GearMeta for that episode size.

        Invariants:
            - ep_size must be >= 0
            - metadata is generated for every valid gear in [1, max_gear]
            - each GearMeta is internally validated by ATHCommon
        """
        if not isinstance(ep_size, int):
            raise TypeError(
                f"ep_size must be int, got {type(ep_size).__name__}"
            )
        if ep_size < 0:
            raise ValueError(f"ep_size must be >= 0, got {ep_size}")

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

        expected_gears = set(get_valid_gears(max_gear=self._max_gear))
        actual_gears = set(gear_meta.keys())
        if actual_gears != expected_gears:
            raise RuntimeError(
                "Gear metadata build failed: "
                f"expected gears {sorted(expected_gears)}, "
                f"got {sorted(actual_gears)}"
            )

        return gear_meta

    def _build_gear_meta_for_size(
        self,
        ep_size: int,
        seq_length: int,
    ) -> GearMeta:
        """
        Build gear-specific chunk metadata for an episode size.

        Chunks are fixed-length and aligned from the end to preserve the
        most recent (terminal) portion of the episode.

        Example:
            ep_size = 25, seq_length = 6

            num_chunks = 4
            remainder  = 1

            ranges:
                (1, 7), (7, 13), (13, 19), (19, 25)

        Invariants:
            - ep_size >= 0
            - seq_length >= 1
            - num_chunks = ep_size // seq_length
            - each range has length == seq_length
            - ranges are non-overlapping and ordered
        """

        if not isinstance(ep_size, int):
            raise TypeError(
                f"ep_size must be int, got {type(ep_size).__name__}"
            )
        if not isinstance(seq_length, int):
            raise TypeError(
                f"seq_length must be int, got {type(seq_length).__name__}"
            )

        if ep_size < 0:
            raise ValueError(f"ep_size must be >= 0, got {ep_size}")
        if seq_length < 1:
            raise ValueError(f"seq_length must be >= 1, got {seq_length}")

        num_chunks = ep_size // seq_length
        rem = ep_size % seq_length

        ranges: list[tuple[int, int]] = []

        for chunk_idx in range(num_chunks):
            start = rem + (chunk_idx * seq_length)
            end = start + seq_length

            # Defensive checks (cheap, worth it)
            if end > ep_size:
                raise RuntimeError(
                    f"Chunk exceeds episode bounds: end={end}, ep_size={ep_size}"
                )
            if end - start != seq_length:
                raise RuntimeError(
                    f"Invalid chunk length: expected {seq_length}, got {end - start}"
                )

            ranges.append((start, end))

        # Optional: sanity check for coverage
        if num_chunks > 0:
            first_start = ranges[0][0]
            last_end = ranges[-1][1]

            if last_end != ep_size:
                raise RuntimeError(
                    f"Last chunk must end at ep_size ({ep_size}), got {last_end}"
                )

            # remainder check
            if first_start != rem:
                raise RuntimeError(
                    f"First chunk must start at remainder ({rem}), got {first_start}"
                )

        return GearMeta(
            seq_length=seq_length,
            num_chunks=num_chunks,
            ranges=tuple(ranges),
        )

    async def _finalize_game(self) -> None:
        """
        Finalize the current in-progress episode.

        This method:
        - reads the current episode size from canonical storage
        - builds per-gear metadata for that episode
        - commits the finalized episode to canonical storage
        - prunes old episodes if memory exceeds max_frames
        - rebuilds the derived bucket index

        Invariants:
            - store must contain a non-empty in-progress episode
            - finalized episode metadata must match the episode size
            - derived bucket state must be rebuilt after any storage mutation
        """
        self._num_episodes += 1

        ep_size = self.store.get_cur_ep_size()

        if not isinstance(ep_size, int):
            raise TypeError(
                f"store.get_cur_ep_size() must return int, got {type(ep_size).__name__}"
            )
        if ep_size <= 0:
            raise ValueError(
                f"Cannot finalize episode with invalid size: {ep_size}"
            )

        gear_meta = self._build_episode_gear_meta(ep_size=ep_size)

        if not gear_meta:
            raise RuntimeError("Gear metadata build returned empty result")

        self.store.finalize_game(gear_meta=gear_meta)

        await self._prune_if_needed()
        self._rebuild_bucket_index()

    async def _log_startup(self) -> None:
        self.log.info(f"Initialized. Setting max_frames to {self._max_frames}")

    async def _prune_if_needed(self) -> None:
        """
        Prune oldest episodes until stored frame count is within max_frames.

        This method:
        - removes finalized episodes from canonical storage in FIFO order
        - logs the start of pruning once
        - stops only when stored frame count <= max_frames

        Invariants:
            - store.get_stored_frame_count() must return a non-negative int
            - pruning must monotonically reduce stored frame count
            - canonical storage must not be popped when already empty
        """
        stored_frames = self.store.get_stored_frame_count()

        if not isinstance(stored_frames, int):
            raise TypeError(
                "store.get_stored_frame_count() must return int, "
                f"got {type(stored_frames).__name__}"
            )
        if stored_frames < 0:
            raise RuntimeError(
                f"Stored frame count cannot be negative, got {stored_frames}"
            )

        while stored_frames > self._max_frames:
            before = stored_frames

            self.store.pop_first_game()

            stored_frames = self.store.get_stored_frame_count()

            if stored_frames >= before:
                raise RuntimeError(
                    "Pruning failed to reduce stored frame count: "
                    f"before={before}, after={stored_frames}"
                )

            if not self._has_logged_pruning:
                msg = f"Memory is full ({self._max_frames}), pruning initiated"
                self.log.info(msg)
                await self.event.publish(
                    EventMsg(level=EV_STATUS.INFO, message=msg)
                )
                self._has_logged_pruning = True

    def _rebuild_bucket_index(self) -> None:
        """
        Rebuild all derived bucket-index state from canonical storage.

        This method:
        - resets the bucket index to an empty state
        - repopulates bucket membership from canonical episodes
        - recomputes warmed bucket indices

        Invariants:
            - canonical storage (_games in store) is the source of truth
            - current gear must already be set on the store
            - warmed buckets must be recomputed after bucket membership changes
        """
        if self._cur_gear is None:
            raise ValueError(
                "Cannot rebuild bucket index: current gear is not set"
            )

        self.store.reset_bucket_indexes()
        self.store.update_episodes_by_bucket()
        self.store.update_warmed_buckets_list()

        bucket_counts = self.store.get_bucket_counts()
        warmed_buckets = self.store.get_warmed_buckets()

        if not isinstance(bucket_counts, dict):
            raise TypeError(
                "store.get_bucket_counts() must return dict, "
                f"got {type(bucket_counts).__name__}"
            )
        if not isinstance(warmed_buckets, list):
            raise TypeError(
                "store.get_warmed_buckets() must return list, "
                f"got {type(warmed_buckets).__name__}"
            )

        for bucket_idx in warmed_buckets:
            count = bucket_counts.get(bucket_idx)
            if count is None:
                raise RuntimeError(
                    f"Warmed bucket {bucket_idx} missing from bucket counts"
                )
            if count <= 0:
                raise RuntimeError(
                    f"Warmed bucket {bucket_idx} has invalid count {count}"
                )

    def _get_eligible_eps_for_bucket(self, bucket_idx: int) -> list[Episode]:
        if self._cur_gear is None:
            raise ValueError("Current gear is not set")

        episodes = self.store.get_eps_by_bucket_idx(bucket_idx=bucket_idx)
        eligible: list[Episode] = []

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

        if self._num_episodes < MIN_EPISODES:
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
                    raise RuntimeError(
                        f"Invalid sample: bucket_idx={bucket_idx}, "
                        f"num_chunks={meta.num_chunks}, gear={self._cur_gear}"
                    )
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
        """
        Set the active gear for sampling and rebuild derived bucket state.

        Parameters:
            gear (int):
                Active ATH gear. Must be in the valid 1-based range.

        Invariants:
            - gear must be valid for this manager's configured max_gear
            - store gear and manager gear must remain synchronized
            - changing gear requires rebuilding derived bucket indexes
        """
        assert_valid_gear(gear=gear, max_gear=self._max_gear)

        if self._cur_gear == gear:
            return

        self._cur_gear = gear
        self.store.set_gear(gear)
        self._rebuild_bucket_index()

        warmed_buckets = self.store.get_warmed_buckets()
        bucket_counts = self.store.get_bucket_counts()

        for bucket_idx in warmed_buckets:
            count = bucket_counts.get(bucket_idx)
            if count is None or count <= 0:
                raise RuntimeError(
                    f"Invalid warmed bucket state after gear change: "
                    f"bucket={bucket_idx}, count={count}"
                )
