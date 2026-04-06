# ai_hydra/nnet/ATH/ATHGearBox.py
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

from ai_hydra.constants.DReplayMemory import DMemDef
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DEvent import EV_TYPE, EV_STATUS

from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ, EventMsg
from ai_hydra.nnet.ATH.ATHDataStore import ATHDataStore
from ai_hydra.nnet.ATH.ATHCommon import Episode, GearMeta, get_gear_data


class ATHGearBox:
    """
    Adaptive gear controller for ATH replay memory.

    Responsibilities:
    - inspect derived bucket-count state from ATHDataStore
    - decide whether to upshift or downshift
    - track cooldown and stagnation state
    - expose the active gear and its derived training shape

    Notes:
    - gears are 1-based
    - gear decisions are based on derived replay-memory signals
    - canonical replay storage lives in ATHDataStore, not here
    """

    def __init__(
        self,
        log_level: DHydraLog,
        data_store: ATHDataStore,
        pub_func,
        max_buckets: int,
        upshift_count_thresh: int,
        downshift_count_thresh: int,
        num_cooldown_eps: int,
        max_gear: int,
        max_training_frames: int,
        is_mcts: bool,
    ):
        if not isinstance(data_store, ATHDataStore):
            raise TypeError(
                f"data_store must be ATHDataStore, got {type(data_store).__name__}"
            )

        if not isinstance(max_buckets, int):
            raise TypeError(
                f"max_buckets must be int, got {type(max_buckets).__name__}"
            )
        if not isinstance(upshift_count_thresh, int):
            raise TypeError(
                "upshift_count_thresh must be int, got "
                f"{type(upshift_count_thresh).__name__}"
            )
        if not isinstance(downshift_count_thresh, int):
            raise TypeError(
                "downshift_count_thresh must be int, got "
                f"{type(downshift_count_thresh).__name__}"
            )
        if not isinstance(num_cooldown_eps, int):
            raise TypeError(
                "num_cooldown_eps must be int, got "
                f"{type(num_cooldown_eps).__name__}"
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

        if max_buckets < 1:
            raise ValueError(f"max_buckets must be >= 1, got {max_buckets}")
        if upshift_count_thresh < 0:
            raise ValueError(
                f"upshift_count_thresh must be >= 0, got {upshift_count_thresh}"
            )
        if downshift_count_thresh < 0:
            raise ValueError(
                f"downshift_count_thresh must be >= 0, got {downshift_count_thresh}"
            )
        if num_cooldown_eps < 0:
            raise ValueError(
                f"num_cooldown_eps must be >= 0, got {num_cooldown_eps}"
            )
        if max_gear < 1:
            raise ValueError(f"max_gear must be >= 1, got {max_gear}")
        if max_training_frames < 4:
            raise ValueError(
                f"max_training_frames must be >= 4, got {max_training_frames}"
            )

        self.store = data_store

        # User-defined settings / static config
        self._max_buckets = max_buckets
        self._upshift_count_thresh = upshift_count_thresh
        self._downshift_count_thresh = downshift_count_thresh
        self._num_cooldown_eps = num_cooldown_eps
        self._max_gear = max_gear
        self._max_training_frames = max_training_frames
        self._is_mcts = is_mcts

        # MCTS Label
        if is_mcts:
            self._mcts_label = "Monte Carlo "
        else:
            self._mcts_label = ""

        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_GEARBOX,
            log_level=log_level,
            to_console=True,
        )

        # Remote logging: ZeroMQ "Events" PUB/SUB topic
        self.event = HydraEventMQ(
            client_id=DModule.ATH_GEARBOX,
            pub_func=pub_func,
        )

        # Active gear state
        self._cur_gear: int = 1
        self._cur_seq_length, self._cur_batch_size = get_gear_data(
            gear=self._cur_gear,
            max_gear=self._max_gear,
            max_training_frames=self._max_training_frames,
        )

        # Latest derived replay signal
        self._cur_bucket_counts: dict[int, int] | None = None

        # Cooldown / stagnation state
        self._cooldown_count: int = 0
        self._stagnation_flag: bool = False
        self._stagnation_alert_count: int = 0
        self._crit_stagnation_alert_count: int = 0

        # Epoch tracking
        self._cur_epoch: int = 0
        self._last_upshift: int = 0

        # Log initial settings
        self._logged_initial_settings = False

    async def get_gear(self) -> int:
        """
        Return the active gear, applying any required shift first.

        Decision order:
        1. stagnation-driven downshift
        2. deep-signal-driven upshift
        3. deep-signal-driven downshift
        4. otherwise remain in current gear

        Invariants:
            - _cur_gear is always a valid 1-based gear
            - _cur_bucket_counts must match configured bucket shape
            - _cur_seq_length / _cur_batch_size always match _cur_gear
        """
        bucket_counts = self.store.get_bucket_counts()

        if not isinstance(bucket_counts, dict):
            raise TypeError(
                "store.get_bucket_counts() must return dict, "
                f"got {type(bucket_counts).__name__}"
            )

        expected_keys = set(range(self._max_buckets))
        actual_keys = set(bucket_counts.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                "Bucket count state is corrupted: "
                f"expected keys {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        for bucket_idx, count in bucket_counts.items():
            if not isinstance(count, int):
                raise TypeError(
                    f"Bucket {bucket_idx} count must be int, got "
                    f"{type(count).__name__}"
                )
            if count < 0:
                raise RuntimeError(
                    f"Bucket {bucket_idx} count cannot be negative, got {count}"
                )

        self._cur_bucket_counts = bucket_counts

        deep_bucket_signal = self.get_deep_bucket_chunk_count()
        cooldown_ready = self._cooldown_count > self._num_cooldown_eps

        if not self._logged_initial_settings:
            self._logged_initial_settings = True
            self._cur_seq_length, self._cur_batch_size = get_gear_data(
                gear=self._cur_gear,
                max_gear=self._max_gear,
                max_training_frames=self._max_training_frames,
            )
            if self._is_mcts:
                mcts_flag = True
            else:
                mcts_flag = False

            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.WARN,
                    ev_type=EV_TYPE.SHIFTING,
                    payload={
                        DField.GEAR: self._cur_gear,
                        DField.SEQ_LENGTH: self._cur_seq_length,
                        DField.BATCH_SIZE: self._cur_batch_size,
                        DField.MCTS_MEMORY: mcts_flag,
                    },
                )
            )

        # ----------------------------
        # 1. Stagnation-driven downshift
        # ----------------------------
        if self._stagnation_flag and cooldown_ready:
            old_gear = self._cur_gear
            downshift_by = max(1, self._stagnation_alert_count)
            self._cur_gear = max(1, self._cur_gear - downshift_by)

            self._cur_seq_length, self._cur_batch_size = get_gear_data(
                gear=self._cur_gear,
                max_gear=self._max_gear,
                max_training_frames=self._max_training_frames,
            )

            self._cooldown_count = 0
            self._stagnation_flag = False

            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.WARN,
                    message=(
                        f"Stagnation alert({self._stagnation_alert_count}) - "
                        f"Shifting DOWN: {old_gear} > {self._cur_gear} - "
                        f"{self._cur_seq_length}/{self._cur_batch_size}"
                    ),
                    ev_type=EV_TYPE.SHIFTING,
                    payload={
                        DField.GEAR: self._cur_gear,
                        DField.SEQ_LENGTH: self._cur_seq_length,
                        DField.BATCH_SIZE: self._cur_batch_size,
                    },
                )
            )
            return self._cur_gear

        # ----------------------------
        # 2. Deep-signal-driven upshift
        # ----------------------------
        if deep_bucket_signal > self._upshift_count_thresh and cooldown_ready:
            if self._cur_gear < self._max_gear:
                old_gear = self._cur_gear
                self._stagnation_alert_count = 0
                self._cur_gear += 1
                self._last_upshift = self._cur_epoch
                self._cooldown_count = 0

                self._cur_seq_length, self._cur_batch_size = get_gear_data(
                    gear=self._cur_gear,
                    max_gear=self._max_gear,
                    max_training_frames=self._max_training_frames,
                )

                await self.event.publish(
                    EventMsg(
                        level=EV_STATUS.GOOD,
                        message=(
                            f"Shifting UP: {old_gear} > {self._cur_gear} - "
                            f"{self._cur_seq_length}/{self._cur_batch_size}"
                        ),
                        ev_type=EV_TYPE.SHIFTING,
                        payload={
                            DField.GEAR: self._cur_gear,
                            DField.SEQ_LENGTH: self._cur_seq_length,
                            DField.BATCH_SIZE: self._cur_batch_size,
                        },
                    )
                )

            return self._cur_gear

        # ----------------------------
        # 3. Deep-signal-driven downshift
        # ----------------------------
        if (
            deep_bucket_signal < self._downshift_count_thresh
            and cooldown_ready
        ):
            if self._cur_gear > 1:
                old_gear = self._cur_gear
                self._cur_gear -= 1
                self._cooldown_count = 0

                self._cur_seq_length, self._cur_batch_size = get_gear_data(
                    gear=self._cur_gear,
                    max_gear=self._max_gear,
                    max_training_frames=self._max_training_frames,
                )

                await self.event.publish(
                    EventMsg(
                        level=EV_STATUS.WARN,
                        message=(
                            f"Shifting DOWN: {old_gear} > {self._cur_gear} - "
                            f"{self._cur_seq_length}/{self._cur_batch_size})"
                        ),
                        ev_type=EV_TYPE.SHIFTING,
                        payload={
                            DField.GEAR: self._cur_gear,
                            DField.SEQ_LENGTH: self._cur_seq_length,
                            DField.BATCH_SIZE: self._cur_batch_size,
                        },
                    )
                )

        return self._cur_gear

    def get_deep_bucket_chunk_count(self) -> int:
        """
        Return the number of chunk references in the deepest 3 buckets.

        Under ATH semantics, buckets represent chunk/sequence-reference depth.
        Longer episodes populate more buckets. When sequence length increases,
        the deepest buckets thin out. This signal is therefore used as a proxy
        for whether the current replay memory can support the current gear, and
        whether it can sustain an upshift or requires a downshift.

        Invariants:
            - _cur_bucket_counts must already be populated
            - _cur_bucket_counts must have keys 0 .. max_buckets - 1
            - bucket counts must be non-negative ints
        """
        if self._cur_bucket_counts is None:
            raise RuntimeError("Bucket counts not initialized")

        if not isinstance(self._cur_bucket_counts, dict):
            raise TypeError(
                f"_cur_bucket_counts must be dict, got "
                f"{type(self._cur_bucket_counts).__name__}"
            )

        expected_keys = set(range(self._max_buckets))
        actual_keys = set(self._cur_bucket_counts.keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                "Bucket count state is corrupted: "
                f"expected keys {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        signal = 0
        start_idx = max(0, self._max_buckets - 3)

        for bucket_idx in range(start_idx, self._max_buckets):
            count = self._cur_bucket_counts[bucket_idx]

            if not isinstance(count, int):
                raise TypeError(
                    f"Bucket {bucket_idx} count must be int, got "
                    f"{type(count).__name__}"
                )
            if count < 0:
                raise RuntimeError(
                    f"Bucket {bucket_idx} count cannot be negative, got {count}"
                )

            signal += count

        return signal

    async def hard_reset(self, crit_count: int):
        self._stagnation_flag = False
        self._cooldown_count = 0
        old_gear = self._cur_gear
        self._cur_gear = 1
        self._cur_seq_length, self._cur_batch_size = get_gear_data(
            gear=self._cur_gear,
            max_gear=self._max_gear,
            max_training_frames=self._max_training_frames,
        )
        msg = (
            f"{self._mcts_label}Critical Stagnation alert({crit_count}): "
            f"Radical DOWN shift: "
            f"{old_gear} > {self._cur_gear} - "
            f"{self._cur_seq_length}/{self._cur_batch_size}"
        )
        if self._is_mcts:
            mcts_flag = True
        else:
            mcts_flag = False

        await self.event.publish(
            EventMsg(
                level=EV_STATUS.BAD,
                message=msg,
                ev_type=EV_TYPE.SHIFTING,
                payload={
                    DField.GEAR: self._cur_gear,
                    DField.SEQ_LENGTH: self._cur_seq_length,
                    DField.BATCH_SIZE: self._cur_batch_size,
                    DField.MCTS_MEMORY: mcts_flag,
                },
            )
        )
        self.log.debug(msg)
        return

    def incr_cooldown_count(self):
        self._cooldown_count += 1

    def incr_epoch(self):
        self._cur_epoch += 1

    async def stagnation_cleared(self):
        if self._stagnation_alert_count != 0:
            msg = "Resetting stagnation count to 0"
            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.INFO,
                    message=msg,
                )
            )
            self.log.debug(msg)
        self._stagnation_flag = False
        self._stagnation_alert_count = 0

    def stagnation_warning(self):
        # Don't downshift if we just upshifted....
        num_ep_since_upshift = self._cur_epoch - self._last_upshift
        if num_ep_since_upshift > DMemDef.MAX_STAGNANT_EPISODES:
            self.log.info(
                f"{self._mcts_label}Received stagnation warning: Setting stagnation flag"
            )
            self._stagnation_flag = True
            self._stagnation_alert_count += 1
            return

        self.log.info(
            f"{self._mcts_label}Received stagnation warning: Ignoring because "
            "of recent upshift"
        )
