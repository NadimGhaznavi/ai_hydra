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
from ai_hydra.nnet.ATH.ATHCommon import get_gear_data


class ATHGearBox:

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
    ):
        self.store = data_store

        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_GEARBOX,
            log_level=log_level,
            to_console=True,
        )
        # Remote logging: ZeroMQ "Events" PUB/SUB topic
        self.event = HydraEventMQ(
            client_id=DModule.ATH_GEARBOX, pub_func=pub_func
        )

        self._cur_gear = 1
        self._max_gear = max_gear
        self._max_training_frames = max_training_frames
        self._cur_seq_length, self._cur_batch_size = get_gear_data(
            gear=self._cur_gear,
            max_gear=max_gear,
            max_training_frames=max_training_frames,
        )

        self._cur_bucket_counts: int | None = None

        self._cooldown_count = 0

        # Handling stagnation by downshifting
        self._stagnation_flag = False
        self._stagnation_alert_count = 0
        # Track critical stagnation events
        self._crit_stagnation_alert_count = 0

        # Track when the last upshift happened
        self._cur_epoch = 0
        self._last_upshift = 0

        # User defined settings
        self._max_buckets = max_buckets
        self._upshift_count_thresh = upshift_count_thresh
        self._downshift_count_thresh = downshift_count_thresh
        self._num_cooldown_eps = num_cooldown_eps

    async def get_gear(self) -> int:
        self._cur_bucket_counts = bucket_counts = (
            self.store.get_bucket_counts()
        )

        if not bucket_counts:
            return self._cur_gear

        if len(bucket_counts) != self._max_buckets:
            return self._cur_gear

        chunk_count = self.get_thresh_bucket_chunk_count()

        # We're dealing with a stagnation event
        if (
            self._stagnation_flag
            and self._cooldown_count > self._num_cooldown_eps
        ):

            # Downshift
            self._cur_gear -= self._stagnation_alert_count

            # Make sure we don't downshift past 1
            if self._cur_gear < 1:
                self._cur_gear = 1
                return self._cur_gear

            self._cur_seq_length, self._cur_batch_size = get_gear_data(
                gear=self._cur_gear,
                max_gear=self._max_gear,
                max_training_frames=self._max_training_frames,
            )
            # Stay here till the cooldown period is over
            self._cooldown_count = 0
            # Clear the stagnation flag
            self._stagnation_flag = False

            # Log the event
            await self.event.publish(
                EventMsg(
                    level=EV_STATUS.WARN,
                    message=(
                        f"Stagnation alert({self._stagnation_alert_count}) - "
                        f"Shifting DOWN: {self._cur_gear + self._stagnation_alert_count}"
                        f" > {self._cur_gear} - {self._cur_seq_length}/{self._cur_batch_size}"
                    ),
                    ev_type=EV_TYPE.SHIFTING,
                    payload={
                        DField.GEAR: self._cur_gear,
                        DField.SEQ_LENGTH: self._cur_seq_length,
                        DField.BATCH_SIZE: self._cur_batch_size,
                    },
                )
            )

        # Time to shift UP
        elif (
            chunk_count > self._upshift_count_thresh
            and self._cooldown_count > self._num_cooldown_eps
        ):
            # Upshifting resets the stagnation alert count. This allows the
            # system to, up, up, stagnation-alert: down, up, up, ... to
            # work it's way back up to the "top gear" again.
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
                    message=f"Shifting UP: {self._cur_gear - 1} > {self._cur_gear} - {self._cur_seq_length}/{self._cur_batch_size}",
                    ev_type=EV_TYPE.SHIFTING,
                    payload={
                        DField.GEAR: self._cur_gear,
                        DField.SEQ_LENGTH: self._cur_seq_length,
                        DField.BATCH_SIZE: self._cur_batch_size,
                    },
                )
            )

        # Time to shift DOWN
        elif (
            chunk_count < self._downshift_count_thresh
            and self._cooldown_count > self._num_cooldown_eps
        ):
            # First gear is the lowest gear
            if self._cur_gear == 1:
                return self._cur_gear

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
                    message=f"Shifting DOWN: {self._cur_gear} - {self._cur_seq_length}/{self._cur_batch_size}",
                    ev_type=EV_TYPE.SHIFTING,
                    payload={
                        DField.GEAR: self._cur_gear,
                        DField.SEQ_LENGTH: self._cur_seq_length,
                        DField.BATCH_SIZE: self._cur_batch_size,
                    },
                )
            )

        return self._cur_gear

    def get_thresh_bucket_chunk_count(self) -> int:
        count = 0
        n = len(self._cur_bucket_counts)
        for bucket_idx in range(max(0, n - 3), n):
            count += self._cur_bucket_counts[bucket_idx]
        return count

    async def hard_reset(self, crit_count: int):
        self._stagnation_flag = False
        self._cooldown_count = 0
        old_gear = self._cur_gear
        self._cur_gear = 3  # seq_length/batch_size == 8/32
        self._cur_seq_length, self._cur_batch_size = get_gear_data(
            gear=self._cur_gear,
            max_gear=self._max_gear,
            max_training_frames=self._max_training_frames,
        )
        msg = (
            f"Critical Stagnation alert({crit_count}): Radical DOWN shift: "
            f"{old_gear} > {self._cur_gear} - {self._cur_seq_length}/"
            f"{self._cur_batch_size}"
        )
        await self.event.publish(
            EventMsg(
                level=EV_STATUS.BAD,
                message=msg,
                ev_type=EV_TYPE.SHIFTING,
                payload={
                    DField.GEAR: self._cur_gear,
                    DField.SEQ_LENGTH: self._cur_seq_length,
                    DField.BATCH_SIZE: self._cur_batch_size,
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
            self.log.debug(
                "Received stagnation warning: Setting stagnation flag"
            )
            self._stagnation_flag = True
            self._stagnation_alert_count += 1
            return

        self.log.debug(
            "Received stagnation warning: Ignoring because of recent upshift"
        )
