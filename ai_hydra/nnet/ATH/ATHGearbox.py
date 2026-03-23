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
from ai_hydra.constants.DEvent import EV_TYPE

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ, EventMsg
from ai_hydra.nnet.ATH.ATHDataStore import ATHDataStore

MAX_BUCKETS = DMemDef.MAX_BUCKETS
THRESHOLD_BUCKETS = DMemDef.THRESHOLD_BUCKETS
UPSHIFT_COUNT_THRESHOLD = DMemDef.UPSHIFT_COUNT_THRESHOLD
DOWNSHIFT_COUNT_THRESHOLD = DMemDef.DOWNSHIFT_COUNT_THRESHOLD
NUM_COOLDOWN_EPISODES = DMemDef.NUM_COOLDOWN_EPISODES
MAX_TRAINING_FRAMES = DMemDef.MAX_TRAINING_FRAMES


class ATHGearBox:

    def __init__(
        self, log_level: DHydraLog, data_store: ATHDataStore, pub_func
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
        self._cur_seq_length, self._cur_batch_size = self.get_gear_data(
            self._cur_gear
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

    async def get_gear(self) -> int:
        self._cur_bucket_counts = bucket_counts = (
            self.store.get_bucket_counts()
        )

        if not bucket_counts:
            return self._cur_gear

        if len(bucket_counts) != MAX_BUCKETS:
            return self._cur_gear

        chunk_count = self.get_thresh_bucket_chunk_count()

        # We're dealing with a stagnation event
        if (
            self._stagnation_flag
            and self._cooldown_count > NUM_COOLDOWN_EPISODES
        ):

            # Downshift
            self._cur_gear -= self._stagnation_alert_count

            # Make sure we don't downshift past 1
            if self._cur_gear < 1:
                self._cur_gear = 1
                return self._cur_gear

            self._cur_seq_length, self._cur_batch_size = self.get_gear_data(
                self._cur_gear
            )
            # Stay here till the cooldown period is over
            self._cooldown_count = 0
            # Clear the stagnation flag
            self._stagnation_flag = False

            # Log the event
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.WARNING,
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
            chunk_count > UPSHIFT_COUNT_THRESHOLD
            and self._cooldown_count > NUM_COOLDOWN_EPISODES
        ):
            # Upshifting resets the stagnation alert count. This allows the
            # system to, up, up, stagnation-alert: down, up, up, ... to
            # work it's way back up to the "top gear" again.
            self._stagnation_alert_count = 0

            self._cur_gear += 1
            self._last_upshift = self._cur_epoch
            self._cooldown_count = 0
            self._cur_seq_length, self._cur_batch_size = self.get_gear_data(
                self._cur_gear
            )
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.INFO,
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
            chunk_count < DOWNSHIFT_COUNT_THRESHOLD
            and self._cooldown_count > NUM_COOLDOWN_EPISODES
        ):
            # First gear is the lowest gear
            if self._cur_gear == 1:
                return self._cur_gear

            self._cur_gear -= 1
            self._cooldown_count = 0
            self._cur_seq_length, self._cur_batch_size = self.get_gear_data(
                self._cur_gear
            )
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.INFO,
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

    def get_gear_data(self, gear: int) -> tuple[int, int]:
        """
        - Accepts a gear.
        - Returns (sequence_length, batch_size)

        The sequence_length/batch_size should not exceed MAX_TRAINING_FRAMES.
        First gear (1), should have a sequence_length of 4.
        Every gear after one should increase the seq_length by 2.
        """

        if gear < 1:
            raise ValueError("gear must be >= 1")

        seq_len = (gear * 2) + 2
        batch_size = max(1, MAX_TRAINING_FRAMES // seq_len)
        return seq_len, batch_size

    def get_thresh_bucket_chunk_count(self) -> int:
        count = 0
        for bucket_idx in THRESHOLD_BUCKETS:
            count += self._cur_bucket_counts[bucket_idx]
        return count

    async def hard_reset(self, crit_count: int):
        self._stagnation_flag = False
        self._cooldown_count = 0
        old_gear = self._cur_gear
        self._cur_gear = 3  # seq_length/batch_size == 8/32
        self._cur_seq_length, self._cur_batch_size = self.get_gear_data(
            self._cur_gear
        )
        await self.event.publish(
            EventMsg(
                level=DHydraLog.INFO,
                message=(
                    f"Critical Stagnation "
                    f"alert({crit_count}): Radical "
                    f"DOWN shift: {old_gear} > {self._cur_gear} - "
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
        return

    def incr_cooldown_count(self):
        self._cooldown_count += 1

    def incr_epoch(self):
        self._cur_epoch += 1

    async def stagnation_cleared(self):
        if self._stagnation_alert_count != 0:
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.INFO,
                    message=f"Resetting stagnation count to 0",
                )
            )
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
