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

from ai_hydra.constants.DReplayMemory import DMemDef, DMemField, ATH_GEARBOX
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DHydraTui import DField
from ai_hydra.constants.DEvent import EV_TYPE

from ai_hydra.nnet.Transition import Transition
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import HydraEventMQ, EventMsg
from ai_hydra.nnet.ATH.ATHDataStore import ATHDataStore

MAX_BUCKETS = DMemDef.MAX_BUCKETS

THRESHOLD_BUCKETS = [17, 18, 19]

UPSHIFT_COUNT_THRESHOLD = 150
DOWNSHIFT_COUNT_THRESHOLD = 50

NUM_COOLDOWN_EPISODES = 200


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
        self._cur_seq_length, self._cur_batch_size = ATH_GEARBOX[
            self._cur_gear
        ]

        self._highest_gear = max(ATH_GEARBOX.keys())
        self._cur_bucket_counts: int | None = None

        self._cooldown_count = 0

    async def get_gear(self) -> int:
        self._cur_bucket_counts = bucket_counts = (
            self.store.get_bucket_counts()
        )

        if not bucket_counts:
            return self._cur_gear

        if len(bucket_counts) != MAX_BUCKETS:
            return self._cur_gear

        chunk_count = self.get_thresh_bucket_chunk_count()

        if (
            chunk_count > UPSHIFT_COUNT_THRESHOLD
            and self._cooldown_count > NUM_COOLDOWN_EPISODES
        ):
            # The transmission has a top gear
            if self._cur_gear == self._highest_gear:
                return self._cur_gear

            self._cur_gear += 1
            self._cooldown_count = 0
            self._cur_seq_length, self._cur_batch_size = ATH_GEARBOX[
                self._cur_gear
            ]
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.INFO,
                    message=f"Shifting UP: {self._cur_gear}",
                    ev_type=EV_TYPE.SHIFTING,
                    payload={
                        DField.GEAR: self._cur_gear,
                        DField.SEQ_LENGTH: self._cur_seq_length,
                        DField.BATCH_SIZE: self._cur_batch_size,
                    },
                )
            )
        elif (
            chunk_count < DOWNSHIFT_COUNT_THRESHOLD
            and self._cooldown_count > NUM_COOLDOWN_EPISODES
        ):
            # First gear is the lowest gear
            if self._cur_gear == 1:
                return self._cur_gear

            self._cur_gear -= 1
            self._cooldown_count = 0
            self._cur_seq_length, self._cur_batch_size = ATH_GEARBOX[
                self._cur_gear
            ]
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.INFO,
                    message=f"Shifting DOWN: {self._cur_gear}",
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
        for bucket_idx in THRESHOLD_BUCKETS:
            count += self._cur_bucket_counts[bucket_idx]
        return count

    def incr_cooldown_count(self):
        self._cooldown_count += 1
