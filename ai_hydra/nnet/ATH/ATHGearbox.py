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

THRESHOLDS_REQUIRED = DMemDef.THRESHOLDS_REQUIRED


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
        self._cur_seq_length, self._batch_size = ATH_GEARBOX[self._cur_gear]

        self._threshold_achieved_count = 0

    async def get_gear(self) -> int:
        if self._cur_gear == max(ATH_GEARBOX):
            return self._cur_gear

        """
        # OLD MANUAL TRANSMISSION LOGIC
        # -----------------------------

        # The refactor is about creating an algorithm to replace this...

        promotion_score, _, _ = GEARBOX[self._cur_gear]

        if final_score < promotion_score:
            return self._cur_gear

        self._threshold_achieved_count += 1

        if self._threshold_achieved_count < THRESHOLDS_REQUIRED:
            return self._cur_gear

        self._cur_gear += 1
        self._threshold_achieved_count = 0
        _, self._cur_seq_length, self._batch_size = GEARBOX[self._cur_gear]

        self.log.info(
            f"Shifting into higher gear: {self._cur_gear}"
            f"(seq_length={self._cur_seq_length}, batch_size={self._batch_size})"
        )

        await self.event.publish(
            EventMsg(
                level=DHydraLog.INFO,
                message=f"Shifting into higher gear: {self._cur_gear}",
                ev_type=EV_TYPE.SHIFTING,
                payload={
                    DField.GEAR: self._cur_gear,
                    DField.SEQ_LENGTH: self._cur_seq_length,
                    DField.BATCH_SIZE: self._batch_size,
                },
            )
        )
        """

        return self._cur_gear
