# ai_hydra/nnet/ATH/ATHMemory.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from random import Random

from ai_hydra.constants.DHydra import DHydraLog, DModule

from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.nnet.Transition import Transition
from ai_hydra.nnet.ATH.ATHGearbox import ATHGearBox
from ai_hydra.nnet.ATH.ATHDataStore import ATHDataStore
from ai_hydra.nnet.ATH.ATHDataMgr import ATHDataMgr


class ATHMemory:
    def __init__(self, rng: Random, log_level: DHydraLog, pub_func):
        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_MEMORY,
            log_level=log_level,
            to_console=True,
        )

        # Memory storage
        data_store = ATHDataStore(log_level=log_level, pub_func=pub_func)

        # Memory Management
        self.data_mgr = ATHDataMgr(
            log_level=log_level,
            rng=rng,
            pub_func=pub_func,
            data_store=data_store,
        )

        # The ATH Gearbox a.k.a. "The Automatic ATH Transmission."
        self.gearbox = ATHGearBox(
            log_level=log_level, pub_func=pub_func, data_store=data_store
        )

    async def append(
        self, t: Transition, final_score: int | None = None
    ) -> None:
        if t.done:
            self.gearbox.incr_cooldown_count()
        await self.data_mgr.append(t=t, final_score=final_score)
        self.data_mgr.set_gear(await self.gearbox.get_gear())
