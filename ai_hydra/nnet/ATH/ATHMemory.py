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
    def __init__(
        self,
        rng: Random,
        log_level: DHydraLog,
        pub_func,
        max_buckets: int,
        max_gear: int,
        max_training_frames: int,
        max_frames: int,
        upshift_count_thresh: int,
        downshift_count_thresh: int,
        num_cooldown_eps: int,
    ):
        # Local logging
        self.log = HydraLog(
            client_id=DModule.ATH_MEMORY,
            log_level=log_level,
            to_console=True,
        )
        self.log.info(f"Set max buckets: {max_buckets}")
        self.log.info(f"Set max gear: {max_gear}")
        self.log.info(f"Set max training frames: {max_training_frames}")
        self.log.info(f"Set frames: {max_frames}")
        self.log.info(f"Set upshift count threshold: {upshift_count_thresh}")
        self.log.info(
            f"Set downshift count threshold: {downshift_count_thresh}"
        )
        self.log.info(f"Set number of cooldown episodes: {num_cooldown_eps}")

        # Memory storage
        self.store = ATHDataStore(
            log_level=log_level,
            max_buckets=max_buckets,
            max_gear=max_gear,
            max_training_frames=max_training_frames,
        )

        # Memory Management
        self.data_mgr = ATHDataMgr(
            log_level=log_level,
            rng=rng,
            pub_func=pub_func,
            data_store=self.store,
            max_frames=max_frames,
            max_gear=max_gear,
            max_training_frames=max_training_frames,
        )

        # The ATH Gearbox a.k.a. "The Automatic ATH Transmission."
        self.gearbox = ATHGearBox(
            log_level=log_level,
            pub_func=pub_func,
            data_store=self.store,
            max_buckets=max_buckets,
            upshift_count_thresh=upshift_count_thresh,
            downshift_count_thresh=downshift_count_thresh,
            num_cooldown_eps=num_cooldown_eps,
            max_gear=max_gear,
            max_training_frames=max_training_frames,
        )

    async def append(
        self, t: Transition, final_score: int | None = None
    ) -> None:
        if t.done:
            self.gearbox.incr_cooldown_count()
        await self.data_mgr.append(t=t, final_score=final_score)
        self.data_mgr.set_gear(await self.gearbox.get_gear())
