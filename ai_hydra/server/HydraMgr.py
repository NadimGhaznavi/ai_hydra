# ai_hydra/server/HydraMgr.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from ai_hydra.server.HydraServer import HydraServer
from ai_hydra.server.SnakeMgr import SnakeMgr

from ai_hydra.constants.DGame import DGameMethod


class HydraMgr(HydraServer):

    def __init__(self):
        super().__init__()
        
        self.snake = SnakeMgr()

        self._methods.update(
            {
                DGameMethod.RESET: self.reset_game,
                DGameMethod.STEP: self.game_step,
            }
        )
