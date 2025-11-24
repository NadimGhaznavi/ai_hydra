# ai_hydra/env/Env.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai-hydra
#    Website: https://aihydra.osoyalce.com
#    License: GPL 3.0

from ai_hydra.env.BaseEnv import BaseEnv

# -------------------------
# Environment & Policy (simulation)
# -------------------------


class Env(BaseEnv):
    def __init__(self, size: int, light_decay: float):
        super().__init__(size, light_decay)
