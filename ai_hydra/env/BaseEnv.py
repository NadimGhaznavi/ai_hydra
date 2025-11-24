# ai_hydra/env/BaseEnv.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai-hydra
#    Website: https://aihydra.osoyalce.com
#    License: GPL 3.0

# -------------------------
# Environment & Policy (simulation)
# -------------------------
class BaseEnv:
    # A tiny environment class for the Snake Lab-style grid.
    # Two parameters:
    #  - size: grid size or similar
    #  - light_decay: parameter for the light attenuation model
    def __init__(self, size: int, light_decay: float):
        self._size = size
        self._light_decay = light_decay
