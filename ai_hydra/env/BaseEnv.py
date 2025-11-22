# ai-hydra/env/BaseEnv.py


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
