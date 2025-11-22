# ai-hydra/env/Env.py

from env.BaseEnv import BaseEnv

# -------------------------
# Environment & Policy (simulation)
# -------------------------


class Env(BaseEnv):
    def __init__(self, size: int, light_decay: float):
        super().__init__(size, light_decay)
