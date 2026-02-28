# ai_hydra/nnet/RandomPolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import random
from typing import Sequence

from ai_hydra.nnet.HydraPolicy import HydraPolicy


class RandomPolicy(HydraPolicy):
    def __init__(self, rng: random.Random):
        self._rng = rng

    def select_action(self, state: Sequence[float]) -> int:
        return self._rng.randint(0, 2)
