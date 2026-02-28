# ai_hydra_nnet/Transition.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Transition:
    state: Sequence[float]
    action: int
    reward: float
    next_state: Sequence[float]
    done: bool
