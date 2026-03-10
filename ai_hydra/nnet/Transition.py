# ai_hydra_nnet/Transition.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class Transition:
    old_state: Tuple[float, ...]
    action: int
    reward: float
    new_state: Tuple[float, ...]
    done: bool
