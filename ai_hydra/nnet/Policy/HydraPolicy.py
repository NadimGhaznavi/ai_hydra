# ai_hydra/nnet/HydraPolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from abc import ABC, abstractmethod
from typing import Sequence


class HydraPolicy(ABC):

    @abstractmethod
    def select_action(self, state: Sequence[float]) -> int:
        """
        Return action index: 0, 1, 2
        (LEFT, STRAIGHT, RIGHT)
        """
        pass


class HydraTrainPolicy(HydraPolicy):
    @abstractmethod
    def played_game(self) -> None:
        """
        Called exactly once at the end of each episode (epsilon decay).
        """
        pass
