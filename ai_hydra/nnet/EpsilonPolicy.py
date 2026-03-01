# ai_hydra/nnet/EpsilonPolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Sequence

from ai_hydra.nnet.HydraPolicy import HydraPolicy
from ai_hydra.nnet.EpsilonAlgo import EpsilonAlgo


class EpsilonPolicy(HydraPolicy):
    """
    Wraps a base policy and injects epsilon exploration.
    """

    def __init__(self, base_policy: HydraPolicy, epsilon: EpsilonAlgo):
        self._base_policy = base_policy
        self._epsilon = epsilon

    def select_action(self, state: Sequence[float]) -> int:
        a = self._epsilon.maybe_random_action()
        if a is not None:
            return a
        return self._base_policy.select_action(state)

    def played_game(self) -> None:
        self._epsilon.played_game()

    def cur_epsilon(self) -> float:
        return self._epsilon.cur_epsilon()
