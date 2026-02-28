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
    Wraps a greedy policy and injects epsilon exploration.
    """

    def __init__(self, policy: HydraPolicy, epsilon: EpsilonAlgo):
        self._policy = policy
        self._epsilon = epsilon

    def select_action(self, state: Sequence[float]) -> int:
        a = self._epsilon.maybe_random_action()
        if a is not None:
            return a
        return self._policy.select_action(state)

    def played_game(self) -> None:
        self._epsilon.played_game()

    def epsilon(self) -> float:
        return self._epsilon.epsilon()
