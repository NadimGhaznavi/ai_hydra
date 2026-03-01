# ai_hydra/nnet/EpsilonAlgo.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

import random
from typing import Optional

from ai_hydra.constants.DNet import DEpsilonDef, DEpsilonField


class EpsilonAlgo:
    """
    Epsilon-greedy helper.

    - Call maybe_random_action() each step.
      If it returns an int action (0..2), take it.
      If it returns None, use your policy's greedy action.

    - Call played_game() at end of episode to decay epsilon.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng

        self._initial_epsilon = float(DEpsilonDef.INITIAL)
        self._epsilon_min = float(DEpsilonDef.MINIMUM)
        self._epsilon_decay = float(DEpsilonDef.DECAY_RATE)

        self._cur_epsilon = self._initial_epsilon
        self._injected = 0
        self._depleted = False

    def cur_epsilon(self) -> float:
        return self._cur_epsilon

    def get_params(self) -> dict[str, float]:
        return {
            DEpsilonField.INITIAL: self._initial_epsilon,
            DEpsilonField.MINIMUM: self._epsilon_min,
            DEpsilonField.DECAY_RATE: self._epsilon_decay,
        }

    def injected(self) -> int:
        return self._injected

    def maybe_random_action(self) -> Optional[int]:
        """
        Returns:
            int action in {0,1,2} if we inject randomness, else None.
        """
        if self._rng.random() < self._cur_epsilon:
            self._injected += 1
            return self._rng.randrange(3)
        return None

    def played_game(self) -> None:
        """
        Decay epsilon at the end of an episode.
        """
        self._cur_epsilon = max(
            self._epsilon_min, self._cur_epsilon * self._epsilon_decay
        )
        self._depleted = self._cur_epsilon <= self._epsilon_min
        self.reset_injected()

    def reset(self) -> None:
        """
        Start a fresh schedule (called at START_RUN or RESET)
        """
        self._cur_epsilon = self._initial_epsilon
        self._depleted = self.cur_epsilon <= self._epsilon_min
        self.reset_injected()

    def reset_injected(self) -> None:
        """
        Reset the "injected random moves" count.
        """
        self._injected = 0

    def set_params(
        self, *, initial: float, minimum: float, decay: float
    ) -> None:
        self._initial_epsilon = float(initial)
        self._epsilon_min = float(minimum)
        self._epsilon_decay = float(decay)
