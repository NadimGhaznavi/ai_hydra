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

from ai_hydra.constants.DNet import DEpsilonDef  # assuming you have this


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

        self._epsilon = self._initial_epsilon
        self._injected = 0
        self._depleted = False

    def maybe_random_action(self) -> Optional[int]:
        """
        Returns:
            int action in {0,1,2} if we inject randomness, else None.
        """
        if self._rng.random() < self._epsilon:
            self._injected += 1
            return self._rng.randrange(3)
        return None

    def epsilon(self) -> float:
        return self._epsilon

    def played_game(self) -> None:
        """
        Decay epsilon at the end of an episode.
        """
        self._epsilon = max(
            self._epsilon_min, self._epsilon * self._epsilon_decay
        )
        self._depleted = self._epsilon <= self._epsilon_min
        self.reset_injected()

    def reset_injected(self) -> None:
        self._injected = 0

    def injected(self) -> int:
        return self._injected

    # Optional knobs (keep your existing style)
    def epsilon_decay(self, epsilon_decay: float | None = None) -> float:
        if epsilon_decay is not None:
            self._epsilon_decay = float(epsilon_decay)
        return self._epsilon_decay

    def epsilon_min(self, epsilon_min: float | None = None) -> float:
        if epsilon_min is not None:
            self._epsilon_min = float(epsilon_min)
        return self._epsilon_min

    def initial_epsilon(self, initial_epsilon: float | None = None) -> float:
        if initial_epsilon is not None:
            self._initial_epsilon = float(initial_epsilon)
        return self._initial_epsilon
