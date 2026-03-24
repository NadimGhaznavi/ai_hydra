# ai_hydra/nnet/EpsilonAlgo.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

import zmq.asyncio

import random
from typing import Optional, Callable, Awaitable

from ai_hydra.constants.DNNet import DEpsilonField
from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DEvent import EV_STATUS

from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import EventMsg, HydraEventMQ


class EpsilonAlgo:
    """
    Epsilon-greedy helper.

    - Call maybe_random_action() each step.
      If it returns an int action (0..2), take it.
      If it returns None, use your policy's greedy action.

    - Call played_game() at end of episode to decay epsilon.
    """

    def __init__(
        self,
        rng: random.Random,
        log_level: DHydraLog,
        pub_func,
    ):
        self._rng = rng

        self.event = HydraEventMQ(
            client_id=DModule.EPSILON_ALGO, pub_func=pub_func
        )

        self._initial_epsilon = None
        self._min_epsilon = None
        self._decay_rate = None

        self._cur_epsilon = None
        self._injected = 0
        self._depleted = False
        self._epsilon_not_depleted = True
        self.log = HydraLog(
            client_id=DModule.EPSILON_ALGO,
            log_level=log_level,
            to_console=True,
        )

    def cur_epsilon(self) -> float:
        return self._cur_epsilon

    def decay_rate(self, value: float) -> None:
        self._decay_rate = value
        self.log.debug(f"Epsilon decay rate set: {value}")

    def get_params(self) -> dict[str, float]:
        return {
            DEpsilonField.INITIAL: self._initial_epsilon,
            DEpsilonField.MINIMUM: self._min_epsilon,
            DEpsilonField.DECAY_RATE: self._decay_rate,
        }

    def initial_epsilon(self, value: float) -> None:
        self._initial_epsilon = value
        self._cur_epsilon = value
        self.log.debug(f"Initial epsilon set: {value}")

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

    def min_epsilon(self, value: float) -> None:
        self._min_epsilon = value
        self.log.debug(f"Minimum epsilon set: {value}")

    async def played_game(self) -> None:
        """
        Decay epsilon at the end of an episode.
        """
        self._cur_epsilon = max(
            self._min_epsilon, self._cur_epsilon * self._decay_rate
        )
        self._depleted = self._cur_epsilon <= self._min_epsilon

        if self._cur_epsilon < 0.001:
            self._cur_epsilon = 0.0

        if self._epsilon_not_depleted and self._depleted:
            msg = f"Exploration complete: ε = {self._min_epsilon}"
            self.log.info(msg)
            await self.event.publish(
                EventMsg(level=EV_STATUS.INFO, message=msg)
            )
            self._epsilon_not_depleted = False

        self.reset_injected()

    def reset(self) -> None:
        """
        Start a fresh schedule (called at START_RUN or RESET)
        """
        self._cur_epsilon = self._initial_epsilon
        self._depleted = self._cur_epsilon <= self._min_epsilon
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
        self._min_epsilon = float(minimum)
        self._decay_rate = float(decay)
