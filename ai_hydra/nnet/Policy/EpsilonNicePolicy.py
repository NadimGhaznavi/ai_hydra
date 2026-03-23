# ai_hydra/nnet/Policy/EpsilonNicePolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Sequence
import random

from ai_hydra.constants.DNNet import DRNN
from ai_hydra.constants.DHydra import DHydraLog, DModule

from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.EpsilonNiceAlgo import EpsilonNiceAlgo
from ai_hydra.game.GameBoard import GameBoard
from ai_hydra.utils.HydraLog import HydraLog


class EpsilonNicePolicy(HydraPolicy):
    def __init__(
        self,
        log_level: DHydraLog,
        base_policy: HydraPolicy,
        epsilon_n: EpsilonNiceAlgo,
        p_value: float,
        steps: int,
        rng: random.Random,
    ):

        self.log = HydraLog(
            client_id=DModule.EPSILON_NICE_POLICY,
            log_level=log_level,
            to_console=True,
        )

        self.base_policy = base_policy
        self.epsilon_n = epsilon_n
        self._rng = rng
        self._p_value = p_value
        self._steps = steps

        self.log.info(f"Nice P-Value set: {p_value}")
        self.log.info(f"Nice STEPS: {steps}")

        self._nice_steps_remaining = 0

    def select_action(self, state: Sequence[float], board: GameBoard) -> int:

        suggested = self.base_policy.select_action(state, board)

        # Normal Epsilon is still active, do nothing
        if self.base_policy.cur_epsilon() > 0.0001:
            self._nice_steps_remaining = 0
            return suggested

        self.epsilon_n.incr_calls()

        # We're on a "Nice" detour, lets explore!!! :)
        if self._nice_steps_remaining > 0:
            self._nice_steps_remaining -= 1
            return self.epsilon_n.override_action(
                suggested_action=suggested,
                board=board,
            )

        # Determine whether or not to activate EpsilonNice (on the next step)
        if self._rng.random() < self._p_value:
            self._nice_steps_remaining = self._steps

        # Let the NN decide as it normally does....
        return suggested

    async def played_game(self):
        await self.base_policy.played_game()
        await self.epsilon_n.played_game()

    def cur_epsilon(self):
        return self.base_policy.cur_epsilon()
