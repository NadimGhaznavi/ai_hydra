# ai_hydra/nnet/Policy/EpsilonNicePolicy.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Sequence

from ai_hydra.nnet.Policy.HydraPolicy import HydraPolicy
from ai_hydra.nnet.EpsilonNiceAlgo import EpsilonNiceAlgo
from ai_hydra.game.GameBoard import GameBoard


class EpsilonNicePolicy(HydraPolicy):
    def __init__(self, base_policy: HydraPolicy, epsilon_n: EpsilonNiceAlgo):
        self._base_policy = base_policy
        self._epsilon_n = epsilon_n

    def select_action(self, state: Sequence[float], board: GameBoard) -> int:
        suggested = self._base_policy.select_action(state, board)

        if self._base_policy.cur_epsilon() > 0.0001:
            return suggested

        return self._epsilon_n.maybe_override_action(
            suggested_action=suggested,
            board=board,
        )

    async def played_game(self):
        await self._base_policy.played_game()
        await self._epsilon_n.played_game()

    def cur_epsilon(self):
        return self._base_policy.cur_epsilon()
