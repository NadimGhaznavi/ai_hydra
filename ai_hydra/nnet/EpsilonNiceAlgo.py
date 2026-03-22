# ai_hydra/nnet/EpsilonNiceAlgo.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

import random

from ai_hydra.constants.DHydra import DHydraLog, DModule
from ai_hydra.constants.DEpsilonNice import DEpsilonNice
from ai_hydra.game.GameBoard import GameBoard
from ai_hydra.game.GameLogic import GameLogic
from ai_hydra.utils.HydraLog import HydraLog
from ai_hydra.zmq.HydraEventMQ import EventMsg, HydraEventMQ


class EpsilonNiceAlgo:
    """
    Post-epsilon collision rescue helper.

    Behavior:
      - Called after the normal policy has selected an action.
      - With probability `p_value`, checks whether the suggested action
        would immediately collide.
      - If so, attempts to replace it with a safe alternative.
      - Otherwise, leaves the action unchanged.

    Stats:
      - calls: total invocations
      - triggered: times the probability gate fired
      - overrides: times the suggested action was replaced
      - no_safe_alternative: times the suggested action was fatal and no
        non-fatal alternative existed
    """

    def __init__(
        self,
        rng: random.Random,
        log_level: DHydraLog,
        pub_func,
        p_value: float,
    ) -> None:
        self._rng = rng
        self._p_value = float(p_value)

        self.event = HydraEventMQ(
            client_id=DModule.EPSILON_NICE_ALGO,
            pub_func=pub_func,
        )

        self.log = HydraLog(
            client_id=DModule.EPSILON_NICE_ALGO,
            log_level=log_level,
            to_console=True,
        )
        self._epoch = 0
        self.log.info(f"P-Value set: {p_value}")
        self._reset_window()

    def maybe_override_action(
        self,
        suggested_action: int,
        board: GameBoard,
    ) -> int:
        """
        Return the action to execute.

        Usually returns `suggested_action` unchanged.
        With probability `p_value`, try to replace it with a different
        non-fatal action. If no safe alternative exists, keep the original.
        """
        self._calls += 1

        if self._rng.random() >= self._p_value:
            return suggested_action

        self._triggered += 1

        safe_alternatives = [
            action
            for action in range(3)
            if action != suggested_action
            and not GameLogic.would_collide(board, action)
        ]

        if not safe_alternatives:
            self._no_safe_alternative += 1
            return suggested_action

        if GameLogic.would_collide(board, suggested_action):
            self._fatal_suggested += 1

        self._overrides += 1
        return self._rng.choice(safe_alternatives)

    def get_stats(self) -> dict[str, int | float]:
        trigger_rate = self._triggered / self._calls if self._calls else 0.0
        override_rate = self._overrides / self._calls if self._calls else 0.0

        return {
            DEpsilonNice.EPOCH: self._epoch,
            DEpsilonNice.CALLS: self._calls,
            DEpsilonNice.TRIGGERED: self._triggered,
            DEpsilonNice.FATAL_SUGGESTED: self._fatal_suggested,
            DEpsilonNice.OVERRIDES: self._overrides,
            DEpsilonNice.NO_SAFE_ALTERNATIVE: self._no_safe_alternative,
            DEpsilonNice.TRIGGER_RATE: round(trigger_rate, 6),
            DEpsilonNice.OVERRIDE_RATE: round(override_rate, 6),
        }

    async def played_game(self) -> None:
        self._epoch += 1
        if self._epoch % 100 == 0:
            payload = self.get_stats()
            payload[DEpsilonNice.WINDOW] = f"{self._epoch-99}-{self._epoch}"
            await self.event.publish(
                EventMsg(
                    level=DHydraLog.INFO,
                    payload=payload,
                )
            )
            self._reset_window()

    def _reset_window(self) -> None:
        """
        Reset rolling window counters.
        """
        self._calls = 0
        self._triggered = 0
        self._overrides = 0
        self._no_safe_alternative = 0
        self._fatal_suggested = 0
