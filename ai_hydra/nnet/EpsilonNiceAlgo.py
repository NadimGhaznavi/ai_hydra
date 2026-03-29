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

    def __init__(
        self,
        log_level: DHydraLog,
        pub_func,
        rng: random.Random,
    ) -> None:
        self._rng = rng

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
        self._calls = 0
        self._reset_window()

    def override_action(self, suggested_action: int, board: GameBoard) -> int:

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

    def incr_calls(self):
        self._calls += 1

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
