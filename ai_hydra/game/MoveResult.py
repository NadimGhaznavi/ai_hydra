# ai_hydra/game/MoveResult.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations
from dataclasses import dataclass

from ai_hydra.game.GameBoard import GameBoard  # adjust path to your real one
from ai_hydra.constants.DGame import DGameField


@dataclass(frozen=True)
class MoveResult:
    new_board: GameBoard
    reward: int
    outcome: str
    is_terminal: bool

    def is_collision(self) -> bool:
        return self.outcome in (DGameField.WALL, DGameField.SNAKE)

    def is_food_eaten(self) -> bool:
        return self.outcome == DGameField.FOOD
