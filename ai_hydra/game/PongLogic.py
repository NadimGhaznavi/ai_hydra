# ai_hydra/game/GameLogic.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0
#

"""
Game logic for the Pong game.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from ai_hydra.constants.DGame import DGameDef, DGameField
from ai_hydra.game.GameBoard import (
    Direction,
    GameBoard,
    Move,
    MoveAction,
    Position,
)
from ai_hydra.game.MoveResult import MoveResult


class PongLogic:

    @staticmethod
    def step(
        board: GameBoard,
        action: MoveAction | int,
        rng: random.Random,
        *,
        food_ends_episode: bool = False,
    ) -> MoveResult:
        """
        Execute one action and return a MoveResult containing the new board.

        Notes:
          - rng is the ONLY source of randomness.
          - board is treated as immutable; a new GameBoard is returned.
        """

        # Normalize action
        if isinstance(action, str):
            action = int(action)

        if isinstance(action, int):
            if action == 0:
                action = MoveAction.LEFT_TURN
            elif action == 1:
                action = MoveAction.STRAIGHT
            elif action == 2:
                action = MoveAction.RIGHT_TURN
            else:
                raise ValueError(f"Invalid action index: {action}")

        # Build the move (fixes the undefined 'move' bug)
        move = PongLogic.create_move(board.direction, action)

        # Compute new head position
        new_head = Position(
            board.snake_head.x + move.resulting_direction.dx,
            board.snake_head.y + move.resulting_direction.dy,
        )
