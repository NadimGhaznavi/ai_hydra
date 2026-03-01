from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ai_hydra.game.GameLogic import GameLogic
from ai_hydra.game.GameBoard import GameBoard, MoveAction
from ai_hydra.constants.DGame import DGameField


@dataclass(frozen=True)
class LookaheadDecision:
    fatal: bool
    food: bool


class LookaheadPolicy:
    """
    Wrapper policy: takes a base policy's suggested action,
    rejects it if it's instant death (wall/snake), optionally prefers food.

    ON/OFF is controlled outside (per episode), via a bool passed into select_action().
    """

    def __init__(self, base_policy):
        self._base = base_policy

    def played_game(self) -> None:
        # preserve the decay path (epsilon, etc.)
        if hasattr(self._base, "played_game"):
            self._base.played_game()

    def cur_epsilon(self) -> float | None:
        # preserve introspection if present
        if hasattr(self._base, "cur_epsilon"):
            return self._base.cur_epsilon()
        return None

    @staticmethod
    def _preview(board: GameBoard, action: int) -> LookaheadDecision:
        # normalize action to MoveAction without consuming RNG
        if action == 0:
            act = MoveAction.LEFT_TURN
        elif action == 1:
            act = MoveAction.STRAIGHT
        elif action == 2:
            act = MoveAction.RIGHT_TURN
        else:
            raise ValueError(f"Invalid action: {action}")

        move = GameLogic.create_move(board.direction, act)
        new_head = board.snake_head.__class__(
            board.snake_head.x + move.resulting_direction.dx,
            board.snake_head.y + move.resulting_direction.dy,
        )

        # food is “auto win” for lookahead purposes
        if new_head == board.food_position:
            return LookaheadDecision(fatal=False, food=True)

        # wall/self collision checks
        if not board.is_position_within_bounds(new_head):
            return LookaheadDecision(fatal=True, food=False)

        if board.is_position_occupied_by_snake(new_head):
            return LookaheadDecision(fatal=True, food=False)

        return LookaheadDecision(fatal=False, food=False)

    def select_action(
        self, state: Sequence[float], *, board: GameBoard, lookahead_on: bool
    ) -> int:
        # base suggestion first (keeps “policy stack” intact)
        suggested = int(self._base.select_action(state))

        if not lookahead_on:
            return suggested

        d = self._preview(board, suggested)
        if not d.fatal:
            return suggested

        # If suggested action is fatal, try alternatives.
        # Prefer food, else any safe action. Keep it deterministic (fixed ordering).
        candidates = [0, 1, 2]

        best_safe = None
        for a in candidates:
            if a == suggested:
                continue
            dec = self._preview(board, a)
            if dec.food:
                return a
            if not dec.fatal and best_safe is None:
                best_safe = a

        if best_safe is not None:
            return best_safe

        # No safe moves: death is inevitable
        return suggested
