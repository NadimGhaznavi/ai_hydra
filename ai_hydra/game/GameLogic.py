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
Game logic module for Snake Game mechanics.

All functions operate on immutable GameBoard instances and return new instances.
Randomness is provided explicitly via an rng parameter (no global random, no RNG
stored in the board). This supports deterministic replay and exact cloning.
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


class GameLogic:
    """
    Pure functions for Snake mechanics.

    Deterministic given: (board, action, rng state).
    """

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
        if isinstance(action, int):
            # Map int -> MoveAction (define your encoding once and keep it stable)
            # 0=LEFT, 1=STRAIGHT, 2=RIGHT
            if action == 0:
                action = MoveAction.LEFT_TURN
            elif action == 2:
                action = MoveAction.RIGHT_TURN
            else:
                action = MoveAction.STRAIGHT

        # Build the move (fixes the undefined 'move' bug)
        move = GameLogic.create_move(board.direction, action)

        # Increment move count first
        new_move_count = board.move_count + 1

        # Max moves check (before executing the move)
        snake_length = board.get_snake_length()
        max_moves = DGameDef.MAX_MOVES_MULTIPLIER * snake_length
        if new_move_count > max_moves:
            max_moves_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                grid_size=board.grid_size,
            )
            return MoveResult(
                new_board=max_moves_board,
                reward=GameLogic.calculate_reward(DGameField.MAX_MOVES),
                outcome=DGameField.MAX_MOVES,
                is_terminal=True,
            )

        # Compute new head position
        new_head = Position(
            board.snake_head.x + move.resulting_direction.dx,
            board.snake_head.y + move.resulting_direction.dy,
        )

        # Wall collision
        if not board.is_position_within_bounds(new_head):
            collision_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                grid_size=board.grid_size,
            )
            return MoveResult(
                new_board=collision_board,
                reward=GameLogic.calculate_reward(DGameField.WALL),
                outcome=DGameField.WALL,
                is_terminal=True,
            )

        # Self collision
        if board.is_position_occupied_by_snake(new_head):
            collision_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                grid_size=board.grid_size,
            )
            return MoveResult(
                new_board=collision_board,
                reward=GameLogic.calculate_reward(DGameField.SNAKE),
                outcome=DGameField.SNAKE,
                is_terminal=True,
            )

        # Food check
        ate_food = new_head == board.food_position

        if ate_food:
            # Grow: keep all body segments
            new_body = (board.snake_head,) + board.snake_body
            new_score = board.score + 1

            new_food_position = GameLogic._generate_food_position(
                head=new_head,
                body=new_body,
                grid_size=board.grid_size,
                rng=rng,
            )

            new_board = GameBoard(
                snake_head=new_head,
                snake_body=new_body,
                direction=move.resulting_direction,
                food_position=new_food_position,
                score=new_score,
                move_count=new_move_count,
                grid_size=board.grid_size,
            )

            outcome = DGameField.FOOD
            return MoveResult(
                new_board=new_board,
                reward=GameLogic.calculate_reward(outcome),
                outcome=outcome,
                is_terminal=bool(food_ends_episode),
            )

        # Empty move: shift body (drop tail)
        new_body = (board.snake_head,) + board.snake_body[:-1]

        new_board = GameBoard(
            snake_head=new_head,
            snake_body=new_body,
            direction=move.resulting_direction,
            food_position=board.food_position,
            score=board.score,
            move_count=new_move_count,
            grid_size=board.grid_size,
        )

        outcome = DGameField.EMPTY
        return MoveResult(
            new_board=new_board,
            reward=GameLogic.calculate_reward(outcome),
            outcome=outcome,
            is_terminal=False,
        )

    @staticmethod
    def _generate_food_position(
        head: Position,
        body: Tuple[Position, ...],
        grid_size: Tuple[int, int],
        rng: random.Random,
    ) -> Position:
        """
        Generate a new food position that doesn't overlap with the snake.
        """
        width, height = grid_size
        snake_positions = {head} | set(body)

        available_positions: list[Position] = []
        for x in range(width):
            for y in range(height):
                pos = Position(x, y)
                if pos not in snake_positions:
                    available_positions.append(pos)

        if not available_positions:
            # Should be impossible in normal play unless snake fills the board
            return Position(0, 0)

        return rng.choice(available_positions)

    @staticmethod
    def create_move(current_direction: Direction, action: MoveAction) -> Move:
        if action == MoveAction.LEFT_TURN:
            resulting_direction = current_direction.turn_left()
        elif action == MoveAction.RIGHT_TURN:
            resulting_direction = current_direction.turn_right()
        else:
            resulting_direction = current_direction
        return Move(action=action, resulting_direction=resulting_direction)

    @staticmethod
    def get_possible_moves(current_direction: Direction) -> List[Move]:
        return [
            GameLogic.create_move(current_direction, MoveAction.LEFT_TURN),
            GameLogic.create_move(current_direction, MoveAction.STRAIGHT),
            GameLogic.create_move(current_direction, MoveAction.RIGHT_TURN),
        ]

    @staticmethod
    def calculate_reward(outcome: str) -> int:
        reward_map = {
            DGameField.EMPTY: DGameDef.EMPTY_MOVE_REWARD,
            DGameField.FOOD: DGameDef.FOOD_REWARD,
            DGameField.WALL: DGameDef.COLLISION_PENALTY,
            DGameField.SNAKE: DGameDef.COLLISION_PENALTY,
            DGameField.MAX_MOVES: DGameDef.EMPTY_MOVE_REWARD,
        }
        return int(reward_map.get(outcome, 0))
