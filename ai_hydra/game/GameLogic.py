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
from ai_hydra.game.GameHelper import MoveResult, RewardCfg

# When the snake length is greater than the SHAPING_REWARD_THRESH, then
# disable the shaping reward. This is to encourage the discovery of
# "folding behaviour" in the later stages of the game.

SHAPING_REWARD_THRESH = 6


class GameLogic:
    """
    Pure functions for Snake mechanics.

    Deterministic given: (board, action, rng state).
    """

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
    def _food_direction_reward(
        head: Position,
        food: Position,
        move_dx: int,
        move_dy: int,
    ) -> str:
        """
        Return the reward field based on whether the move direction points
        toward or away from the food.
        """
        food_dx = food.x - head.x
        food_dy = food.y - head.y

        if move_dx != 0:
            if food_dx == 0:
                return DGameField.FURTHER_FROM_FOOD
            return (
                DGameField.CLOSER_TO_FOOD
                if (food_dx > 0 and move_dx > 0)
                or (food_dx < 0 and move_dx < 0)
                else DGameField.FURTHER_FROM_FOOD
            )

        if move_dy != 0:
            if food_dy == 0:
                return DGameField.FURTHER_FROM_FOOD
            return (
                DGameField.CLOSER_TO_FOOD
                if (food_dy > 0 and move_dy > 0)
                or (food_dy < 0 and move_dy < 0)
                else DGameField.FURTHER_FROM_FOOD
            )

        return DGameField.EMPTY

    @staticmethod
    def get_new_snakehead(board: GameBoard, move: Move):
        return Position(
            board.snake_head.x + move.resulting_direction.dx,
            board.snake_head.y + move.resulting_direction.dy,
        )

    @staticmethod
    def _manhattan_distance(a: Position, b: Position) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    @staticmethod
    def normalize_action(action: str | int) -> MoveAction:
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
        return action

    @staticmethod
    def step(
        board: GameBoard,
        action: MoveAction | int,
        rng: random.Random,
        reward_cfg: RewardCfg,
        mmm: int,  # Max-Moves-Multiplier
        reward_shaping: bool,
        food_ends_episode: bool = False,
    ) -> MoveResult:
        """
        Execute one action and return a MoveResult containing the new board.

        Notes:
          - rng is the ONLY source of randomness.
          - board is treated as immutable; a new GameBoard is returned.
        """

        # Normalize action
        action = GameLogic.normalize_action(action)

        # Build the move (fixes the undefined 'move' bug)
        move = GameLogic.create_move(board.direction, action)

        # Increment move count first
        new_move_count = board.move_count + 1

        # Max moves check (before executing the move)
        snake_length = board.get_snake_length()

        # mmm == Max Moves Multiplier
        max_moves = mmm * snake_length

        # Exceeded max moves
        if new_move_count > max_moves:
            max_moves_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                grid_size=board.grid_size,
                seed=board.seed,
                episode_id=board.episode_id,
            )
            return MoveResult(
                new_board=max_moves_board,
                reward=reward_cfg.get(DGameField.MAX_MOVES),
                outcome=DGameField.MAX_MOVES,
                is_terminal=True,
            )

        # Compute new head position
        new_head = GameLogic.get_new_snakehead(board=board, move=move)

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
                seed=board.seed,
                episode_id=board.episode_id,
            )
            return MoveResult(
                new_board=collision_board,
                reward=reward_cfg.get(DGameField.WALL),
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
                seed=board.seed,
                episode_id=board.episode_id,
            )
            return MoveResult(
                new_board=collision_board,
                reward=reward_cfg.get(DGameField.SNAKE),
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
                seed=board.seed,
                episode_id=board.episode_id,
            )

            return MoveResult(
                new_board=new_board,
                reward=reward_cfg.get(DGameField.FOOD),
                outcome=DGameField.FOOD,
                is_terminal=bool(food_ends_episode),
            )

        # Empty move....
        base_reward = reward_cfg.get(DGameField.EMPTY)

        # Shaping reward: moving closer/further to/from food.
        shaping_reward = 0.0

        # Disable the shaping reward once the snake is longer than
        # if snake_length <= SHAPING_REWARD_THRESH:

        # Disable the shaping reward (set by the SnakeMgr, based on a
        # highscore value)
        # if reward_shaping:

        if True:
            reward_field = GameLogic._food_direction_reward(
                head=board.snake_head,
                food=board.food_position,
                move_dx=move.resulting_direction.dx,
                move_dy=move.resulting_direction.dy,
            )
            shaping_reward = reward_cfg.get(reward_field)

        reward = base_reward + shaping_reward

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
            seed=board.seed,
            episode_id=board.episode_id,
        )

        outcome = DGameField.EMPTY
        return MoveResult(
            new_board=new_board,
            reward=reward,
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
    def get_possible_moves(current_direction: Direction) -> List[Move]:
        return [
            GameLogic.create_move(current_direction, MoveAction.LEFT_TURN),
            GameLogic.create_move(current_direction, MoveAction.STRAIGHT),
            GameLogic.create_move(current_direction, MoveAction.RIGHT_TURN),
        ]

    @staticmethod
    def would_collide(board: GameBoard, action: int | str) -> bool:
        action = GameLogic.normalize_action(action)
        move = GameLogic.create_move(
            current_direction=board.direction, action=action
        )
        new_snakehead = GameLogic.get_new_snakehead(board=board, move=move)

        if not board.is_position_within_bounds(new_snakehead):
            return True

        if board.is_position_occupied_by_snake(new_snakehead):
            return True

        return False
