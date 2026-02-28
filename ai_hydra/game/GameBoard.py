# ai_hydra/game_board/GameBoard.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DGame import DGameDef, DGameLabel, DGameField
from ai_hydra.constants.DNet import DNetDef


class MoveAction(Enum):
    LEFT_TURN = DGameField.LEFT_TURN
    STRAIGHT = DGameField.STRAIGHT
    RIGHT_TURN = DGameField.RIGHT_TURN


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)


def _random_unoccupied_position(
    rng: random.Random,
    grid_size: tuple[int, int],
    occupied: set[Position],
) -> Position:
    width, height = grid_size
    free = [
        Position(x, y)
        for x in range(width)
        for y in range(height)
        if Position(x, y) not in occupied
    ]
    if not free:
        raise ValueError(DGameLabel.NO_FREE_POSITIONS)
    return rng.choice(free)


@dataclass(frozen=True)
class Direction:
    dx: int
    dy: int

    @classmethod
    def left(cls) -> "Direction":
        return cls(-1, 0)

    @classmethod
    def right(cls) -> "Direction":
        return cls(1, 0)

    @classmethod
    def up(cls) -> "Direction":
        return cls(0, -1)

    @classmethod
    def down(cls) -> "Direction":
        return cls(0, 1)

    def turn_left(self) -> "Direction":
        return Direction(self.dy, -self.dx)

    def turn_right(self) -> "Direction":
        return Direction(-self.dy, self.dx)


@dataclass(frozen=True)
class Move:
    action: MoveAction
    resulting_direction: Direction


@dataclass(frozen=True)
class GameBoard:
    snake_head: Position
    snake_body: Tuple[Position, ...]
    direction: Direction
    food_position: Position
    score: int
    move_count: int
    grid_size: Tuple[int, int]
    seed: int
    episode_id: int
    STATE_LENGTH_BITS = DNetDef.STATE_LENGTH_BITS  # int(7)

    def clone(self) -> "GameBoard":
        # With immutability, "clone" is basically identity, but keep it explicit.
        return GameBoard(
            snake_head=self.snake_head,
            snake_body=self.snake_body,
            direction=self.direction,
            food_position=self.food_position,
            score=self.score,
            move_count=self.move_count,
            grid_size=self.grid_size,
            seed=self.seed,
            episode_id=self.episode_id,
        )

    def get_all_snake_positions(self) -> List[Position]:
        return [self.snake_head] + list(self.snake_body)

    def get_direction(self) -> Direction:
        return self.direction

    def get_food_position(self) -> Position:
        return self.food_position

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size

    def get_move_count(self) -> int:
        return self.move_count

    def get_snake_head(self) -> Position:
        return self.snake_head

    def get_state(self) -> list[float]:
        head = self.snake_head
        width, height = self.grid_size

        # Adjacent absolute positions
        point_l = Position(head.x - 1, head.y)
        point_r = Position(head.x + 1, head.y)
        point_u = Position(head.x, head.y - 1)
        point_d = Position(head.x, head.y + 1)

        # Direction flag (Direction is a dx/dy vector)
        dir_l = (self.direction.dx, self.direction.dy) == (-1, 0)
        dir_r = (self.direction.dx, self.direction.dy) == (1, 0)
        dir_u = (self.direction.dx, self.direction.dy) == (0, -1)
        dir_d = (self.direction.dx, self.direction.dy) == (0, 1)

        def danger_at(p: Position) -> bool:
            return self._is_wall_collision(p) or self._is_snake_collision(p)

        # Relative danger probes (straight, right, left)
        if dir_r:
            danger_straight = danger_at(point_r)
            danger_right = danger_at(point_d)
            danger_left = danger_at(point_u)
        elif dir_l:
            danger_straight = danger_at(point_l)
            danger_right = danger_at(point_u)
            danger_left = danger_at(point_d)
        elif dir_u:
            danger_straight = danger_at(point_u)
            danger_right = danger_at(point_r)
            danger_left = danger_at(point_l)
        else:  # dir_d
            danger_straight = danger_at(point_d)
            danger_right = danger_at(point_l)
            danger_left = danger_at(point_r)

        # Food relative direction (normalized -[-1, 1])
        dx = self.food_position.x - head.x
        dy = self.food_position.y - head.y
        food_dx = dx / max(1, width)
        food_dy = dy / max(1, height)

        # Length bits (use snake length, not just body length)
        length_bits = self._int_to_bits(
            self.STATE_LENGTH_BITS, self.get_snake_length()
        )

        state = [
            # 1 - 3 danger (relative)
            danger_left,
            danger_straight,
            danger_right,
            # 4 - 7 direction, one-hot
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # 8 - 9 food delta (normalized
            food_dx,
            food_dy,
            # 10.. length bits
            *length_bits,
        ]
        # Make sure size of the state map matches what we use in the NN
        assert len(state) == DNetDef.INPUT_SIZE
        return [float(x) for x in state]

    def get_score(self) -> int:
        return self.score

    def get_snake_body(self) -> List[Position]:
        return list(self.snake_body)

    def get_snake_length(self) -> int:
        return 1 + len(self.snake_body)

    @staticmethod
    def _int_to_bits(n_bits: int, value: int) -> list[int]:
        """
        Encode `value` as big-endian bits, fixed width.
        Example: n_bits=4, value=3 -> [0, 0, 1, 1]
        """
        value = max(0, int(value))
        out = []
        for i in range(n_bits - 1, -1, -1):
            out.append((value >> i) & 1)
        return out

    def is_position_occupied_by_snake(self, position: Position) -> bool:
        return position in self.get_all_snake_positions()

    def is_position_within_bounds(self, position: Position) -> bool:
        width, height = self.grid_size
        return 0 <= position.x < width and 0 <= position.y < height

    def _is_snake_collision(self, position: Position) -> bool:
        return self.is_position_occupied_by_snake(position)

    def _is_wall_collision(self, position: Position) -> bool:
        return not self.is_position_within_bounds(position)

    def to_dict(self) -> dict:
        return {
            DGameField.SNAKE_HEAD: {
                DGameField.X: self.snake_head.x,
                DGameField.Y: self.snake_head.y,
            },
            DGameField.SNAKE_BODY: [
                {DGameField.X: p.x, DGameField.Y: p.y} for p in self.snake_body
            ],
            DGameField.DIRECTION: {
                DGameField.DX: self.direction.dx,
                DGameField.DY: self.direction.dy,
            },
            DGameField.FOOD_POSITION: {
                DGameField.X: self.food_position.x,
                DGameField.Y: self.food_position.y,
            },
            DGameField.SCORE: self.score,
            DGameField.MOVE_COUNT: self.move_count,
            DGameField.GRID_SIZE: {
                DGameField.W: self.grid_size[0],
                DGameField.H: self.grid_size[1],
            },
            DGameField.SEED: self.seed,
            DGameField.EPISODE_ID: self.episode_id,
        }

    @classmethod
    def new_game(
        cls,
        *,
        rng: random.Random,
        seed: Optional[int] = None,
        episode_id: Optional[int] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> "GameBoard":
        # Seed is primarily metadata now; rng is authoritative.
        if seed is None:
            seed = DHydra.RANDOM_SEED

        if grid_size is None:
            grid_size = (DGameDef.BOARD_WIDTH, DGameDef.BOARD_HEIGHT)

        if episode_id is None:
            episode_id = rng.getrandbits(32)

        width, height = grid_size
        head = Position(width // 2, height // 2)
        direction = Direction.right()

        body = tuple(
            Position(head.x - i, head.y)
            for i in range(1, DGameDef.INITIAL_SNAKE_LENGTH)
        )
        occupied = {head, *body}
        food = _random_unoccupied_position(rng, grid_size, occupied)

        return cls(
            snake_head=head,
            snake_body=body,
            direction=direction,
            food_position=food,
            score=0,
            move_count=0,
            grid_size=grid_size,
            seed=int(seed),
            episode_id=int(episode_id),
        )
