# ai_hydra/game_board/GameBoard.py

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from ai_hydra.constants.DHydra import DHydra
from ai_hydra.constants.DGame import DGameDef, DGameLabel, DGameField


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

    def get_snake_head(self) -> Position:
        return self.snake_head

    def get_snake_body(self) -> List[Position]:
        return list(self.snake_body)

    def get_direction(self) -> Direction:
        return self.direction

    def get_food_position(self) -> Position:
        return self.food_position

    def get_score(self) -> int:
        return self.score

    def get_move_count(self) -> int:
        return self.move_count

    def get_snake_length(self) -> int:
        return 1 + len(self.snake_body)

    def get_grid_size(self) -> Tuple[int, int]:
        return self.grid_size

    def get_all_snake_positions(self) -> List[Position]:
        return [self.snake_head] + list(self.snake_body)

    def is_position_occupied_by_snake(self, position: Position) -> bool:
        return position in self.get_all_snake_positions()

    def is_position_within_bounds(self, position: Position) -> bool:
        width, height = self.grid_size
        return 0 <= position.x < width and 0 <= position.y < height

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
