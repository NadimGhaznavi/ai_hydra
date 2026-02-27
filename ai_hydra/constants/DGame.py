# ai_hydra/constants/DGame.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0


from typing import Final


class DGameDef:
    BOARD_HEIGHT: Final[int] = 20
    BOARD_WIDTH: Final[int] = 20
    INITIAL_SNAKE_LENGTH: Final[int] = 3
    # MOVE_BUDGET: Final[int] = 100
    MAX_MOVES_MULTIPLIER: Final[int] = 100
    FOOD_REWARD: Final[int] = 10
    COLLISION_PENALTY: Final[int] = -10
    EMPTY_MOVE_REWARD: Final[int] = 0


class DGameField:
    BOARD: Final[str] = "board"
    DONE: Final[str] = "done"
    DIRECTION: Final[str] = "direction"
    DX: Final[str] = "dx"
    DY: Final[str] = "dy"
    EMPTY: Final[str] = "empty"
    ERROR: Final[str] = "error"
    EPISODE_DONE: Final[str] = "episode_done"
    EPISODE_ID: Final[str] = "episode_id"
    FOOD: Final[str] = "food"
    FOOD_POSITION: Final[str] = "food_position"
    GRID_SIZE: Final[str] = "grid_size"
    H: Final[str] = "h"
    INFO: Final[str] = "info"
    LEFT_TURN: Final[str] = "left_turn"
    MAX_MOVES: Final[str] = "max_moves"
    MOVE_COUNT: Final[str] = "move_count"
    OK: Final[str] = "ok"
    REASON: Final[str] = "reason"
    REWARD: Final[str] = "reward"
    RIGHT_TURN: Final[str] = "right_turn"
    SCORE: Final[str] = "score"
    SCORE_DELTA: Final[str] = "score_delta"
    SEED: Final[str] = "seed"
    SNAKE: Final[str] = "snake"
    SNAKE_BODY: Final[str] = "snake_body"
    SNAKE_HEAD: Final[str] = "snake_head"
    SNAPSHOT: Final[str] = "snapshot"
    STEP_N: Final[str] = "step_n"
    STRAIGHT: Final[str] = "straight"
    WALL: Final[str] = "wall"
    W: Final[str] = "w"
    X: Final[str] = "x"
    Y: Final[str] = "y"


class DGameLabel:
    NO_FREE_POSITIONS: Final[str] = "No free positions available for food"


class DGameMethod:
    RESET: Final[str] = "reset_game"
    STEP: Final[str] = "game_step"
