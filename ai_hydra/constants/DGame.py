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
    EMPTY_MOVE_REWARD: Final[int] = 0.0
    CLOSER_TO_FOOD: Final[float] = 0.0
    FURTHER_FROM_FOOD: Final[float] = -0.0


class DGameField:
    ACTION: Final[str] = "action"
    BOARD: Final[str] = "board"
    CLOSER_TO_FOOD: Final[str] = "closer_to_food"
    CUR_SCORE: Final[str] = "cur_score"
    CUR_EPOCH: Final[str] = "cur_epoch"
    DIRECTION: Final[str] = "direction"
    DONE: Final[str] = "done"
    DX: Final[str] = "dx"
    DY: Final[str] = "dy"
    EMPTY: Final[str] = "empty"
    ERROR: Final[str] = "error"
    ELAPSED_TIME: Final[str] = "elapsed_time"
    EPISODE_DONE: Final[str] = "episode_done"
    EPISODE_ID: Final[str] = "episode_id"
    EPOCH: Final[str] = "epoch"
    FOOD: Final[str] = "food"
    FOOD_POSITION: Final[str] = "food_position"
    FURTHER_FROM_FOOD: Final[str] = "further_from_food"
    GRID_SIZE: Final[str] = "grid_size"
    H: Final[str] = "h"
    HIGHSCORE: Final[str] = "highscore"
    HIGHSCORE_NLH: Final[str] = "highscore_nlh"
    HIGHSCORE_LH: Final[str] = "highscore_lh"
    HIGHSCORE_EVENT: Final[str] = "highscore_event"
    HIGHSCORE_EVENT_LH: Final[str] = "highscore_event_lh"
    HIGHSCORE_EVENT_NLH: Final[str] = "highscore_event_nlh"
    INFO: Final[str] = "info"
    INVALID_ACTION: Final[str] = "invalid_action"
    LEFT_TURN: Final[str] = "left_turn"
    MAX_MOVES: Final[str] = "max_moves"
    MISSING_ACTION: Final[str] = "missing_action"
    MOVE_COUNT: Final[str] = "move_count"
    OK: Final[str] = "ok"
    PUB_TYPE: Final[str] = "pub_type"
    REASON: Final[str] = "reason"
    REWARD: Final[str] = "reward"
    REWARD_TOTAL: Final[str] = "reward_total"
    RIGHT_TURN: Final[str] = "right_turn"
    SCORE: Final[str] = "score"
    SCORE_DELTA: Final[str] = "score_delta"
    SEED: Final[str] = "seed"
    SESSION_ID: Final[str] = "session_id"
    SIM_PAUSED: Final[str] = "sim_paused"
    SIM_RESUMED: Final[str] = "sim_resumed"
    SIM_RUNNING: Final[str] = "sim_running"
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
    PAUSE_RUN: Final[str] = "pause_run"
    PAUSE_RUN_REPLY: Final[str] = "pause_run_reply"
    RESET_RUN: Final[str] = "reset_run"
    RESET_RUN_REPLY: Final[str] = "reset_run_reply"
    RESUME_RUN: Final[str] = "resume_run"
    RESUME_RUN_REPLY: Final[str] = "resume_run_reply"
    START_RUN: Final[str] = "start_run"
    STOP_RUN: Final[str] = "stop_run"
    UPDATE_CONFIG: Final[str] = "update_config"
