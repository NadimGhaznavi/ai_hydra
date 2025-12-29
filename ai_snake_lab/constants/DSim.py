"""
constants/DSim.py

    AI Snake Lab
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai_snake_lab
    Website: https://snakelab.osoyalce.com
    License: GPL 3.0
"""

from ai_snake_lab.utils.ConstGroup import ConstGroup


class DSim(ConstGroup):
    """Simulation Constants"""

    # Size of the game board
    BOARD_SIZE: int = 20

    # Random, random seed to make simulation runs repeatable
    RANDOM_SEED: int = 1970

    # Size of the statemap, this is from the GameBoard class
    STATE_SIZE: int = 19

    # The number of "choices" the snake has: go forward, left or right.
    OUTPUT_SIZE: int = 3

    # The discount (gamma) default
    DISCOUNT_RATE: float = 0.93

    # Training loop states
    PAUSED: str = "paused"
    RUNNING: str = "running"
    STOPPED: str = "stopped"

    # Stats dictionary keys
    GAME_SCORE: str = "game_score"
    GAME_NUM: str = "game_num"

    ## Distributed settings
    # Server protocol and port
    HOST: str = "0.0.0.0"
    PROTOCOL: str = "tcp"
    MQ_PORT: int = 5555

    # Heartbeat interval, how often clients send "I'm alive" messages to the router.
    # In seconds.
    HEARTBEAT_INTERVAL: int = 5

    # Batch size used in the AIAgent:train_long_memory
    BATCH_SIZE: int = 32
