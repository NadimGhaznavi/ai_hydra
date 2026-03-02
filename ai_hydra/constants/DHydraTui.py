# ai_hydra/constants/DHydraTui.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final


class DField:
    """
    Machine readable field names.
    """

    BOARD_BOX: Final[str] = "board_box"
    BOTH: Final[str] = "both"
    BUTTONS: Final[str] = "buttons"
    CLIENTS: Final[str] = "clients"
    CLIENTS_SCREEN: Final[str] = "clients_screen"
    CONNECTED: Final[str] = "connected"
    CONFIG: Final[str] = "config"
    CONSOLE: Final[str] = "console"
    CONSOLE_SCREEN: Final[str] = "console_screen"
    CUR_EPSILON: Final[str] = "cur_epsilon"
    HIGHSCORES: Final[str] = "highscores"
    HIGHSCORES_BOX: Final[str] = "highscores_box"
    HIGHSCORES_LH: Final[str] = "highscores_lh"
    LOOKAHEAD_ENABLED: Final[str] = "lookahead_enabled"
    QUIT: Final[str] = "quit"
    RESET: Final[str] = "reset"
    RUNNING: Final[str] = "running"
    RUNTIME_VALUES: Final[str] = "runtime_values"
    STATUS: Final[str] = "status"
    START: Final[str] = "start"
    STOP: Final[str] = "stop"
    TABBED_SCORES: Final[str] = "tabbed_scores"
    TITLE: Final[str] = "title"


class DFile:
    """
    Filenames.
    """

    CLIENT_CSS_PATH: Final[str] = "HydraClient.tcss"
    ROUTER_CSS_PATH: Final[str] = "HydraRouter.tcss"


class DLabel:
    """
    Human readable text.
    """

    BOTH: Final[str] = "Both"
    CLIENT_TITLE: Final[str] = "Hydra Client"
    CLIENTS: Final[str] = "Clients"
    CONFIG: Final[str] = "Configuration"
    CONNECTED: Final[str] = "Connected"
    CUR_EPSILON: Final[str] = "Current Epsilon"
    DEBUG: Final[str] = "DEBUG"
    DISCONNECTED: Final[str] = "Disonnected"
    EPSILON_DECAY: Final[str] = "Epsilon Decay"
    ERROR: Final[str] = "ERROR"
    GAME: Final[str] = "Game"
    HIGHSCORE: Final[str] = "Highscore"
    HIGHSCORES: Final[str] = "Highscores"
    HIGHSCORES_LH: Final[str] = "Look Ahead"
    INITIAL_EPSILON: Final[str] = "Initial Epsilon"
    LEFT: Final[str] = "Left"
    LH: Final[str] = "LA"
    LISTEN_PORT: Final[str] = "Listening Port"
    LOOKAHEAD: Final[str] = "Look Ahead"
    LOOKAHEAD_ENABLED: Final[str] = "Look Ahead Enabled"
    LOOKAHEAD_P_VAL: Final[str] = "Look Ahead P-Value"
    MIN_EPSILON: Final[str] = "Minimum Epsilon"
    PING_ROUTER: Final[str] = "Ping Router"
    PING_SERVER: Final[str] = "Ping Server"
    QUIT: Final[str] = "Quit"
    RESET: Final[str] = "Reset"
    RIGHT: Final[str] = "Right"
    ROUTER_TITLE: Final[str] = "Hydra Router"
    RUNTIME_VALS: Final[str] = "Runtime Values"
    SCORE: Final[str] = "Score"
    SPACE: Final[str] = " "
    START: Final[str] = "Start"
    STEPS: Final[str] = "Steps"
    STOP: Final[str] = "Stop"
    STATUS: Final[str] = "Status"
    STRAIGHT: Final[str] = "Straight"
    TARGET_HOST: Final[str] = "Target Host"
    TARGET_PORT: Final[str] = "Target Port"
    TIME: Final[str] = "Time"
    VERSION: Final[str] = "Version"


class DStatus:
    """
    Status indicators.
    """

    GOOD: Final[str] = "🟢"
    OK: Final[str] = "🟡"
    BAD: Final[str] = "🔴"
