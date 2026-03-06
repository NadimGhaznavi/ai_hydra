# ai_hydra/constants/DHydraTui.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from typing import Final


class DColor:
    """Color constants"""

    GREEN: Final[str] = "green"
    RED: Final[str] = "red"
    BLUE: Final[str] = "blue"
    YELLOW: Final[str] = "yellow"
    WHITE: Final[str] = "white"
    BLACK: Final[str] = "black"


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
    NETWORK: Final[str] = "network"
    CONSOLE: Final[str] = "console"
    CONSOLE_SCREEN: Final[str] = "console_screen"
    CUR_EPSILON: Final[str] = "cur_epsilon"
    HIGHSCORES: Final[str] = "highscores"
    HIGHSCORES_BOX: Final[str] = "highscores_box"
    HIGHSCORES_LH: Final[str] = "highscores_lh"
    LOOKAHEAD_ENABLED: Final[str] = "lookahead_enabled"
    LOSS_PLOT: Final[str] = "loss_plot"
    NO_BOARD: Final[str] = "no_board"
    QUIT: Final[str] = "quit"
    RESET: Final[str] = "reset"
    RUNNING: Final[str] = "running"
    RUNTIME_VALUES: Final[str] = "runtime_values"
    SCORES_PLOT: Final[str] = "scores_plot"
    SCORES_PLOT_LH: Final[str] = "scores_plot_lh"
    SCORES_PLOT_NLH: Final[str] = "scores_plot_nlh"
    SHOW_BOARD: Final[str] = "show_board"
    STATUS: Final[str] = "status"
    START: Final[str] = "start"
    STOP: Final[str] = "stop"
    TABBED_PLOTS: Final[str] = "tabbed_plots"
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

    ACTIONS: Final[str] = "Actions"
    BOARD: Final[str] = "Normal"
    BOTH: Final[str] = "Both"
    CLIENT_TITLE: Final[str] = "Hydra Client"
    CLIENTS: Final[str] = "Clients"
    NETWORK: Final[str] = "Network"
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
    LOSS: Final[str] = "Loss"
    MIN_EPSILON: Final[str] = "Minimum Epsilon"
    NO_BOARD: Final[str] = "Turbo"
    PING_ROUTER: Final[str] = "Ping Router"
    PING_SERVER: Final[str] = "Ping Server"
    QUIT: Final[str] = "Quit"
    RESET: Final[str] = "Reset"
    RIGHT: Final[str] = "Right"
    ROUTER_TITLE: Final[str] = "Hydra Router"
    RUNTIME_VALS: Final[str] = "Runtime Values"
    SCORE: Final[str] = "Score"
    SCORES: Final[str] = "Scores/All"
    SCORES_LH: Final[str] = "Scores/L.A."
    SCORES_NLH: Final[str] = "Scores/No L.A."
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


class DPlotDef:
    """
    Default Textual plot settings.
    """

    MAX_LOSS_DATA_POINTS: Final[int] = 100


class DStatus:
    """
    Status indicators.
    """

    GOOD: Final[str] = "🟢"
    OK: Final[str] = "🟡"
    BAD: Final[str] = "🔴"
