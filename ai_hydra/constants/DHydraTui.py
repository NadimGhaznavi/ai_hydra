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

    BLACK: Final[str] = "black"
    BLUE: Final[str] = "blue"
    DARK_RED: Final[str] = "dark_red"
    GREEN: Final[str] = "green"
    RED: Final[str] = "red"
    WHITE: Final[str] = "white"
    YELLOW: Final[str] = "yellow"


class DField:
    """
    Machine readable field names.
    """

    BAD_HANDSHAKE: Final[str] = "bad_handshake"
    BOARD_BOX: Final[str] = "board_box"
    BOTH: Final[str] = "both"
    BUTTONS: Final[str] = "buttons"
    CLIENTS: Final[str] = "clients"
    CLIENTS_SCREEN: Final[str] = "clients_screen"
    CONFIG: Final[str] = "config"
    CONSOLE_BOX: Final[str] = "console_box"
    DISCONNECTED: Final[str] = "disconnected"
    CONSOLE: Final[str] = "console"
    CONSOLE_SCREEN: Final[str] = "console_screen"
    CUR_EPSILON: Final[str] = "cur_epsilon"
    EPSILON_DECAY_INPUT: Final[str] = "epsilon_decay_input"
    EPSILON_DECAY_LABEL: Final[str] = "epsilon_decay_label"
    HANDSHAKE: Final[str] = "handshake"
    ROUTER_HB: Final[str] = "router_hb"
    HIDDEN_WIDGET: Final[str] = "hidden_widget"
    HIGHSCORES: Final[str] = "highscores"
    HIGHSCORES_BOX: Final[str] = "highscores_box"
    HIGHSCORES_LH: Final[str] = "highscores_lh"
    INITIAL_EPSILON_INPUT: Final[str] = "initial_epsilon_input"
    INITIAL_EPSILON_LABEL: Final[str] = "initial_epsilon_label"
    INPUT_FIELD: Final[str] = "input_field"
    LOOKAHEAD_STATUS: Final[str] = "lookahead_status"
    LOOKAHEAD_P_VAL_INPUT: Final[str] = "lookahead_p_val_input"
    LOOKAHEAD_P_VAL_LABEL: Final[str] = "lookahead_p_val_label"
    LOSS_PLOT: Final[str] = "loss_plot"
    MIN_EPSILON_INPUT: Final[str] = "min_epsilon_input"
    MIN_EPSILON_LABEL: Final[str] = "min_epsilon_label"
    MOVE_DELAY_LABEL: Final[str] = "move_delay_label"
    MOVE_DELAY_INPUT: Final[str] = "move_delay_input"
    NETWORK: Final[str] = "network"
    NORMAL: Final[str] = "normal"
    NUMBER: Final[str] = "number"
    QUIT: Final[str] = "quit"
    RANDOM_SEED_LABEL: Final[str] = "random_seed_label"
    RESET: Final[str] = "reset"
    RUNNING: Final[str] = "running"
    SETTINGS: Final[str] = "settings"
    SCORES_PLOT: Final[str] = "scores_plot"
    SCORES_PLOT_LH: Final[str] = "scores_plot_lh"
    SCORES_PLOT_NLH: Final[str] = "scores_plot_nlh"
    SIM_RUNNING: Final[str] = "sim_running"
    SIM_STOPPED: Final[str] = "sim_stopped"
    STATUS: Final[str] = "status"
    START: Final[str] = "start"
    START_RUN: Final[str] = "start_run"
    STOP: Final[str] = "stop"
    STOPPED: Final[str] = "stopped"
    TABBED_PLOTS: Final[str] = "tabbed_plots"
    TABBED_SCORES: Final[str] = "tabbed_scores"
    TITLE: Final[str] = "title"
    TURBO_MODE: Final[str] = "turbo_mode"
    TURBO_OFF: Final[str] = "turbo_off"
    TURBO_ON: Final[str] = "turbo_on"
    UPDATE_RUNTIME_CONFIG: Final[str] = "update_runtime_config"
    UPDATE_CONFIG_SPACER: Final[str] = "update_config_spacer"


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
    BOTH: Final[str] = "Both"
    CLIENT_TITLE: Final[str] = "Hydra Client"
    CLIENTS: Final[str] = "Clients"
    CONFIG: Final[str] = "Configuration"
    CONSOLE: Final[str] = "Console"
    NETWORK: Final[str] = "Network"
    CUR_EPSILON: Final[str] = "Current Epsilon"
    DEBUG: Final[str] = "DEBUG"
    DISCONNECTED: Final[str] = "Disonnected"
    EPISODES: Final[str] = "Episodes"
    EPSILON_DECAY: Final[str] = "Epsilon Decay"
    ERROR: Final[str] = "ERROR"
    GAME: Final[str] = "Game"
    HANDSHAKE: Final[str] = "Handshake"
    HIGHSCORE: Final[str] = "Highscore"
    HIGHSCORES: Final[str] = "Highscores"
    HIGHSCORES_LH: Final[str] = "Look Ahead"
    INITIAL_EPSILON: Final[str] = "Initial Epsilon"
    LEFT: Final[str] = "Left"
    LH: Final[str] = "LA"
    LISTEN_PORT: Final[str] = "Listening Port"
    LOOKAHEAD: Final[str] = "Look Ahead"
    LOOKAHEAD_P_VAL: Final[str] = "Look Ahead P-Value"
    LOOKAHEAD_STATUS: Final[str] = "Look Ahead Status"
    LOSS: Final[str] = "Loss"
    MIN_EPSILON: Final[str] = "Minimum Epsilon"
    MOVE_DELAY: Final[str] = "Move Delay"
    NORMAL: Final[str] = "Normal"
    PING_ROUTER: Final[str] = "Ping Router"
    PING_SERVER: Final[str] = "Ping Server"
    QUIT: Final[str] = "Quit"
    RANDOM_SEED: Final[str] = "Random Seed"
    RESET: Final[str] = "Reset"
    RIGHT: Final[str] = "Right"
    ROUTER: Final[str] = "Router"
    ROUTER_TITLE: Final[str] = "Hydra Router"
    SETTINGS: Final[str] = "Settings and Runtime Values"
    SCORE: Final[str] = "Score"
    SCORES: Final[str] = "Scores"
    SCORES_ALL: Final[str] = "Scores/All"
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
    TURBO_MODE: Final[str] = "Turbo Mode"
    UPDATE_CONFIG: Final[str] = "Update Config"
    VERSION: Final[str] = "Version"
    VISUALIZATIONS: Final[str] = "Visualizations"


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
    ON: Final[str] = "🟢"
    OK: Final[str] = "🟡"
    BAD: Final[str] = "🔴"
    OFF: Final[str] = "🔴"
    UNKNOWN: Final[str] = "⚪"
