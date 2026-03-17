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
    PURPLE: Final[str] = "purple"
    RED: Final[str] = "red"
    WHITE: Final[str] = "white"
    YELLOW: Final[str] = "yellow"


class DField:
    """
    Machine readable field names.
    """

    ALL: Final[str] = "all"
    BAD_HANDSHAKE: Final[str] = "bad_handshake"
    BATCH_SIZE_INPUT: Final[str] = "batch_size_input"
    BATCH_SIZE_LABEL: Final[str] = "batch_size_label"
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
    COUNT: Final[str] = "count"
    CUR_EPSILON: Final[str] = "cur_epsilon"
    CUR_LOSS: Final[str] = "cur_loss"
    DROPOUT_P: Final[str] = "dropout_p"
    DROPOUT_P_INPUT: Final[str] = "dropout_p_input"
    DROPOUT_P_LABEL: Final[str] = "dropout_p_label"
    EPSILON: Final[str] = "epsilon"
    EPSILON_DECAY: Final[str] = "epsilon_decay"
    EPSILON_DECAY_INPUT: Final[str] = "epsilon_decay_input"
    EPSILON_DECAY_LABEL: Final[str] = "epsilon_decay_label"
    EPSILON_DEPLETED: Final[str] = "epsilon_depleted"
    GAMMA_INPUT: Final[str] = "gamma_input"
    GAMMA_LABEL: Final[str] = "gamma_label"
    GAME_SCORE_PLOT: Final[str] = "game_score_plot"
    GAME_SCORES: Final[str] = "game_scores"
    HANDSHAKE: Final[str] = "handshake"
    HIDDEN_SIZE_INPUT: Final[str] = "hidden_size_input"
    HIDDEN_SIZE_LABEL: Final[str] = "hidden_size_label"
    HIDDEN_WIDGET: Final[str] = "hidden_widget"
    HIGHSCORES: Final[str] = "highscores"
    HIGHSCORES_BOX: Final[str] = "highscores_box"
    HIGHSCORES_LOG: Final[str] = "highscores_log"
    HIGHSCORES_PLOT: Final[str] = "highscores_plot"
    HIGHSCORES_LH: Final[str] = "highscores_lh"
    HYDRA_TELEMETRY: Final[str] = "hydra_telemetry"
    INITIAL_EPSILON: Final[str] = "initial_epsilon"
    INITIAL_EPSILON_INPUT: Final[str] = "initial_epsilon_input"
    INITIAL_EPSILON_LABEL: Final[str] = "initial_epsilon_label"
    INPUT_FIELD: Final[str] = "input_field"
    INPUT_SIZE: Final[str] = "input_size"
    INTEGER: Final[str] = "integer"
    LEARNING_RATE_INPUT: Final[str] = "learning_rate_input"
    LEARNING_RATE_LABEL: Final[str] = "learning_rate_label"
    LINEAR: Final[str] = "linear"
    LOOKAHEAD_BOX: Final[str] = "lookahead_box"
    LOOKAHEAD_STATUS: Final[str] = "lookahead_status"
    LOOKAHEAD_P_VAL_INPUT: Final[str] = "lookahead_p_val_input"
    LOOKAHEAD_P_VAL_LABEL: Final[str] = "lookahead_p_val_label"
    LOOKAHEAD_SAMPLE_P_VALUE_LABEL: Final[str] = "lookahead_sample_p_val_label"
    LOSS: Final[str] = "loss"
    LOSS_PLOT: Final[str] = "loss_plot"
    MIN_EPSILON: Final[str] = "min_epsilon"
    MIN_EPSILON_INPUT: Final[str] = "min_epsilon_input"
    MIN_EPSILON_LABEL: Final[str] = "min_epsilon_label"
    MODEL: Final[str] = "model"
    MODEL_TYPE_LABEL: Final[str] = "model_type_label"
    MODEL_TYPE_SELECT: Final[str] = "model_type_select"
    MOVE_DELAY_LABEL: Final[str] = "move_delay_label"
    MOVE_DELAY_INPUT: Final[str] = "move_delay_input"
    NETWORK: Final[str] = "network"
    NORMAL: Final[str] = "normal"
    NUMBER: Final[str] = "number"
    PLOT_CUR_SCORE: Final[str] = "plot_cur_score"
    PLOT_HIGHSCORES: Final[str] = "plot_highscores"
    PLOT_LOSS: Final[str] = "plot_loss"
    PLOT_RECENT_LOSS: Final[str] = "plot_recent_loss"
    PLOT_RECENT_SCORES_DIST: Final[str] = "plot_recent_scores_dist"
    PLOT_SCORES_DIST: Final[str] = "plot_scores_dist"
    QUIT: Final[str] = "quit"
    RECENT: Final[str] = "recent"
    RANDOM_SEED_INPUT: Final[str] = "random_seed_input"
    RANDOM_SEED_LABEL: Final[str] = "random_seed_label"
    RESET: Final[str] = "reset"
    RNN: Final[str] = "rnn"
    RNN_TAU_INPUT: Final[str] = "rnn_tau_input"
    RNN_TAU_LABEL: Final[str] = "rnn_tau_label"
    RNN_TAU_OPT: Final[str] = "rnn_tau_opt"
    RNN_LAYERS_INPUT: Final[str] = "rnn_layers_input"
    RNN_LAYERS_LABEL: Final[str] = "rnn_layers_label"
    RNN_LAYERS_OPT: Final[str] = "rnn_layers_opt"
    RNN_DROPOUT: Final[str] = "rnn_dropout"
    RNN_LAYERS: Final[str] = "rnn_layers"
    ROUTER_HB: Final[str] = "router_hb"
    RUNNING: Final[str] = "running"
    SCORES_DIST_PLOT: Final[str] = "scores_list_plot"
    SETTINGS: Final[str] = "settings"
    SEQ_LENGTH_INPUT: Final[str] = "seq_length_input"
    SEQ_LENGTH_LABEL: Final[str] = "seq_length_label"
    SEQ_LENGTH_OPT: Final[str] = "seq_length_opt"
    SCORES_PLOT: Final[str] = "scores_plot"
    SCORES_PLOT_LH: Final[str] = "scores_plot_lh"
    SCORES_PLOT_NLH: Final[str] = "scores_plot_nlh"
    SCORES_SCATTER_PLOT: Final[str] = "scores_scatter_plot"
    SIM_RUNNING: Final[str] = "sim_running"
    SIM_STOPPED: Final[str] = "sim_stopped"
    SNAPSHOT: Final[str] = "snapshot"
    STATUS: Final[str] = "status"
    START: Final[str] = "start"
    START_RUN: Final[str] = "start_run"
    STOP: Final[str] = "stop"
    STOPPED: Final[str] = "stopped"
    TABBED_PLOTS: Final[str] = "tabbed_plots"
    TITLE: Final[str] = "title"
    TRAINING_BOX: Final[str] = "training_box"
    TURBO_MODE: Final[str] = "turbo_mode"
    TURBO_OFF: Final[str] = "turbo_off"
    TURBO_ON: Final[str] = "turbo_on"
    UPDATE_RUNTIME_CONFIG: Final[str] = "update_runtime_config"
    UPDATE_CONFIG_SPACER: Final[str] = "update_config_spacer"


class DFile:
    """
    Filenames.
    """

    CLIENT_CSS: Final[str] = "HydraClient.tcss"
    ROUTER_CSS: Final[str] = "HydraRouter.tcss"
    HYDRA_SERVER_DB: Final[str] = "HydraServer.db"
    BASE_SNAPSHOT: Final[str] = "AI-Hydra-Snapshot_"


class DLabel:
    """
    Human readable text.
    """

    ACTIONS: Final[str] = "Actions"
    AVERAGE: Final[str] = "Average"
    BATCH_SIZE: Final[str] = "Batch Size"
    BOTH: Final[str] = "Both"
    CLIENT_TITLE: Final[str] = "Hydra Client"
    CLIENTS: Final[str] = "Clients"
    COMPLETE: Final[str] = "Complete"
    CONFIG: Final[str] = "Configuration"
    CONSOLE: Final[str] = "Console"
    CUR_LOSS: Final[str] = "Current Loss"
    CUR_EPSILON: Final[str] = "Current Epsilon"
    CURRENT: Final[str] = "Current"
    CUR_SCORES: Final[str] = "Current Scores"
    COUNT: Final[str] = "Count"
    DEBUG: Final[str] = "DEBUG"
    DISCONNECTED: Final[str] = "Disonnected"
    DROPOUT_P_VAL: Final[str] = "Dropout P-Value"
    EPISODE: Final[str] = "Episode"
    EPISODES: Final[str] = "Episodes"
    EPSILON: Final[str] = "Epsilon"
    EPSILON_DECAY: Final[str] = "Epsilon Decay"
    ERROR: Final[str] = "ERROR"
    EVENT_TIME: Final[str] = "Event"
    GAME: Final[str] = "Game"
    GAME_SCORES: Final[str] = "Game Scores"
    GAMMA: Final[str] = "Discount/Gamma"
    HANDSHAKE: Final[str] = "Handshake"
    HIDDEN_SIZE: Final[str] = "Hidden Size"
    HIGHSCORE: Final[str] = "Highscore"
    HIGHSCORES: Final[str] = "Highscores"
    HIGHSCORES_LH: Final[str] = "Look Ahead"
    INITIAL_EPSILON: Final[str] = "Initial Epsilon"
    LEFT: Final[str] = "Left"
    LH: Final[str] = "LA"
    LEARNING_RATE: Final[str] = "Learning Rate"
    LINEAR: Final[str] = "Linear"
    LISTEN_PORT: Final[str] = "Listening Port"
    LOOKAHEAD: Final[str] = "Look Ahead"
    LOOKAHEAD_SETTINGS: Final[str] = "Look Ahead Settings"
    LOSS: Final[str] = "Loss"
    MEAN: Final[str] = "Mean"
    MEDIAN: Final[str] = "Median"
    MIN_EPSILON: Final[str] = "Minimum Epsilon"
    MODEL: Final[str] = "Model"
    MOVE_DELAY: Final[str] = "Move Delay"
    NETWORK: Final[str] = "Network"
    NN_MODEL: Final[str] = "NN Model"
    NORMAL: Final[str] = "Normal"
    P_VALUE: Final[str] = "P-Value"
    PING_ROUTER: Final[str] = "Ping Router"
    PING_SERVER: Final[str] = "Ping Server"
    QUIT: Final[str] = "Quit"
    RANDOM_SEED: Final[str] = "Random Seed"
    RECENT_COUNT: Final[str] = "Recent Scores"
    RECENT_LOSS: Final[str] = "Recent Loss"
    RESET: Final[str] = "Reset"
    RIGHT: Final[str] = "Right"
    RNN: Final[str] = "RNN"
    RNN_LAYERS: Final[str] = "RNN Layers"
    RNN_TAU: Final[str] = "RNN Tau"
    ROUTER: Final[str] = "Router"
    ROUTER_TITLE: Final[str] = "Hydra Router"
    SETTINGS: Final[str] = "Settings"
    SAMPLING_P_VALUE: Final[str] = "Sampling P-Value"
    SCORE: Final[str] = "Score"
    SCORE_DISTRIBUTION: Final[str] = "Score Distribution"
    SCORES: Final[str] = "Scores"
    SCORES_ALL: Final[str] = "Scores/All"
    SCORES_LH: Final[str] = "Scores/L.A."
    SCORES_NLH: Final[str] = "Scores/No L.A."
    SCORES_SCATTER: Final[str] = "Scores Scatterplot"
    SEQUENCE_LENGTH: Final[str] = "Sequence Length"
    SNAPSHOT: Final[str] = "Snapshot"
    SPACE: Final[str] = " "
    START: Final[str] = "Start"
    STATUS: Final[str] = "Status"
    STEPS: Final[str] = "Steps"
    STOP: Final[str] = "Stop"
    STATUS: Final[str] = "Status"
    STRAIGHT: Final[str] = "Straight"
    TARGET_HOST: Final[str] = "Host"
    TARGET_PORT: Final[str] = "Port"
    TIME: Final[str] = "Time"
    TRAINING: Final[str] = "Training"
    TURBO_MODE: Final[str] = "Turbo Mode"
    UPDATE_CONFIG: Final[str] = "Update Config"
    VERSION: Final[str] = "Version"
    VISUALIZATIONS: Final[str] = "Visualizations"


class DPlotDef:
    """
    Default Textual plot settings.
    """

    # The average is calculated by dividing the MAX_DATA_POINTS by this number
    AVG_DIVISOR: int = 50

    MAX_LOSS_DATA_POINTS: Final[int] = 125
    MAX_CUR_DATA_POINTS: Final[int] = 200
    MAX_CUR_LOSS_DATA_POINTS: Final[int] = 75

    # GameScorePlot widget settings
    MAX_GAMES = 75
    MAX_GAMES_AVG_DIVISOR: int = 50

    # Maximum number of scores stored in the metrics object
    MAX_CUR_SCORES = 75

    # The number of "recent" games in the score distribution histogram
    RECENT_SCORES_MAX = 500

    # The number of "recent" loss data points in the "Recent Loss" plot
    RECENT_LOSS_MAX = 75


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
