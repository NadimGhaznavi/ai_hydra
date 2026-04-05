# ai_hydra/constants/DHydra.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import logging
from enum import StrEnum
from typing import Final, Mapping


# This should be DHydraDef....
class DHydra:
    """
    Global project constants.
    """

    HEARTBEAT_INTERVAL: Final[float] = 5.0
    NETWORK_TIMEOUT: Final[float] = 2.0
    PROTOCOL_VERSION: Final[int] = 1
    RANDOM_SEED: Final[int] = 1970
    VERSION: Final[str] = "0.26.0"
    HYDRA_DIR: Final[str] = "AI-Hydra"


class DHydraLog(StrEnum):
    """
    Logging level constants for HydraLog configuration.

    Defines string constants for different logging levels that map
    to Python's standard logging levels via the LOG_LEVELS dictionary.
    """

    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DHydraLogDef:
    """
    Hydra Log defaults.
    """

    DEFAULT_LOG_LEVEL: Final[DHydraLog] = DHydraLog.DEBUG


class DHydraMsg(StrEnum):
    """
    Attribute definitions for HydraMsg class messages.
    """

    METHOD = "method"
    SENDER = "sender"
    TARGET = "target"
    PAYLOAD = "payload"
    PROTOCOL_VERSION = "protocol_version"


class DHydraRouterDef:
    """
    Hydra Router defaults.
    """

    HOSTNAME: Final[str] = "localhost"
    PORT: Final[int] = 5757
    HEARTBEAT_PORT: Final[int] = 5758


class DHydraServerDef:
    """
    Hydra Server defaults.
    """

    HOSTNAME: Final[str] = "localhost"
    PORT: Final[int] = 5759
    PUB_PORT: Final[int] = 5760


class DMethod(StrEnum):
    """
    Method identifier constants for AI Hydra methods.

    Provides standardized string identifiers for different AI Hydra
    modules, used in logging and component identification.
    """

    ACTION_LEFT = "action_left"
    ACTION_STRAIGHT = "action_straight"
    ACTION_RIGHT = "action_right"
    COUNTER = "counter"
    GAME_STEP = "game_step"
    HANDSHAKE = "handshake"
    HANDSHAKE_REPLY = "handshake_reply"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_REPLY = "heartbeat_reply"
    PER_EP_BATCH = "per_ep_batch"
    PER_STEP_BATCH = "per_step_batch"
    PING = "ping"
    PING_ROUTER = "ping_router"
    PING_SERVER = "ping_server"
    PONG = "pong"
    QUIT = "quit"
    SCORES_BATCH = "scores_batch"
    START = "start"
    STOP = "stop"


class DModule(StrEnum):
    """
    Module identifier constants for AI Hydra components.

    Provides standardized string identifiers for different AI Hydra
    modules, used in logging and component identification.
    """

    ATH_REPLAY_MEMORY = "ATHReplayMemory"
    ATH_MEMORY = "ATHMemory"
    ATH_DATA_MGR = "ATHDataMgr"
    ATH_DATA_STORE = "ATHDataStore"
    ATH_GEARBOX = "ATHGearBox"
    EPSILON_ALGO = "EpsilonAlgo"
    EPSILON_NICE_ALGO = "EpsilonNiceAlgo"
    EPSILON_NICE_POLICY = "EpsilonNicePolicy"
    GRU_MODEL = "GRUModel"
    HYDRA_CLIENT = "HydraClient"
    HYDRA_MGR = "HydraMgr"
    HYDRA_MQ = "HydraMQ"
    HYDRA_ROUTER = "HydraRouter"
    HYDRA_RNG = "HydraRng"
    HYDRA_SERVER = "HydraServer"
    HYDRA_SERVER_MQ = "HydraServerMQ"
    LINEAR_MODEL = "LinearModel"
    LINEAR_TRAINER = "LinearTrainer"
    RNN_MODEL = "RNNModel"
    RECURRENT_TRAINER = "RecurrentTrainer"
    SIMPLE_REPLAY_MEMORY = "SimpleReplayMemory"
    TRAIN_MGR = "TrainMgr"


LOG_LEVELS: Mapping[DHydraLog, int] = {
    DHydraLog.INFO: logging.INFO,
    DHydraLog.DEBUG: logging.DEBUG,
    DHydraLog.WARNING: logging.WARNING,
    DHydraLog.ERROR: logging.ERROR,
    DHydraLog.CRITICAL: logging.CRITICAL,
}
