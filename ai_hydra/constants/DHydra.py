# ai_hydra/constants/DHydraTui.py
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


class DHydra:
    """
    Global project constants.
    """

    HEARTBEAT_INTERVAL: Final[float] = 5.0
    NETWORK_TIMEOUT: Final[float] = 2.0
    PROTOCOL_VERSION: Final[int] = 1
    RANDOM_SEED: Final[int] = 1970
    VERSION: Final[str] = "0.8.0"


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


class DMethod(StrEnum):
    """
    Method identifier constants for AI Hydra methods.

    Provides standardized string identifiers for different AI Hydra
    modules, used in logging and component identification.
    """

    ACTION_LEFT = "action_left"
    ACTION_STRAIGHT = "action_straight"
    ACTION_RIGHT = "action_right"
    GAME_STEP = "game_step"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_REPLY = "heartbeat_reply"
    PING = "ping"
    PING_ROUTER = "ping_router"
    PING_SERVER = "ping_server"
    PONG = "pong"
    QUIT = "quit"
    RESET_GAME = "reset_game"
    START = "start"
    STOP = "stop"


class DModule(StrEnum):
    """
    Module identifier constants for AI Hydra components.

    Provides standardized string identifiers for different AI Hydra
    modules, used in logging and component identification.
    """

    HYDRA_CLIENT = "HydraClient"
    HYDRA_MGR = "HydraMgr"
    HYDRA_MQ = "HydraMQ"
    HYDRA_ROUTER = "HydraRouter"
    HYDRA_SERVER = "HydraServer"


LOG_LEVELS: Mapping[DHydraLog, int] = {
    DHydraLog.INFO: logging.INFO,
    DHydraLog.DEBUG: logging.DEBUG,
    DHydraLog.WARNING: logging.WARNING,
    DHydraLog.ERROR: logging.ERROR,
    DHydraLog.CRITICAL: logging.CRITICAL,
}
