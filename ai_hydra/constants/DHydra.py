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
    RANDOM_SEED: Final[int] = 129
    VERSION: Final[str] = "0.15.0"
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


class DHydraMQ:
    HEARTBEAT: Final[str] = "heartbeat"
    MSGS: Final[str] = "msgs"
    PER_EP: Final[str] = "per_ep"
    PER_STEP: Final[str] = "per_step"
    SCORES: Final[str] = "scores"
    TIMER: Final[str] = "timer"
    UTF_8: Final[str] = "utf-8"


class DHydraMQDef:
    TOPIC_PREFIX: Final[str] = "ai_hydra"
    PER_STEP_TOPIC: Final[str] = "per_step_topic"
    PER_EPISODE_TOPIC: Final[str] = "per_episode_topic"
    SCORES_TOPIC: Final[str] = "scores_topic"
    MAX_BATCH_TIME: Final[float] = 0.5
    MAX_BATCH_SIZE: Final[int] = 100


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
