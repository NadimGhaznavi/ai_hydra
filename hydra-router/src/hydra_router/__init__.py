"""
Hydra Router - Standalone ZeroMQ-based message routing system.

A reusable message routing system for distributed applications with automatic
client discovery, heartbeat monitoring, and comprehensive error handling.
"""

from .router import HydraRouter
from .mq_client import MQClient
from .router_constants import RouterConstants
from .zmq_protocol import ZMQMessage, MessageType
from .exceptions import (
    HydraRouterError,
    MessageValidationError,
    ConnectionError,
    ClientRegistrationError,
    MessageFormatError,
    RouterConfigurationError,
    HeartbeatError,
    RoutingError,
)

__version__ = "0.1.0"
__author__ = "AI Hydra Team"
__email__ = "team@aihydra.dev"

__all__ = [
    # Core components
    "HydraRouter",
    "MQClient",
    "RouterConstants",
    # Message protocol
    "ZMQMessage",
    "MessageType",
    # Exceptions
    "HydraRouterError",
    "MessageValidationError",
    "ConnectionError",
    "ClientRegistrationError",
    "MessageFormatError",
    "RouterConfigurationError",
    "HeartbeatError",
    "RoutingError",
]
