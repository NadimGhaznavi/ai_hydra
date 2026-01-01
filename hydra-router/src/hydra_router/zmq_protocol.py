"""
ZeroMQ Message Protocol for Hydra Router.

This module defines the message protocol for communication between clients
and the Hydra Router system. It provides a standalone implementation without
dependencies on external game logic.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import json
import time


class MessageType(Enum):
    """Types of messages in the protocol."""

    # Commands (Client -> Server)
    START_SIMULATION = "start_simulation"
    STOP_SIMULATION = "stop_simulation"
    PAUSE_SIMULATION = "pause_simulation"
    RESUME_SIMULATION = "resume_simulation"
    GET_STATUS = "get_status"
    UPDATE_CONFIG = "update_config"
    RESET_SIMULATION = "reset_simulation"

    # Responses (Server -> Client)
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_STOPPED = "simulation_stopped"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_RESUMED = "simulation_resumed"
    STATUS_RESPONSE = "status_response"
    CONFIG_UPDATED = "config_updated"
    SIMULATION_RESET = "simulation_reset"

    # Broadcasts (Server -> All Clients)
    STATUS_UPDATE = "status_update"
    GAME_STATE_UPDATE = "game_state_update"
    DECISION_CYCLE_COMPLETE = "decision_cycle_complete"
    GAME_OVER = "game_over"
    ERROR_OCCURRED = "error_occurred"

    # System Messages
    CLIENT_CONNECTED = "client_connected"
    CLIENT_DISCONNECTED = "client_disconnected"
    HEARTBEAT = "heartbeat"


@dataclass
class ZMQMessage:
    """Base message structure for ZeroMQ communication."""

    message_type: MessageType
    timestamp: float
    client_id: Optional[str] = None
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert message to JSON string."""
        message_dict = {
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
            "request_id": self.request_id,
            "data": self.data or {},
        }
        return json.dumps(message_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "ZMQMessage":
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            message_type=MessageType(data["message_type"]),
            timestamp=data["timestamp"],
            client_id=data.get("client_id"),
            request_id=data.get("request_id"),
            data=data.get("data", {}),
        )

    @classmethod
    def create_command(
        cls,
        message_type: MessageType,
        client_id: str,
        request_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> "ZMQMessage":
        """Create a command message."""
        return cls(
            message_type=message_type,
            timestamp=time.time(),
            client_id=client_id,
            request_id=request_id,
            data=data,
        )

    @classmethod
    def create_response(
        cls,
        message_type: MessageType,
        request_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> "ZMQMessage":
        """Create a response message."""
        return cls(
            message_type=message_type,
            timestamp=time.time(),
            request_id=request_id,
            data=data,
        )

    @classmethod
    def create_broadcast(
        cls, message_type: MessageType, data: Optional[Dict[str, Any]] = None
    ) -> "ZMQMessage":
        """Create a broadcast message."""
        return cls(message_type=message_type, timestamp=time.time(), data=data)


class MessageBuilder:
    """Helper class for building common message types."""

    @staticmethod
    def status_update(simulation_status: str, **kwargs) -> ZMQMessage:
        """Build a status update broadcast message."""
        data = {"simulation_status": simulation_status}
        data.update(kwargs)
        return ZMQMessage.create_broadcast(MessageType.STATUS_UPDATE, data)

    @staticmethod
    def error_occurred(
        error_type: str, error_message: str, component: str, recoverable: bool
    ) -> ZMQMessage:
        """Build an error broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.ERROR_OCCURRED,
            {
                "error_type": error_type,
                "error_message": error_message,
                "component": component,
                "recoverable": recoverable,
            },
        )

    @staticmethod
    def heartbeat(
        server_status: str, uptime_seconds: float, active_clients: int
    ) -> ZMQMessage:
        """Build a heartbeat broadcast message."""
        return ZMQMessage.create_broadcast(
            MessageType.HEARTBEAT,
            {
                "server_status": server_status,
                "uptime_seconds": uptime_seconds,
                "active_clients": active_clients,
            },
        )


class MessageValidator:
    """Validates incoming messages for correctness."""

    REQUIRED_FIELDS = {
        MessageType.START_SIMULATION: ["config"],
        MessageType.UPDATE_CONFIG: ["config"],
        MessageType.GET_STATUS: [],
        MessageType.STOP_SIMULATION: [],
        MessageType.PAUSE_SIMULATION: [],
        MessageType.RESUME_SIMULATION: [],
        MessageType.RESET_SIMULATION: [],
    }

    @classmethod
    def validate_message(cls, message: ZMQMessage) -> tuple[bool, Optional[str]]:
        """
        Validate a message for correctness.

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check if message type is supported
        if message.message_type not in cls.REQUIRED_FIELDS:
            return False, f"Unsupported message type: {message.message_type.value}"

        # Check required fields
        required_fields = cls.REQUIRED_FIELDS[message.message_type]
        message_data = message.data or {}

        for field in required_fields:
            if field not in message_data:
                return False, f"Missing required field: {field}"

        # Validate specific message types
        if message.message_type == MessageType.START_SIMULATION:
            return cls._validate_start_simulation(message_data)
        elif message.message_type == MessageType.UPDATE_CONFIG:
            return cls._validate_update_config(message_data)

        return True, None

    @classmethod
    def _validate_start_simulation(
        cls, data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate start simulation message data."""
        config = data.get("config", {})

        # Basic validation - can be extended for specific use cases
        if not isinstance(config, dict):
            return False, "config must be a dictionary"

        return True, None

    @classmethod
    def _validate_update_config(
        cls, data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate update config message data."""
        # Use same validation as start simulation
        return cls._validate_start_simulation(data)
