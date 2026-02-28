# ai_hydra/utils/HydraMsg.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

import json
from typing import Any, Optional

from ai_hydra.constants.DHydra import DHydra, DHydraMsg


class HydraMsg:
    """
    Structured message class for HydraRouter communication protocol.

    HydraMsg encapsulates messages for ZeroMQ-based RPC communication.
    It handles serialization/deserialization and provides a clean API
    for creating and accessing message components.

    Internal representation uses Python objects (dict for payload).
    Serialization to JSON happens only when converting to wire format.

        # Serialize for transmission
        json_bytes = msg.to_json()

        # Deserialize from received data
        msg = HydraMsg.from_json(json_bytes)
    """

    def __init__(
        self,
        sender: str,
        method: str,
        target: Optional[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a new HydraMsg instance.

        Args:
            sender: Identifier of the message sender
            target: Identifier of the intended message recipient
            method: RPC method or action to be performed
            payload: Message data as a dictionary (not JSON string)

        Returns:
            None
        """
        self._sender = sender
        self._target = target
        self._method = method
        self._payload = payload if payload is not None else {}

    @property
    def sender(self) -> str:
        """Get the message sender identifier."""
        return self._sender

    @sender.setter
    def sender(self, value: str) -> None:
        """Set the message sender identifier."""
        self._sender = value

    @property
    def target(self) -> str:
        """Get the message target identifier."""
        return self._target

    @target.setter
    def target(self, value: str) -> None:
        """Set the message target identifier."""
        self._target = value

    @property
    def method(self) -> str:
        """Get the RPC method name."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set the RPC method name."""
        self._method = value

    @property
    def payload(self) -> dict[str, Any]:
        """Get the message payload as a dictionary."""
        return self._payload

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        """Set the message payload from a dictionary."""
        self._payload = value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HydraMsg":
        """
        Create a HydraMsg from a dictionary.

        Args:
            data: Dictionary containing message fields

        Returns:
            HydraMsg instance

        Raises:
            KeyError: If required fields are missing
        """
        pv = data[DHydraMsg.PROTOCOL_VERSION]
        if pv != DHydra.PROTOCOL_VERSION:
            raise ValueError(f"Unsupported Hydra protocol version: {pv}")

        payload = data.get(DHydraMsg.PAYLOAD) or {}
        if not isinstance(payload, dict):
            raise TypeError("Payload must be a dict")

        return cls(
            sender=data[DHydraMsg.SENDER],
            target=data[DHydraMsg.TARGET],
            method=data[DHydraMsg.METHOD],
            payload=payload,
        )

    @classmethod
    def from_json(cls, json_data: bytes) -> "HydraMsg":
        """
        Create a HydraMsg from JSON bytes.

        Args:
            json_data: JSON-encoded message as bytes

        Returns:
            HydraMsg instance

        Raises:
            json.JSONDecodeError: If json_data is not valid JSON
            UnicodeDecodeError: If json_data cannot be decoded as UTF-8
        """
        return cls.from_dict(json.loads(json_data.decode("utf-8")))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert message to dictionary representation.

        Returns:
            Dictionary containing all message fields including version
        """
        return {
            DHydraMsg.SENDER: self._sender,
            DHydraMsg.TARGET: self._target,
            DHydraMsg.METHOD: self._method,
            DHydraMsg.PAYLOAD: self._payload,
            DHydraMsg.PROTOCOL_VERSION: DHydra.PROTOCOL_VERSION,
        }

    def to_json(self) -> bytes:
        """
        Convert message to JSON bytes for transmission.

        Returns:
            JSON-encoded message as UTF-8 bytes ready for ZeroMQ

        Raises:
            TypeError: If message contains non-serializable objects
        """
        return json.dumps(self.to_dict()).encode("utf-8")

    def __repr__(self) -> str:
        """
        Return string representation of message for debugging.

        Returns:
            String showing message structure
        """
        return (
            f"HydraMsg:{self._sender}->{self._target}:{self._method}"
            f"({self._payload})"
        )

    def __str__(self) -> str:
        """
        Return human-readable string representation.

        Returns:
            Formatted string with message details
        """
        return f"HydraMsg:{self._sender}->{self._target}:{self._method}()"
