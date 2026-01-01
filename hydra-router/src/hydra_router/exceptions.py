"""
Hydra Router Custom Exceptions

Custom exception hierarchy for router-specific errors with detailed context
and debugging information.
"""

from typing import Optional, Dict, Any


class HydraRouterError(Exception):
    """Base exception for Hydra Router errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize router error with context.

        Args:
            message: Error description
            context: Additional context for debugging
        """
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted error message with context."""
        base_msg = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} (Context: {context_str})"
        return base_msg


class MessageValidationError(HydraRouterError):
    """Raised when message format validation fails."""

    def __init__(
        self,
        message: str,
        invalid_message: Optional[Dict[str, Any]] = None,
        expected_format: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize message validation error.

        Args:
            message: Error description
            invalid_message: The message that failed validation
            expected_format: Expected message format specification
        """
        context = {}
        if invalid_message is not None:
            context["invalid_message"] = str(invalid_message)[
                :200
            ]  # Truncate for safety
        if expected_format is not None:
            context["expected_format"] = expected_format

        super().__init__(message, context)
        self.invalid_message = invalid_message
        self.expected_format = expected_format


class ConnectionError(HydraRouterError):
    """Raised when network connection issues occur."""

    def __init__(
        self,
        message: str,
        address: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[str] = None,
    ):
        """
        Initialize connection error.

        Args:
            message: Error description
            address: Network address involved
            port: Network port involved
            client_id: Client identifier if applicable
        """
        context = {}
        if address is not None:
            context["address"] = address
        if port is not None:
            context["port"] = port
        if client_id is not None:
            context["client_id"] = client_id

        super().__init__(message, context)
        self.address = address
        self.port = port
        self.client_id = client_id


class ClientRegistrationError(HydraRouterError):
    """Raised when client registration or management issues occur."""

    def __init__(
        self,
        message: str,
        client_id: Optional[str] = None,
        client_type: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        """
        Initialize client registration error.

        Args:
            message: Error description
            client_id: Client identifier involved
            client_type: Type of client (HydraClient, HydraServer, etc.)
            operation: Operation that failed (register, update, remove, etc.)
        """
        context = {}
        if client_id is not None:
            context["client_id"] = client_id
        if client_type is not None:
            context["client_type"] = client_type
        if operation is not None:
            context["operation"] = operation

        super().__init__(message, context)
        self.client_id = client_id
        self.client_type = client_type
        self.operation = operation


class MessageFormatError(HydraRouterError):
    """Raised when message format conversion fails."""

    def __init__(
        self,
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        conversion_stage: Optional[str] = None,
    ):
        """
        Initialize message format error.

        Args:
            message: Error description
            source_format: Source message format (ZMQMessage, RouterConstants, etc.)
            target_format: Target message format
            conversion_stage: Stage where conversion failed
        """
        context = {}
        if source_format is not None:
            context["source_format"] = source_format
        if target_format is not None:
            context["target_format"] = target_format
        if conversion_stage is not None:
            context["conversion_stage"] = conversion_stage

        super().__init__(message, context)
        self.source_format = source_format
        self.target_format = target_format
        self.conversion_stage = conversion_stage


class RouterConfigurationError(HydraRouterError):
    """Raised when router configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        valid_values: Optional[list] = None,
    ):
        """
        Initialize router configuration error.

        Args:
            message: Error description
            config_key: Configuration key that is invalid
            config_value: Invalid configuration value
            valid_values: List of valid values for the configuration
        """
        context = {}
        if config_key is not None:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)
        if valid_values is not None:
            context["valid_values"] = valid_values

        super().__init__(message, context)
        self.config_key = config_key
        self.config_value = config_value
        self.valid_values = valid_values


class HeartbeatError(HydraRouterError):
    """Raised when heartbeat mechanism fails."""

    def __init__(
        self,
        message: str,
        client_id: Optional[str] = None,
        last_heartbeat: Optional[float] = None,
        timeout_threshold: Optional[float] = None,
    ):
        """
        Initialize heartbeat error.

        Args:
            message: Error description
            client_id: Client that failed heartbeat
            last_heartbeat: Timestamp of last successful heartbeat
            timeout_threshold: Heartbeat timeout threshold
        """
        context = {}
        if client_id is not None:
            context["client_id"] = client_id
        if last_heartbeat is not None:
            context["last_heartbeat"] = last_heartbeat
        if timeout_threshold is not None:
            context["timeout_threshold"] = timeout_threshold

        super().__init__(message, context)
        self.client_id = client_id
        self.last_heartbeat = last_heartbeat
        self.timeout_threshold = timeout_threshold


class RoutingError(HydraRouterError):
    """Raised when message routing fails."""

    def __init__(
        self,
        message: str,
        message_type: Optional[str] = None,
        sender_id: Optional[str] = None,
        target_id: Optional[str] = None,
        routing_rule: Optional[str] = None,
    ):
        """
        Initialize routing error.

        Args:
            message: Error description
            message_type: Type of message that failed to route
            sender_id: ID of message sender
            target_id: ID of intended target
            routing_rule: Routing rule that was applied
        """
        context = {}
        if message_type is not None:
            context["message_type"] = message_type
        if sender_id is not None:
            context["sender_id"] = sender_id
        if target_id is not None:
            context["target_id"] = target_id
        if routing_rule is not None:
            context["routing_rule"] = routing_rule

        super().__init__(message, context)
        self.message_type = message_type
        self.sender_id = sender_id
        self.target_id = target_id
        self.routing_rule = routing_rule
