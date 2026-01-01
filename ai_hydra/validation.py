"""
Hydra Router Message Validation

Comprehensive message validation framework with detailed error reporting
for RouterConstants format compliance.
"""

import time
from typing import Dict, Any, Tuple, Optional, List
from ai_hydra.router_constants import RouterConstants
from ai_hydra.exceptions import MessageValidationError


class MessageValidator:
    """Validates RouterConstants format messages with detailed error reporting."""

    # Valid sender types
    VALID_SENDERS = [
        RouterConstants.HYDRA_CLIENT,
        RouterConstants.HYDRA_SERVER,
        RouterConstants.HYDRA_ROUTER,
    ]

    # Valid message elements
    VALID_ELEMENTS = [
        RouterConstants.HEARTBEAT,
        RouterConstants.STATUS,
        RouterConstants.ERROR,
        RouterConstants.START_SIMULATION,
        RouterConstants.STOP_SIMULATION,
        RouterConstants.PAUSE_SIMULATION,
        RouterConstants.RESUME_SIMULATION,
        RouterConstants.RESET_SIMULATION,
        RouterConstants.SIMULATION_STARTED,
        RouterConstants.SIMULATION_STOPPED,
        RouterConstants.SIMULATION_PAUSED,
        RouterConstants.SIMULATION_RESUMED,
        RouterConstants.SIMULATION_RESET,
        RouterConstants.GET_STATUS,
        RouterConstants.STATUS_UPDATE,
        RouterConstants.GAME_STATE_UPDATE,
        RouterConstants.PERFORMANCE_UPDATE,
        RouterConstants.GET_CONFIG,
        RouterConstants.SET_CONFIG,
        RouterConstants.CONFIG_UPDATE,
    ]

    # Required fields for RouterConstants format
    REQUIRED_FIELDS = [
        RouterConstants.SENDER,
        RouterConstants.ELEM,
        RouterConstants.DATA,
        RouterConstants.CLIENT_ID,
        RouterConstants.TIMESTAMP,
    ]

    # Optional fields
    OPTIONAL_FIELDS = [
        RouterConstants.REQUEST_ID,
    ]

    def __init__(self):
        """Initialize message validator."""
        self.validation_cache = {}  # Cache for performance optimization

    def validate_router_message(self, message: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate RouterConstants format message with comprehensive error reporting.

        Args:
            message: Message dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic type validation
            if not isinstance(message, dict):
                return (
                    False,
                    f"Message must be a dictionary, got {type(message).__name__}",
                )

            # Check for required fields
            missing_fields = self._check_required_fields(message)
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"

            # Validate field types
            type_validation_result = self.validate_field_types(message)
            if not type_validation_result[0]:
                return type_validation_result

            # Validate field values
            value_validation_result = self._validate_field_values(message)
            if not value_validation_result[0]:
                return value_validation_result

            # Validate sender type
            if not self.validate_sender_type(message[RouterConstants.SENDER]):
                return (
                    False,
                    f"Invalid sender type: '{message[RouterConstants.SENDER]}'. Valid types: {', '.join(self.VALID_SENDERS)}",
                )

            # Validate message element
            if not self._validate_message_element(message[RouterConstants.ELEM]):
                return (
                    False,
                    f"Invalid message element: '{message[RouterConstants.ELEM]}'. Valid elements: {', '.join(self.VALID_ELEMENTS)}",
                )

            # Validate timestamp
            timestamp_validation = self._validate_timestamp(
                message[RouterConstants.TIMESTAMP]
            )
            if not timestamp_validation[0]:
                return timestamp_validation

            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_field_types(self, message: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that all fields have correct types.

        Args:
            message: Message to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        type_requirements = {
            RouterConstants.SENDER: str,
            RouterConstants.ELEM: str,
            RouterConstants.DATA: dict,
            RouterConstants.CLIENT_ID: str,
            RouterConstants.TIMESTAMP: (int, float),
            RouterConstants.REQUEST_ID: (str, type(None)),  # Optional string
        }

        for field, expected_type in type_requirements.items():
            if field in message:
                if not isinstance(message[field], expected_type):
                    actual_type = type(message[field]).__name__
                    if isinstance(expected_type, tuple):
                        expected_names = [t.__name__ for t in expected_type]
                        expected_str = " or ".join(expected_names)
                    else:
                        expected_str = expected_type.__name__

                    return (
                        False,
                        f"Field '{field}' must be {expected_str}, got {actual_type}",
                    )

        return True, ""

    def validate_sender_type(self, sender: str) -> bool:
        """
        Validate sender type against allowed values.

        Args:
            sender: Sender type string

        Returns:
            True if valid, False otherwise
        """
        return isinstance(sender, str) and sender in self.VALID_SENDERS

    def get_validation_error_details(self, message: Dict[str, Any]) -> str:
        """
        Get detailed validation error information for debugging.

        Args:
            message: Message that failed validation

        Returns:
            Detailed error description with context
        """
        details = []

        # Check message type
        if not isinstance(message, dict):
            details.append(f"Message type: Expected dict, got {type(message).__name__}")
            return "; ".join(details)

        # Check required fields
        missing_fields = self._check_required_fields(message)
        if missing_fields:
            details.append(f"Missing fields: {', '.join(missing_fields)}")

        # Check field types
        for field in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            if field in message:
                field_type = type(message[field]).__name__
                details.append(f"Field '{field}': {field_type}")

        # Check sender validation
        sender = message.get(RouterConstants.SENDER)
        if sender and not self.validate_sender_type(sender):
            details.append(
                f"Invalid sender '{sender}', expected one of: {', '.join(self.VALID_SENDERS)}"
            )

        # Check element validation
        elem = message.get(RouterConstants.ELEM)
        if elem and not self._validate_message_element(elem):
            details.append(
                f"Invalid element '{elem}', expected one of: {', '.join(self.VALID_ELEMENTS)}"
            )

        # Check for common format issues
        if (
            RouterConstants.MESSAGE_TYPE in message
            and RouterConstants.ELEM not in message
        ):
            details.append(
                "Detected ZMQMessage format (has 'message_type' but missing 'elem')"
            )

        return (
            "; ".join(details) if details else "No specific validation errors detected"
        )

    def _check_required_fields(self, message: Dict[str, Any]) -> List[str]:
        """Check for missing required fields."""
        missing = []
        for field in self.REQUIRED_FIELDS:
            if field not in message:
                missing.append(field)
        return missing

    def _validate_field_values(self, message: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate field values are not empty/invalid."""
        # Check string fields are not empty
        string_fields = [
            RouterConstants.SENDER,
            RouterConstants.ELEM,
            RouterConstants.CLIENT_ID,
        ]
        for field in string_fields:
            if field in message:
                value = message[field]
                if isinstance(value, str) and not value.strip():
                    return False, f"Field '{field}' cannot be empty"

        # Check data field is valid dict
        if RouterConstants.DATA in message:
            data = message[RouterConstants.DATA]
            if not isinstance(data, dict):
                return False, f"Field '{RouterConstants.DATA}' must be a dictionary"

        return True, ""

    def _validate_message_element(self, elem: str) -> bool:
        """Validate message element against allowed values."""
        return isinstance(elem, str) and elem in self.VALID_ELEMENTS

    def _validate_timestamp(self, timestamp: float) -> Tuple[bool, str]:
        """Validate timestamp is reasonable."""
        if not isinstance(timestamp, (int, float)):
            return False, f"Timestamp must be number, got {type(timestamp).__name__}"

        if timestamp < 0:
            return False, "Timestamp cannot be negative"

        # Check timestamp is not unreasonably far in future (year 2100)
        if timestamp > 4102444800:
            return False, "Timestamp is unreasonably far in the future"

        # Check timestamp is not too old (before 1970)
        if timestamp < 0:
            return False, "Timestamp is before Unix epoch"

        return True, ""

    def create_validation_error(
        self, message: Dict[str, Any], error_msg: str
    ) -> MessageValidationError:
        """
        Create a detailed MessageValidationError with context.

        Args:
            message: Invalid message
            error_msg: Error description

        Returns:
            MessageValidationError with full context
        """
        expected_format = {
            RouterConstants.SENDER: f"string, one of: {', '.join(self.VALID_SENDERS)}",
            RouterConstants.ELEM: f"string, one of: {', '.join(self.VALID_ELEMENTS)}",
            RouterConstants.DATA: "dict",
            RouterConstants.CLIENT_ID: "string (non-empty)",
            RouterConstants.TIMESTAMP: "number (Unix timestamp)",
            RouterConstants.REQUEST_ID: "string (optional)",
        }

        return MessageValidationError(
            message=error_msg, invalid_message=message, expected_format=expected_format
        )


# Global validator instance for performance
_validator_instance = None


def get_validator() -> MessageValidator:
    """Get singleton validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = MessageValidator()
    return _validator_instance


def validate_message(message: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Convenience function for message validation.

    Args:
        message: Message to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return get_validator().validate_router_message(message)


def validate_message_strict(message: Dict[str, Any]) -> None:
    """
    Strict validation that raises exception on failure.

    Args:
        message: Message to validate

    Raises:
        MessageValidationError: If validation fails
    """
    validator = get_validator()
    is_valid, error_msg = validator.validate_router_message(message)

    if not is_valid:
        raise validator.create_validation_error(message, error_msg)
