"""
Unit tests for Hydra Router message validation.

Tests the MessageValidator class and validation functions for RouterConstants
format compliance with comprehensive error reporting.
"""

import pytest
import time
from typing import Dict, Any

from ai_hydra.validation import (
    MessageValidator,
    validate_message,
    validate_message_strict,
)
from ai_hydra.router_constants import RouterConstants
from ai_hydra.exceptions import MessageValidationError


class TestMessageValidator:
    """Unit tests for MessageValidator class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.validator = MessageValidator()

        # Valid message template
        self.valid_message = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: RouterConstants.HEARTBEAT,
            RouterConstants.DATA: {},
            RouterConstants.CLIENT_ID: "test-client-123",
            RouterConstants.TIMESTAMP: time.time(),
            RouterConstants.REQUEST_ID: "req-123",
        }

    def test_valid_message_passes_validation(self):
        """Test that a valid RouterConstants message passes validation."""
        is_valid, error_msg = self.validator.validate_router_message(self.valid_message)

        assert is_valid is True
        assert error_msg == ""

    def test_missing_required_fields_fails_validation(self):
        """Test that missing required fields fail validation."""
        # Test missing sender
        invalid_message = self.valid_message.copy()
        del invalid_message[RouterConstants.SENDER]

        is_valid, error_msg = self.validator.validate_router_message(invalid_message)

        assert is_valid is False
        assert "Missing required fields" in error_msg
        assert RouterConstants.SENDER in error_msg

    def test_invalid_field_types_fail_validation(self):
        """Test that invalid field types fail validation."""
        # Test invalid sender type (should be string)
        invalid_message = self.valid_message.copy()
        invalid_message[RouterConstants.SENDER] = 123

        is_valid, error_msg = self.validator.validate_router_message(invalid_message)

        assert is_valid is False
        assert "must be str" in error_msg

    def test_invalid_sender_type_fails_validation(self):
        """Test that invalid sender types fail validation."""
        invalid_message = self.valid_message.copy()
        invalid_message[RouterConstants.SENDER] = "InvalidSender"

        is_valid, error_msg = self.validator.validate_router_message(invalid_message)

        assert is_valid is False
        assert "Invalid sender type" in error_msg
        assert "InvalidSender" in error_msg

    def test_invalid_message_element_fails_validation(self):
        """Test that invalid message elements fail validation."""
        invalid_message = self.valid_message.copy()
        invalid_message[RouterConstants.ELEM] = "invalid_element"

        is_valid, error_msg = self.validator.validate_router_message(invalid_message)

        assert is_valid is False
        assert "Invalid message element" in error_msg
        assert "invalid_element" in error_msg

    def test_empty_string_fields_fail_validation(self):
        """Test that empty string fields fail validation."""
        invalid_message = self.valid_message.copy()
        invalid_message[RouterConstants.SENDER] = ""

        is_valid, error_msg = self.validator.validate_router_message(invalid_message)

        assert is_valid is False
        assert "cannot be empty" in error_msg

    def test_invalid_timestamp_fails_validation(self):
        """Test that invalid timestamps fail validation."""
        # Test negative timestamp
        invalid_message = self.valid_message.copy()
        invalid_message[RouterConstants.TIMESTAMP] = -1

        is_valid, error_msg = self.validator.validate_router_message(invalid_message)

        assert is_valid is False
        assert "cannot be negative" in error_msg

    def test_non_dict_message_fails_validation(self):
        """Test that non-dictionary messages fail validation."""
        is_valid, error_msg = self.validator.validate_router_message("not a dict")

        assert is_valid is False
        assert "must be a dictionary" in error_msg

    def test_validate_sender_type_method(self):
        """Test the validate_sender_type method."""
        # Valid sender types
        assert self.validator.validate_sender_type(RouterConstants.HYDRA_CLIENT) is True
        assert self.validator.validate_sender_type(RouterConstants.HYDRA_SERVER) is True
        assert self.validator.validate_sender_type(RouterConstants.HYDRA_ROUTER) is True

        # Invalid sender types
        assert self.validator.validate_sender_type("InvalidSender") is False
        assert self.validator.validate_sender_type(123) is False
        assert self.validator.validate_sender_type(None) is False

    def test_validate_field_types_method(self):
        """Test the validate_field_types method."""
        # Valid field types
        is_valid, error_msg = self.validator.validate_field_types(self.valid_message)
        assert is_valid is True
        assert error_msg == ""

        # Invalid field type
        invalid_message = self.valid_message.copy()
        invalid_message[RouterConstants.DATA] = "should be dict"

        is_valid, error_msg = self.validator.validate_field_types(invalid_message)
        assert is_valid is False
        assert "must be dict" in error_msg

    def test_get_validation_error_details(self):
        """Test detailed error reporting."""
        invalid_message = {
            RouterConstants.SENDER: "InvalidSender",
            RouterConstants.ELEM: "invalid_element",
            RouterConstants.DATA: "should be dict",
            RouterConstants.MESSAGE_TYPE: "heartbeat",  # ZMQMessage format field
        }

        details = self.validator.get_validation_error_details(invalid_message)

        assert "Invalid sender 'InvalidSender'" in details
        assert "Invalid element 'invalid_element'" in details
        assert "ZMQMessage format" in details

    def test_create_validation_error(self):
        """Test creation of detailed validation errors."""
        invalid_message = {"invalid": "message"}
        error_msg = "Test error"

        error = self.validator.create_validation_error(invalid_message, error_msg)

        assert isinstance(error, MessageValidationError)
        assert str(error) == error_msg
        assert error.invalid_message == invalid_message
        assert error.expected_format is not None


class TestValidationFunctions:
    """Test module-level validation functions."""

    def test_validate_message_function(self):
        """Test the validate_message convenience function."""
        valid_message = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: RouterConstants.HEARTBEAT,
            RouterConstants.DATA: {},
            RouterConstants.CLIENT_ID: "test-client",
            RouterConstants.TIMESTAMP: time.time(),
        }

        is_valid, error_msg = validate_message(valid_message)
        assert is_valid is True
        assert error_msg == ""

    def test_validate_message_strict_function(self):
        """Test the validate_message_strict function."""
        valid_message = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: RouterConstants.HEARTBEAT,
            RouterConstants.DATA: {},
            RouterConstants.CLIENT_ID: "test-client",
            RouterConstants.TIMESTAMP: time.time(),
        }

        # Should not raise exception for valid message
        validate_message_strict(valid_message)

        # Should raise exception for invalid message
        invalid_message = {"invalid": "message"}
        with pytest.raises(MessageValidationError):
            validate_message_strict(invalid_message)


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_optional_fields_validation(self):
        """Test validation with optional fields."""
        validator = MessageValidator()

        # Message without optional fields should be valid
        minimal_message = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: RouterConstants.HEARTBEAT,
            RouterConstants.DATA: {},
            RouterConstants.CLIENT_ID: "test-client",
            RouterConstants.TIMESTAMP: time.time(),
        }

        is_valid, error_msg = validator.validate_router_message(minimal_message)
        assert is_valid is True

        # Message with optional fields should be valid
        full_message = minimal_message.copy()
        full_message[RouterConstants.REQUEST_ID] = "req-123"

        is_valid, error_msg = validator.validate_router_message(full_message)
        assert is_valid is True

    def test_timestamp_boundary_values(self):
        """Test timestamp validation with boundary values."""
        validator = MessageValidator()

        base_message = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: RouterConstants.HEARTBEAT,
            RouterConstants.DATA: {},
            RouterConstants.CLIENT_ID: "test-client",
        }

        # Test valid timestamps
        valid_timestamps = [0, 1, time.time(), 4102444799]  # Just before year 2100

        for timestamp in valid_timestamps:
            message = base_message.copy()
            message[RouterConstants.TIMESTAMP] = timestamp

            is_valid, error_msg = validator.validate_router_message(message)
            assert is_valid is True, f"Timestamp {timestamp} should be valid"

        # Test invalid timestamps
        invalid_timestamps = [-1, 4102444801]  # Negative and too far in future

        for timestamp in invalid_timestamps:
            message = base_message.copy()
            message[RouterConstants.TIMESTAMP] = timestamp

            is_valid, error_msg = validator.validate_router_message(message)
            assert is_valid is False, f"Timestamp {timestamp} should be invalid"

    def test_data_field_variations(self):
        """Test validation with different data field values."""
        validator = MessageValidator()

        base_message = {
            RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
            RouterConstants.ELEM: RouterConstants.HEARTBEAT,
            RouterConstants.CLIENT_ID: "test-client",
            RouterConstants.TIMESTAMP: time.time(),
        }

        # Test valid data values
        valid_data_values = [
            {},
            {"key": "value"},
            {"nested": {"data": "value"}},
            {"list": [1, 2, 3]},
            {"mixed": {"string": "value", "number": 42, "list": [1, 2]}},
        ]

        for data in valid_data_values:
            message = base_message.copy()
            message[RouterConstants.DATA] = data

            is_valid, error_msg = validator.validate_router_message(message)
            assert is_valid is True, f"Data {data} should be valid"

        # Test invalid data values
        invalid_data_values = ["string", 123, [1, 2, 3], None]

        for data in invalid_data_values:
            message = base_message.copy()
            message[RouterConstants.DATA] = data

            is_valid, error_msg = validator.validate_router_message(message)
            assert is_valid is False, f"Data {data} should be invalid"
