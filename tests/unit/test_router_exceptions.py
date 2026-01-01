"""
Unit tests for Hydra Router custom exceptions.

Tests the custom exception hierarchy with context and debugging information.
"""

import pytest
from ai_hydra.exceptions import (
    HydraRouterError,
    MessageValidationError,
    ConnectionError,
    ClientRegistrationError,
    MessageFormatError,
    RouterConfigurationError,
    HeartbeatError,
    RoutingError,
)


class TestHydraRouterError:
    """Test base HydraRouterError class."""

    def test_basic_error_creation(self):
        """Test creating basic error without context."""
        error = HydraRouterError("Test error message")

        assert str(error) == "Test error message"
        assert error.context == {}

    def test_error_with_context(self):
        """Test creating error with context information."""
        context = {"client_id": "test-123", "operation": "connect"}
        error = HydraRouterError("Test error", context=context)

        assert "Test error" in str(error)
        assert "client_id=test-123" in str(error)
        assert "operation=connect" in str(error)
        assert error.context == context

    def test_error_inheritance(self):
        """Test that HydraRouterError inherits from Exception."""
        error = HydraRouterError("Test error")

        assert isinstance(error, Exception)
        assert isinstance(error, HydraRouterError)


class TestMessageValidationError:
    """Test MessageValidationError class."""

    def test_basic_validation_error(self):
        """Test creating basic validation error."""
        error = MessageValidationError("Invalid message format")

        assert "Invalid message format" in str(error)
        assert error.invalid_message is None
        assert error.expected_format is None

    def test_validation_error_with_message(self):
        """Test validation error with invalid message."""
        invalid_msg = {"invalid": "message"}
        error = MessageValidationError(
            "Missing required fields", invalid_message=invalid_msg
        )

        assert "Missing required fields" in str(error)
        assert error.invalid_message == invalid_msg
        assert "invalid_message" in str(error)

    def test_validation_error_with_expected_format(self):
        """Test validation error with expected format."""
        expected = {"sender": "string", "elem": "string"}
        error = MessageValidationError("Format mismatch", expected_format=expected)

        assert "Format mismatch" in str(error)
        assert error.expected_format == expected
        assert "expected_format" in str(error)

    def test_validation_error_truncates_long_message(self):
        """Test that long invalid messages are truncated."""
        long_message = {"data": "x" * 300}  # Very long message
        error = MessageValidationError("Test error", invalid_message=long_message)

        # Should be truncated to 200 characters
        context_str = str(error)
        assert len(context_str) < 400  # Much shorter than original


class TestConnectionError:
    """Test ConnectionError class."""

    def test_basic_connection_error(self):
        """Test creating basic connection error."""
        error = ConnectionError("Connection failed")

        assert "Connection failed" in str(error)
        assert error.address is None
        assert error.port is None
        assert error.client_id is None

    def test_connection_error_with_network_info(self):
        """Test connection error with network information."""
        error = ConnectionError(
            "Failed to bind", address="localhost", port=5556, client_id="test-client"
        )

        assert "Failed to bind" in str(error)
        assert error.address == "localhost"
        assert error.port == 5556
        assert error.client_id == "test-client"
        assert "address=localhost" in str(error)
        assert "port=5556" in str(error)


class TestClientRegistrationError:
    """Test ClientRegistrationError class."""

    def test_basic_registration_error(self):
        """Test creating basic registration error."""
        error = ClientRegistrationError("Registration failed")

        assert "Registration failed" in str(error)
        assert error.client_id is None
        assert error.client_type is None
        assert error.operation is None

    def test_registration_error_with_client_info(self):
        """Test registration error with client information."""
        error = ClientRegistrationError(
            "Invalid client type",
            client_id="test-123",
            client_type="InvalidType",
            operation="register",
        )

        assert "Invalid client type" in str(error)
        assert error.client_id == "test-123"
        assert error.client_type == "InvalidType"
        assert error.operation == "register"
        assert "client_id=test-123" in str(error)


class TestMessageFormatError:
    """Test MessageFormatError class."""

    def test_basic_format_error(self):
        """Test creating basic format error."""
        error = MessageFormatError("Format conversion failed")

        assert "Format conversion failed" in str(error)
        assert error.source_format is None
        assert error.target_format is None
        assert error.conversion_stage is None

    def test_format_error_with_conversion_info(self):
        """Test format error with conversion information."""
        error = MessageFormatError(
            "Conversion failed",
            source_format="ZMQMessage",
            target_format="RouterConstants",
            conversion_stage="field_mapping",
        )

        assert "Conversion failed" in str(error)
        assert error.source_format == "ZMQMessage"
        assert error.target_format == "RouterConstants"
        assert error.conversion_stage == "field_mapping"
        assert "source_format=ZMQMessage" in str(error)


class TestRouterConfigurationError:
    """Test RouterConfigurationError class."""

    def test_basic_config_error(self):
        """Test creating basic configuration error."""
        error = RouterConfigurationError("Invalid configuration")

        assert "Invalid configuration" in str(error)
        assert error.config_key is None
        assert error.config_value is None
        assert error.valid_values is None

    def test_config_error_with_details(self):
        """Test configuration error with detailed information."""
        error = RouterConfigurationError(
            "Invalid log level",
            config_key="log_level",
            config_value="INVALID",
            valid_values=["DEBUG", "INFO", "WARNING", "ERROR"],
        )

        assert "Invalid log level" in str(error)
        assert error.config_key == "log_level"
        assert error.config_value == "INVALID"
        assert error.valid_values == ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert "config_key=log_level" in str(error)


class TestHeartbeatError:
    """Test HeartbeatError class."""

    def test_basic_heartbeat_error(self):
        """Test creating basic heartbeat error."""
        error = HeartbeatError("Heartbeat timeout")

        assert "Heartbeat timeout" in str(error)
        assert error.client_id is None
        assert error.last_heartbeat is None
        assert error.timeout_threshold is None

    def test_heartbeat_error_with_timing_info(self):
        """Test heartbeat error with timing information."""
        error = HeartbeatError(
            "Client timeout",
            client_id="test-client",
            last_heartbeat=1234567890.0,
            timeout_threshold=15.0,
        )

        assert "Client timeout" in str(error)
        assert error.client_id == "test-client"
        assert error.last_heartbeat == 1234567890.0
        assert error.timeout_threshold == 15.0
        assert "client_id=test-client" in str(error)


class TestRoutingError:
    """Test RoutingError class."""

    def test_basic_routing_error(self):
        """Test creating basic routing error."""
        error = RoutingError("Message routing failed")

        assert "Message routing failed" in str(error)
        assert error.message_type is None
        assert error.sender_id is None
        assert error.target_id is None
        assert error.routing_rule is None

    def test_routing_error_with_routing_info(self):
        """Test routing error with routing information."""
        error = RoutingError(
            "No route found",
            message_type="start_simulation",
            sender_id="client-123",
            target_id="server-456",
            routing_rule="client_to_server",
        )

        assert "No route found" in str(error)
        assert error.message_type == "start_simulation"
        assert error.sender_id == "client-123"
        assert error.target_id == "server-456"
        assert error.routing_rule == "client_to_server"
        assert "message_type=start_simulation" in str(error)


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from HydraRouterError."""
        exceptions = [
            MessageValidationError("test"),
            ConnectionError("test"),
            ClientRegistrationError("test"),
            MessageFormatError("test"),
            RouterConfigurationError("test"),
            HeartbeatError("test"),
            RoutingError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, HydraRouterError)
            assert isinstance(exc, Exception)

    def test_exception_context_inheritance(self):
        """Test that all exceptions inherit context functionality."""
        error = MessageValidationError("test", invalid_message={"test": "data"})

        # Should have context from base class
        assert hasattr(error, "context")
        assert isinstance(error.context, dict)
        assert "invalid_message" in error.context
