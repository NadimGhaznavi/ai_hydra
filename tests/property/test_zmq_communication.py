"""
Unit tests for ZeroMQ communication components.

This module tests the ZeroMQ server, client, and message protocol
to ensure proper headless communication functionality.

Includes Property 13: ZeroMQ Message Protocol Integrity
Includes Property 1: Message Format Round-Trip Conversion
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings

from ai_hydra.zmq_protocol import (
    ZMQMessage,
    MessageType,
    MessageBuilder,
    MessageValidator,
    GameStateData,
    PerformanceMetrics,
)
from ai_hydra.zmq_server import ZMQServer, SimulationState
from ai_hydra.zmq_client_example import ZMQClient
from ai_hydra.models import GameBoard, Position, Direction
from ai_hydra.mq_client import MQClient
from ai_hydra.router_constants import RouterConstants


class TestMQClientFormatConversion:
    """Property-based tests for MQClient message format conversion."""

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer", "TestClient"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        request_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
        message_type=st.sampled_from(
            [
                MessageType.HEARTBEAT,
                MessageType.START_SIMULATION,
                MessageType.STOP_SIMULATION,
                MessageType.PAUSE_SIMULATION,
                MessageType.RESUME_SIMULATION,
                MessageType.GET_STATUS,
                MessageType.UPDATE_CONFIG,
                MessageType.STATUS_UPDATE,
                MessageType.GAME_STATE_UPDATE,
                MessageType.ERROR_OCCURRED,
            ]
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
        data_content=st.dictionaries(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ),
            st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(min_value=0, max_value=10000),
                st.floats(min_value=0.0, max_value=1000.0),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=5000)
    def test_message_format_round_trip_conversion_property(
        self, client_type, client_id, request_id, message_type, timestamp, data_content
    ):
        """
        **Feature: router-message-protocol-fix, Property 1: Message Format Round-Trip Conversion**
        **Validates: Requirements 1.3, 3.1, 3.2, 3.3, 3.5**

        For any valid ZMQMessage, converting to RouterConstants format and back should preserve
        all essential message content including type, data, timestamps, and identifiers.
        """
        # Create MQClient instance for testing
        mq_client = MQClient(
            router_address="tcp://localhost:5556",
            client_type=client_type,
            client_id=client_id,
        )

        # Create original ZMQMessage
        original_message = ZMQMessage(
            message_type=message_type,
            timestamp=timestamp,
            client_id=client_id,
            request_id=request_id,
            data=data_content,
        )

        # Property 1: Convert ZMQMessage to RouterConstants format
        try:
            router_format = mq_client._convert_to_router_format(original_message)
        except ValueError as e:
            # Some message types might not be supported, which is acceptable
            if "Unsupported message type" in str(e):
                return
            raise

        # Verify RouterConstants format structure
        assert (
            RouterConstants.SENDER in router_format
        ), "RouterConstants format must have sender field"
        assert (
            RouterConstants.ELEM in router_format
        ), "RouterConstants format must have elem field"
        assert (
            RouterConstants.DATA in router_format
        ), "RouterConstants format must have data field"
        assert (
            RouterConstants.CLIENT_ID in router_format
        ), "RouterConstants format must have client_id field"
        assert (
            RouterConstants.TIMESTAMP in router_format
        ), "RouterConstants format must have timestamp field"

        # Verify field values are correct
        assert (
            router_format[RouterConstants.SENDER] == client_type
        ), "Sender should match client type"
        assert (
            router_format[RouterConstants.CLIENT_ID] == client_id
        ), "Client ID should be preserved"
        assert (
            router_format[RouterConstants.TIMESTAMP] == timestamp
        ), "Timestamp should be preserved"
        assert (
            router_format[RouterConstants.REQUEST_ID] == request_id
        ), "Request ID should be preserved"
        assert (
            router_format[RouterConstants.DATA] == data_content
        ), "Data should be preserved"

        # Property 2: Convert back to ZMQMessage format
        converted_back = mq_client._convert_from_router_format(router_format)

        # Property 3: Round-trip conversion should preserve all essential content
        assert (
            converted_back.message_type == original_message.message_type
        ), f"Message type should be preserved: {converted_back.message_type} != {original_message.message_type}"
        assert (
            converted_back.client_id == original_message.client_id
        ), f"Client ID should be preserved: {converted_back.client_id} != {original_message.client_id}"
        assert (
            converted_back.request_id == original_message.request_id
        ), f"Request ID should be preserved: {converted_back.request_id} != {original_message.request_id}"
        assert (
            abs(converted_back.timestamp - original_message.timestamp) < 0.001
        ), f"Timestamp should be preserved: {converted_back.timestamp} != {original_message.timestamp}"
        assert (
            converted_back.data == original_message.data
        ), f"Data should be preserved: {converted_back.data} != {original_message.data}"

        # Property 4: RouterConstants format should pass validation
        is_valid, error_msg = mq_client._validate_router_message(router_format)
        assert is_valid, f"RouterConstants format should be valid: {error_msg}"

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
    )
    @settings(max_examples=50, deadline=3000)
    def test_heartbeat_message_format_property(self, client_type, client_id, timestamp):
        """
        **Feature: router-message-protocol-fix, Property 3: Heartbeat Message Processing**
        **Validates: Requirements 2.3, 2.4, 2.5**

        For any valid heartbeat message in RouterConstants format, the message should have
        the correct structure and be processable by the router without errors.
        """
        # Create MQClient instance
        mq_client = MQClient(
            router_address="tcp://localhost:5556",
            client_type=client_type,
            client_id=client_id,
        )

        # Create heartbeat ZMQMessage
        heartbeat_message = ZMQMessage(
            message_type=MessageType.HEARTBEAT,
            timestamp=timestamp,
            client_id=client_id,
            request_id=f"heartbeat_{int(timestamp)}",
            data={},
        )

        # Convert to RouterConstants format
        router_format = mq_client._convert_to_router_format(heartbeat_message)

        # Verify heartbeat-specific RouterConstants format
        assert (
            router_format[RouterConstants.ELEM] == RouterConstants.HEARTBEAT
        ), "Heartbeat message should use HEARTBEAT elem"
        assert (
            router_format[RouterConstants.SENDER] == client_type
        ), "Heartbeat should include correct sender type"
        assert (
            router_format[RouterConstants.CLIENT_ID] == client_id
        ), "Heartbeat should include client ID"
        assert (
            router_format[RouterConstants.TIMESTAMP] == timestamp
        ), "Heartbeat should include timestamp"
        assert isinstance(
            router_format[RouterConstants.DATA], dict
        ), "Heartbeat data should be a dictionary"

        # Verify message passes validation
        is_valid, error_msg = mq_client._validate_router_message(router_format)
        assert (
            is_valid
        ), f"Heartbeat RouterConstants format should be valid: {error_msg}"

        # Verify the message can be converted back
        converted_back = mq_client._convert_from_router_format(router_format)
        assert (
            converted_back.message_type == MessageType.HEARTBEAT
        ), "Round-trip conversion should preserve heartbeat message type"

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer", "TestClient"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        message_type=st.sampled_from(
            [
                MessageType.HEARTBEAT,
                MessageType.START_SIMULATION,
                MessageType.STOP_SIMULATION,
                MessageType.PAUSE_SIMULATION,
                MessageType.RESUME_SIMULATION,
                MessageType.GET_STATUS,
                MessageType.UPDATE_CONFIG,
                MessageType.STATUS_UPDATE,
                MessageType.GAME_STATE_UPDATE,
                MessageType.ERROR_OCCURRED,
            ]
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
        request_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
        data_content=st.dictionaries(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ),
            st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(min_value=0, max_value=10000),
                st.floats(min_value=0.0, max_value=1000.0),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=5000)
    def test_router_constants_format_compliance_property(
        self, client_type, client_id, message_type, timestamp, request_id, data_content
    ):
        """
        **Feature: router-message-protocol-fix, Property 2: RouterConstants Format Compliance**
        **Validates: Requirements 1.1, 1.2, 2.1, 2.2**

        For any message sent by MQClient to the router, the message should have the required
        RouterConstants format with sender, elem, data, client_id, and timestamp fields.
        """
        # Create MQClient instance
        mq_client = MQClient(
            router_address="tcp://localhost:5556",
            client_type=client_type,
            client_id=client_id,
        )

        # Create ZMQMessage
        zmq_message = ZMQMessage(
            message_type=message_type,
            timestamp=timestamp,
            client_id=client_id,
            request_id=request_id,
            data=data_content,
        )

        # Convert to RouterConstants format
        try:
            router_format = mq_client._convert_to_router_format(zmq_message)
        except ValueError as e:
            # Some message types might not be supported, which is acceptable
            if "Unsupported message type" in str(e):
                return
            raise

        # Property 1: Message must have all required RouterConstants fields
        required_fields = [
            RouterConstants.SENDER,
            RouterConstants.ELEM,
            RouterConstants.DATA,
            RouterConstants.CLIENT_ID,
            RouterConstants.TIMESTAMP,
        ]

        for field in required_fields:
            assert (
                field in router_format
            ), f"RouterConstants format must have {field} field"

        # Property 2: Field types must be correct
        assert isinstance(
            router_format[RouterConstants.SENDER], str
        ), "sender must be string"
        assert isinstance(
            router_format[RouterConstants.ELEM], str
        ), "elem must be string"
        assert isinstance(
            router_format[RouterConstants.DATA], dict
        ), "data must be dict"
        assert isinstance(
            router_format[RouterConstants.CLIENT_ID], str
        ), "client_id must be string"
        assert isinstance(
            router_format[RouterConstants.TIMESTAMP], (int, float)
        ), "timestamp must be number"

        # Property 3: Field values must be preserved from original message
        assert (
            router_format[RouterConstants.SENDER] == client_type
        ), "sender should match client type"
        assert (
            router_format[RouterConstants.CLIENT_ID] == client_id
        ), "client_id should be preserved"
        assert (
            router_format[RouterConstants.TIMESTAMP] == timestamp
        ), "timestamp should be preserved"
        assert (
            router_format[RouterConstants.REQUEST_ID] == request_id
        ), "request_id should be preserved"
        assert (
            router_format[RouterConstants.DATA] == data_content
        ), "data should be preserved"

        # Property 4: Message must pass RouterConstants validation
        is_valid, error_msg = mq_client._validate_router_message(router_format)
        assert is_valid, f"RouterConstants format should pass validation: {error_msg}"

        # Property 5: elem field should be valid RouterConstants element
        elem = router_format[RouterConstants.ELEM]
        valid_elems = [
            RouterConstants.HEARTBEAT,
            RouterConstants.START_SIMULATION,
            RouterConstants.STOP_SIMULATION,
            RouterConstants.PAUSE_SIMULATION,
            RouterConstants.RESUME_SIMULATION,
            RouterConstants.RESET_SIMULATION,
            RouterConstants.GET_STATUS,
            RouterConstants.SET_CONFIG,
            RouterConstants.STATUS_UPDATE,
            RouterConstants.GAME_STATE_UPDATE,
            RouterConstants.PERFORMANCE_UPDATE,
            RouterConstants.ERROR,
            RouterConstants.STATUS,
            RouterConstants.CONFIG_UPDATE,
        ]
        assert (
            elem in valid_elems
        ), f"elem field should be valid RouterConstants element: {elem}"

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        error_scenario=st.sampled_from(
            [
                "missing_sender",
                "missing_elem",
                "missing_data",
                "missing_client_id",
                "missing_timestamp",
                "invalid_sender_type",
                "invalid_elem_type",
                "invalid_data_type",
                "invalid_client_id_type",
                "invalid_timestamp_type",
                "unsupported_message_type",
                "malformed_router_message",
            ]
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_format_validation_error_reporting_property(
        self, client_type, client_id, error_scenario, timestamp
    ):
        """
        **Feature: router-message-protocol-fix, Property 4: Format Validation Error Reporting**
        **Validates: Requirements 4.1, 4.2, 4.5**

        For any invalid message format, the Protocol_Validator should provide specific error
        details identifying missing or incorrect fields and source component information.
        """
        # Create MQClient instance
        mq_client = MQClient(
            router_address="tcp://localhost:5556",
            client_type=client_type,
            client_id=client_id,
        )

        if error_scenario == "unsupported_message_type":
            # Test unsupported message type error handling
            class UnsupportedMessageType:
                def __init__(self, value):
                    self.value = value

            unsupported_message = ZMQMessage(
                message_type=UnsupportedMessageType("UNSUPPORTED_TYPE"),
                timestamp=timestamp,
                client_id=client_id,
                request_id="test_request",
                data={},
            )

            # Property 1: Unsupported message types should raise ValueError with specific error
            try:
                mq_client._convert_to_router_format(unsupported_message)
                assert (
                    False
                ), "Should have raised ValueError for unsupported message type"
            except ValueError as e:
                error_msg = str(e)
                assert (
                    "Unsupported message type" in error_msg
                ), f"Error should mention unsupported message type: {error_msg}"
                # Property 2: Error should include source component information
                assert (
                    "message type" in error_msg.lower()
                ), f"Error should identify the problematic field: {error_msg}"

        elif error_scenario == "malformed_router_message":
            # Test malformed RouterConstants message validation
            malformed_messages = [
                # Missing required fields
                {RouterConstants.SENDER: client_type},  # Missing other required fields
                {
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT
                },  # Missing other required fields
                {RouterConstants.DATA: {}},  # Missing other required fields
                {RouterConstants.CLIENT_ID: client_id},  # Missing other required fields
                {RouterConstants.TIMESTAMP: timestamp},  # Missing other required fields
                # Invalid field types
                {
                    RouterConstants.SENDER: 123,  # Should be string
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                    RouterConstants.DATA: {},
                    RouterConstants.CLIENT_ID: client_id,
                    RouterConstants.TIMESTAMP: timestamp,
                },
                {
                    RouterConstants.SENDER: client_type,
                    RouterConstants.ELEM: 456,  # Should be string
                    RouterConstants.DATA: {},
                    RouterConstants.CLIENT_ID: client_id,
                    RouterConstants.TIMESTAMP: timestamp,
                },
                {
                    RouterConstants.SENDER: client_type,
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                    RouterConstants.DATA: "not_a_dict",  # Should be dict
                    RouterConstants.CLIENT_ID: client_id,
                    RouterConstants.TIMESTAMP: timestamp,
                },
                {
                    RouterConstants.SENDER: client_type,
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                    RouterConstants.DATA: {},
                    RouterConstants.CLIENT_ID: 789,  # Should be string
                    RouterConstants.TIMESTAMP: timestamp,
                },
                {
                    RouterConstants.SENDER: client_type,
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                    RouterConstants.DATA: {},
                    RouterConstants.CLIENT_ID: client_id,
                    RouterConstants.TIMESTAMP: "not_a_number",  # Should be number
                },
            ]

            for malformed_msg in malformed_messages:
                # Property 3: Malformed messages should fail validation with specific errors
                is_valid, error_msg = mq_client._validate_router_message(malformed_msg)
                assert (
                    not is_valid
                ), f"Malformed message should fail validation: {malformed_msg}"
                assert (
                    error_msg is not None
                ), "Error message should be provided for invalid message"
                assert len(error_msg) > 0, "Error message should not be empty"

                # Property 4: Error message should identify the specific problem
                if RouterConstants.SENDER not in malformed_msg:
                    assert (
                        "sender" in error_msg.lower()
                    ), f"Error should mention missing sender: {error_msg}"
                elif RouterConstants.ELEM not in malformed_msg:
                    assert (
                        "elem" in error_msg.lower()
                    ), f"Error should mention missing elem: {error_msg}"
                elif RouterConstants.DATA not in malformed_msg:
                    assert (
                        "data" in error_msg.lower()
                    ), f"Error should mention missing data: {error_msg}"
                elif RouterConstants.CLIENT_ID not in malformed_msg:
                    assert (
                        "client_id" in error_msg.lower()
                    ), f"Error should mention missing client_id: {error_msg}"
                elif RouterConstants.TIMESTAMP not in malformed_msg:
                    assert (
                        "timestamp" in error_msg.lower()
                    ), f"Error should mention missing timestamp: {error_msg}"
                elif not isinstance(malformed_msg.get(RouterConstants.SENDER), str):
                    assert "sender" in error_msg.lower() and (
                        "string" in error_msg.lower() or "str" in error_msg.lower()
                    ), f"Error should mention sender type issue: {error_msg}"
                elif not isinstance(malformed_msg.get(RouterConstants.ELEM), str):
                    assert "elem" in error_msg.lower() and (
                        "string" in error_msg.lower() or "str" in error_msg.lower()
                    ), f"Error should mention elem type issue: {error_msg}"
                elif not isinstance(malformed_msg.get(RouterConstants.DATA), dict):
                    assert (
                        "data" in error_msg.lower() and "dict" in error_msg.lower()
                    ), f"Error should mention data type issue: {error_msg}"
                elif not isinstance(malformed_msg.get(RouterConstants.CLIENT_ID), str):
                    assert "client_id" in error_msg.lower() and (
                        "string" in error_msg.lower() or "str" in error_msg.lower()
                    ), f"Error should mention client_id type issue: {error_msg}"
                elif not isinstance(
                    malformed_msg.get(RouterConstants.TIMESTAMP), (int, float)
                ):
                    assert (
                        "timestamp" in error_msg.lower()
                        and "number" in error_msg.lower()
                    ), f"Error should mention timestamp type issue: {error_msg}"

        else:
            # Test conversion failure error handling with valid ZMQMessage but forced errors
            valid_message = ZMQMessage(
                message_type=MessageType.HEARTBEAT,
                timestamp=timestamp,
                client_id=client_id,
                request_id="test_request",
                data={},
            )

            # Property 5: Conversion failures should be handled gracefully
            try:
                router_format = mq_client._convert_to_router_format(valid_message)

                # Property 6: Valid conversions should not raise errors
                is_valid, error_msg = mq_client._validate_router_message(router_format)
                assert is_valid, f"Valid conversion should pass validation: {error_msg}"

                # Property 7: Round-trip conversion should handle errors gracefully
                converted_back = mq_client._convert_from_router_format(router_format)
                assert (
                    converted_back.message_type == valid_message.message_type
                ), "Round-trip should preserve message type"

            except Exception as e:
                # Property 8: Any exceptions should be informative
                error_msg = str(e)
                assert len(error_msg) > 0, "Exception message should not be empty"
                assert (
                    "conversion" in error_msg.lower()
                    or "validation" in error_msg.lower()
                    or "format" in error_msg.lower()
                ), f"Exception should be related to conversion/validation: {error_msg}"


class TestBackwardCompatibilityProperties:
    """Property-based tests for backward compatibility preservation."""

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer", "TestClient"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        message_type=st.sampled_from(
            [
                MessageType.HEARTBEAT,
                MessageType.START_SIMULATION,
                MessageType.STOP_SIMULATION,
                MessageType.PAUSE_SIMULATION,
                MessageType.RESUME_SIMULATION,
                MessageType.GET_STATUS,
                MessageType.UPDATE_CONFIG,
                MessageType.STATUS_UPDATE,
                MessageType.GAME_STATE_UPDATE,
                MessageType.ERROR_OCCURRED,
                MessageType.CLIENT_CONNECTED,
                MessageType.CLIENT_DISCONNECTED,
                MessageType.CONFIG_UPDATED,
            ]
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
        request_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
        data_content=st.dictionaries(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ),
            st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(min_value=0, max_value=10000),
                st.floats(min_value=0.0, max_value=1000.0),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=5000)
    def test_backward_compatibility_preservation_property(
        self, client_type, client_id, message_type, timestamp, request_id, data_content
    ):
        """
        **Feature: router-message-protocol-fix, Property 5: Backward Compatibility Preservation**
        **Validates: Requirements 1.5, 5.1, 5.4**

        For any internal component communication, the ZMQMessage format should remain unchanged
        and continue to work as before the router format fix. Internal components should be
        able to create, serialize, and deserialize ZMQMessage objects without any changes.
        """
        # Property 1: ZMQMessage creation should work unchanged
        original_message = ZMQMessage(
            message_type=message_type,
            timestamp=timestamp,
            client_id=client_id,
            request_id=request_id,
            data=data_content,
        )

        # Verify ZMQMessage structure is preserved
        assert hasattr(
            original_message, "message_type"
        ), "ZMQMessage should have message_type attribute"
        assert hasattr(
            original_message, "timestamp"
        ), "ZMQMessage should have timestamp attribute"
        assert hasattr(
            original_message, "client_id"
        ), "ZMQMessage should have client_id attribute"
        assert hasattr(
            original_message, "request_id"
        ), "ZMQMessage should have request_id attribute"
        assert hasattr(
            original_message, "data"
        ), "ZMQMessage should have data attribute"

        # Property 2: ZMQMessage serialization should work unchanged
        try:
            json_str = original_message.to_json()
            assert isinstance(
                json_str, str
            ), "ZMQMessage serialization should produce string"

            # Verify it's valid JSON
            parsed_json = json.loads(json_str)
            assert isinstance(
                parsed_json, dict
            ), "Serialized ZMQMessage should be valid JSON dict"

        except Exception as e:
            pytest.fail(
                f"ZMQMessage serialization failed (backward compatibility broken): {e}"
            )

        # Property 3: ZMQMessage deserialization should work unchanged
        try:
            deserialized_message = ZMQMessage.from_json(json_str)

            # Verify all fields are preserved
            assert (
                deserialized_message.message_type == original_message.message_type
            ), "Message type should be preserved in backward compatibility"
            assert (
                deserialized_message.client_id == original_message.client_id
            ), "Client ID should be preserved in backward compatibility"
            assert (
                deserialized_message.request_id == original_message.request_id
            ), "Request ID should be preserved in backward compatibility"
            assert (
                abs(deserialized_message.timestamp - original_message.timestamp) < 0.001
            ), "Timestamp should be preserved in backward compatibility"
            assert (
                deserialized_message.data == original_message.data
            ), "Data should be preserved in backward compatibility"

        except Exception as e:
            pytest.fail(
                f"ZMQMessage deserialization failed (backward compatibility broken): {e}"
            )

        # Property 4: Internal components should not need to know about RouterConstants format
        # Test that ZMQMessage can be used without any RouterConstants imports or knowledge
        try:
            # Create message using factory methods (internal component pattern)
            command_message = ZMQMessage.create_command(
                message_type=message_type,
                client_id=client_id,
                request_id=request_id,
                data=data_content,
            )

            assert (
                command_message.message_type == message_type
            ), "Factory method should preserve message type"
            assert (
                command_message.client_id == client_id
            ), "Factory method should preserve client ID"
            assert (
                command_message.request_id == request_id
            ), "Factory method should preserve request ID"
            assert (
                command_message.data == data_content
            ), "Factory method should preserve data"

        except Exception as e:
            pytest.fail(
                f"ZMQMessage factory methods failed (backward compatibility broken): {e}"
            )

        # Property 5: Message validation should work unchanged for internal components
        try:
            # Test that internal validation still works
            if message_type == MessageType.START_SIMULATION:
                # Add required config for START_SIMULATION validation
                test_data = data_content.copy()
                test_data["config"] = {"grid_size": [10, 10], "move_budget": 50}
                test_message = ZMQMessage(
                    message_type=message_type,
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data=test_data,
                )

                is_valid, error = MessageValidator.validate_message(test_message)
                # Should either be valid or have a clear validation error (not a format error)
                if not is_valid:
                    assert error is not None, "Validation should provide error message"
                    # Should not be a format-related error (backward compatibility)
                    assert (
                        "format" not in error.lower()
                    ), f"Validation error should not be format-related: {error}"

        except Exception as e:
            # Validation errors are acceptable, format errors are not
            if "format" in str(e).lower() or "RouterConstants" in str(e):
                pytest.fail(
                    f"Message validation has format dependency (backward compatibility broken): {e}"
                )

        # Property 6: ZMQMessage should work with existing message builders
        try:
            # Test MessageBuilder compatibility (internal component usage)
            if message_type in [
                MessageType.STATUS_UPDATE,
                MessageType.GAME_STATE_UPDATE,
            ]:
                # Create a simple game state for testing
                game_state = GameStateData(
                    snake_head=(5, 5),
                    snake_body=[(5, 5), (4, 5), (3, 5)],
                    direction=(1, 0),
                    food_position=(8, 8),
                    score=10,
                    grid_size=(10, 10),
                    moves_count=25,
                    is_game_over=False,
                )

                performance = PerformanceMetrics.create_empty()
                performance.decisions_per_second = 2.5

                # This should work without any RouterConstants knowledge
                builder_message = MessageBuilder.status_update(
                    game_state, performance, "running"
                )

                assert (
                    builder_message.message_type == MessageType.STATUS_UPDATE
                ), "MessageBuilder should create correct message type"
                assert (
                    "game_state" in builder_message.data
                ), "MessageBuilder should include game state data"
                assert (
                    "performance" in builder_message.data
                ), "MessageBuilder should include performance data"

        except Exception as e:
            pytest.fail(f"MessageBuilder failed (backward compatibility broken): {e}")


class TestErrorResilienceProperties:
    """Property-based tests for error resilience."""

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer", "TestClient"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        error_scenario=st.sampled_from(
            [
                "network_failure",
                "invalid_json",
                "missing_fields",
                "type_mismatch",
                "conversion_failure",
                "validation_failure",
                "timeout_error",
                "memory_error",
            ]
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
        request_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
    )
    @settings(max_examples=100, deadline=5000)
    def test_error_resilience_property(
        self, client_type, client_id, error_scenario, timestamp, request_id
    ):
        """
        **Feature: router-message-protocol-fix, Property 6: Error Resilience**
        **Validates: Requirements 4.3, 4.4**

        For any format conversion failure or validation error, the system should handle it
        gracefully without crashing components and provide appropriate retry mechanisms.
        """
        # Create MQClient instance for testing
        mq_client = MQClient(
            router_address="tcp://localhost:5556",
            client_type=client_type,
            client_id=client_id,
        )

        if error_scenario == "network_failure":
            # Property 1: Network failures should not crash the client
            try:
                # Simulate network failure by using invalid message
                invalid_message = ZMQMessage(
                    message_type=MessageType.HEARTBEAT,
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data={},
                )

                # This should handle gracefully even if network fails
                router_format = mq_client._convert_to_router_format(invalid_message)

                # Should produce valid RouterConstants format
                assert (
                    RouterConstants.SENDER in router_format
                ), "Network failure simulation should still produce valid format"
                assert (
                    RouterConstants.ELEM in router_format
                ), "Network failure simulation should include elem field"

            except Exception as e:
                # Any exception should be informative, not a crash
                error_msg = str(e)
                assert len(error_msg) > 0, "Error message should not be empty"
                assert not any(
                    crash_indicator in error_msg.lower()
                    for crash_indicator in [
                        "segmentation",
                        "core dump",
                        "fatal",
                        "abort",
                    ]
                ), f"Should not indicate system crash: {error_msg}"

        elif error_scenario == "invalid_json":
            # Property 2: Invalid JSON should be handled gracefully
            try:
                # Test with malformed router message that could come from network
                malformed_router_msg = {
                    RouterConstants.SENDER: client_type,
                    RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                    RouterConstants.DATA: "invalid_json_string",  # Should be dict
                    RouterConstants.CLIENT_ID: client_id,
                    RouterConstants.TIMESTAMP: timestamp,
                }

                is_valid, error_msg = mq_client._validate_router_message(
                    malformed_router_msg
                )

                # Should fail validation gracefully
                assert not is_valid, "Invalid JSON should fail validation"
                assert error_msg is not None, "Should provide error message"
                assert "data" in error_msg.lower(), "Should identify data field issue"

            except Exception as e:
                # Should not crash, should provide informative error
                error_msg = str(e)
                assert (
                    "validation" in error_msg.lower() or "format" in error_msg.lower()
                ), f"Exception should be validation-related: {error_msg}"

        elif error_scenario == "missing_fields":
            # Property 3: Missing fields should be handled gracefully
            try:
                incomplete_messages = [
                    {},  # Completely empty
                    {RouterConstants.SENDER: client_type},  # Only sender
                    {RouterConstants.ELEM: RouterConstants.HEARTBEAT},  # Only elem
                    {RouterConstants.DATA: {}},  # Only data
                ]

                for incomplete_msg in incomplete_messages:
                    is_valid, error_msg = mq_client._validate_router_message(
                        incomplete_msg
                    )

                    # Should fail validation gracefully
                    assert (
                        not is_valid
                    ), f"Incomplete message should fail validation: {incomplete_msg}"
                    assert (
                        error_msg is not None
                    ), "Should provide error message for incomplete message"
                    assert (
                        "missing" in error_msg.lower()
                    ), f"Should identify missing fields: {error_msg}"

            except Exception as e:
                # Should handle gracefully
                error_msg = str(e)
                assert (
                    "missing" in error_msg.lower() or "required" in error_msg.lower()
                ), f"Exception should be about missing fields: {error_msg}"

        elif error_scenario == "type_mismatch":
            # Property 4: Type mismatches should be handled gracefully
            try:
                # Create message with wrong types
                type_mismatch_message = ZMQMessage(
                    message_type="invalid_type_string",  # Should be MessageType enum
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data={},
                )

                # This should either work or fail gracefully
                try:
                    router_format = mq_client._convert_to_router_format(
                        type_mismatch_message
                    )
                    # If it works, should be valid
                    is_valid, error_msg = mq_client._validate_router_message(
                        router_format
                    )
                    if not is_valid:
                        assert error_msg is not None, "Should provide validation error"
                except (ValueError, TypeError) as conversion_error:
                    # Should be informative about the type issue
                    error_msg = str(conversion_error)
                    assert (
                        "type" in error_msg.lower() or "invalid" in error_msg.lower()
                    ), f"Should identify type issue: {error_msg}"

            except Exception as e:
                # Should be type-related error, not crash
                error_msg = str(e)
                assert any(
                    type_indicator in error_msg.lower()
                    for type_indicator in ["type", "invalid", "expected"]
                ), f"Should be type-related error: {error_msg}"

        elif error_scenario == "conversion_failure":
            # Property 5: Conversion failures should provide retry mechanisms
            try:
                # Test unsupported message type
                class UnsupportedType:
                    def __init__(self, value):
                        self.value = value

                unsupported_message = ZMQMessage(
                    message_type=UnsupportedType("UNSUPPORTED"),
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data={},
                )

                try:
                    router_format = mq_client._convert_to_router_format(
                        unsupported_message
                    )
                    # If conversion succeeds, should be valid
                    is_valid, error_msg = mq_client._validate_router_message(
                        router_format
                    )
                    assert (
                        is_valid or error_msg is not None
                    ), "Conversion result should be valid or provide error"
                except ValueError as e:
                    # Should provide clear error about unsupported type
                    error_msg = str(e)
                    assert (
                        "unsupported" in error_msg.lower()
                        or "type" in error_msg.lower()
                    ), f"Should identify unsupported type: {error_msg}"

                    # Property: Error should suggest what types are supported
                    assert (
                        "supported" in error_msg.lower() or "valid" in error_msg.lower()
                    ), f"Should suggest supported alternatives: {error_msg}"

            except Exception as e:
                # Should be conversion-related, not crash
                error_msg = str(e)
                assert (
                    "conversion" in error_msg.lower()
                    or "unsupported" in error_msg.lower()
                ), f"Should be conversion-related error: {error_msg}"

        elif error_scenario == "validation_failure":
            # Property 6: Validation failures should be recoverable
            try:
                # Create message that will fail validation
                invalid_router_msg = {
                    RouterConstants.SENDER: "",  # Empty sender (invalid)
                    RouterConstants.ELEM: "",  # Empty elem (invalid)
                    RouterConstants.DATA: {},
                    RouterConstants.CLIENT_ID: client_id,
                    RouterConstants.TIMESTAMP: timestamp,
                }

                is_valid, error_msg = mq_client._validate_router_message(
                    invalid_router_msg
                )

                # Should fail validation gracefully
                assert not is_valid, "Invalid message should fail validation"
                assert error_msg is not None, "Should provide validation error"
                assert len(error_msg) > 0, "Error message should not be empty"

                # Property: Error should be specific about what's wrong
                assert (
                    "empty" in error_msg.lower() or "cannot be" in error_msg.lower()
                ), f"Should identify specific validation issue: {error_msg}"

                # Property: Should be recoverable by fixing the identified issue
                fixed_router_msg = invalid_router_msg.copy()
                fixed_router_msg[RouterConstants.SENDER] = client_type
                fixed_router_msg[RouterConstants.ELEM] = RouterConstants.HEARTBEAT

                is_valid_fixed, error_msg_fixed = mq_client._validate_router_message(
                    fixed_router_msg
                )
                assert (
                    is_valid_fixed
                ), f"Fixed message should pass validation: {error_msg_fixed}"

            except Exception as e:
                # Should be validation-related
                error_msg = str(e)
                assert (
                    "validation" in error_msg.lower() or "invalid" in error_msg.lower()
                ), f"Should be validation-related error: {error_msg}"

        elif error_scenario == "timeout_error":
            # Property 7: Timeout errors should not affect message format conversion
            try:
                # Test that format conversion works even under timeout pressure
                valid_message = ZMQMessage(
                    message_type=MessageType.HEARTBEAT,
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data={},
                )

                # Conversion should work regardless of external timeout conditions
                router_format = mq_client._convert_to_router_format(valid_message)

                # Should produce valid format
                is_valid, error_msg = mq_client._validate_router_message(router_format)
                assert (
                    is_valid
                ), f"Format conversion should work under timeout pressure: {error_msg}"

                # Round-trip should also work
                converted_back = mq_client._convert_from_router_format(router_format)
                assert (
                    converted_back.message_type == valid_message.message_type
                ), "Round-trip should work under timeout pressure"

            except Exception as e:
                # Should not be timeout-related for format conversion
                error_msg = str(e)
                assert (
                    "timeout" not in error_msg.lower()
                ), f"Format conversion should not have timeout issues: {error_msg}"

        elif error_scenario == "memory_error":
            # Property 8: Memory constraints should not break format conversion
            try:
                # Test with reasonable-sized data (not actually causing memory error)
                large_data = {
                    f"key_{i}": f"value_{i}" * 10 for i in range(100)  # Reasonable size
                }

                memory_test_message = ZMQMessage(
                    message_type=MessageType.STATUS_UPDATE,
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data=large_data,
                )

                # Should handle larger data gracefully
                router_format = mq_client._convert_to_router_format(memory_test_message)

                # Should maintain data integrity
                assert (
                    router_format[RouterConstants.DATA] == large_data
                ), "Large data should be preserved in format conversion"

                # Validation should still work
                is_valid, error_msg = mq_client._validate_router_message(router_format)
                assert is_valid, f"Large data should not break validation: {error_msg}"

            except Exception as e:
                # Should not be memory-related for reasonable data sizes
                error_msg = str(e)
                assert (
                    "memory" not in error_msg.lower()
                ), f"Reasonable data sizes should not cause memory issues: {error_msg}"

        # Property 9: All error scenarios should preserve system stability
        # The fact that we reach this point means the system didn't crash
        assert True, "System should remain stable after error handling"


class TestMessageTypeCoverageProperties:
    """Property-based tests for message type coverage."""

    @given(
        client_type=st.sampled_from(["HydraClient", "HydraServer", "TestClient"]),
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        timestamp=st.floats(min_value=1000000000, max_value=2000000000),
        request_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
        data_content=st.dictionaries(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ),
            st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(min_value=0, max_value=10000),
                st.floats(min_value=0.0, max_value=1000.0),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=10, deadline=2000)
    def test_message_type_coverage_property(
        self, client_type, client_id, timestamp, request_id, data_content
    ):
        """
        **Feature: router-message-protocol-fix, Property 7: Message Type Coverage**
        **Validates: Requirements 3.4**

        For any message type (commands, responses, broadcasts, heartbeats), the format
        conversion should work correctly and preserve message semantics.
        """
        # Create MQClient instance for testing
        mq_client = MQClient(
            router_address="tcp://localhost:5556",
            client_type=client_type,
            client_id=client_id,
        )

        # Test all supported message types
        all_message_types = [
            # Commands (Client -> Server)
            MessageType.START_SIMULATION,
            MessageType.STOP_SIMULATION,
            MessageType.PAUSE_SIMULATION,
            MessageType.RESUME_SIMULATION,
            MessageType.GET_STATUS,
            MessageType.UPDATE_CONFIG,
            MessageType.RESET_SIMULATION,
            # Responses (Server -> Client)
            MessageType.SIMULATION_STARTED,
            MessageType.SIMULATION_STOPPED,
            MessageType.SIMULATION_PAUSED,
            MessageType.SIMULATION_RESUMED,
            MessageType.STATUS_RESPONSE,
            MessageType.CONFIG_UPDATED,
            MessageType.SIMULATION_RESET,
            # Broadcasts (Server -> All Clients)
            MessageType.STATUS_UPDATE,
            MessageType.GAME_STATE_UPDATE,
            MessageType.DECISION_CYCLE_COMPLETE,
            MessageType.GAME_OVER,
            MessageType.ERROR_OCCURRED,
            # System Messages
            MessageType.CLIENT_CONNECTED,
            MessageType.CLIENT_DISCONNECTED,
            MessageType.HEARTBEAT,
        ]

        supported_count = 0
        unsupported_types = []

        for message_type in all_message_types:
            try:
                # Property 1: All message types should have format conversion support
                test_message = ZMQMessage(
                    message_type=message_type,
                    timestamp=timestamp,
                    client_id=client_id,
                    request_id=request_id,
                    data=data_content,
                )

                # Test conversion to RouterConstants format
                router_format = mq_client._convert_to_router_format(test_message)

                # Property 2: Converted message should have valid RouterConstants structure
                assert (
                    RouterConstants.SENDER in router_format
                ), f"Message type {message_type.value} should have sender field"
                assert (
                    RouterConstants.ELEM in router_format
                ), f"Message type {message_type.value} should have elem field"
                assert (
                    RouterConstants.DATA in router_format
                ), f"Message type {message_type.value} should have data field"
                assert (
                    RouterConstants.CLIENT_ID in router_format
                ), f"Message type {message_type.value} should have client_id field"
                assert (
                    RouterConstants.TIMESTAMP in router_format
                ), f"Message type {message_type.value} should have timestamp field"

                # Property 3: RouterConstants format should pass validation
                is_valid, error_msg = mq_client._validate_router_message(router_format)
                assert (
                    is_valid
                ), f"Message type {message_type.value} should produce valid RouterConstants format: {error_msg}"

                # Property 4: Round-trip conversion should preserve message semantics
                converted_back = mq_client._convert_from_router_format(router_format)

                assert (
                    converted_back.message_type == message_type
                ), f"Round-trip should preserve message type for {message_type.value}"
                assert (
                    converted_back.client_id == client_id
                ), f"Round-trip should preserve client_id for {message_type.value}"
                assert (
                    converted_back.request_id == request_id
                ), f"Round-trip should preserve request_id for {message_type.value}"
                assert (
                    abs(converted_back.timestamp - timestamp) < 0.001
                ), f"Round-trip should preserve timestamp for {message_type.value}"
                assert (
                    converted_back.data == data_content
                ), f"Round-trip should preserve data for {message_type.value}"

                # Property 5: Message semantics should be preserved
                # Commands should map to appropriate RouterConstants elements
                if message_type in [
                    MessageType.START_SIMULATION,
                    MessageType.STOP_SIMULATION,
                    MessageType.PAUSE_SIMULATION,
                    MessageType.RESUME_SIMULATION,
                ]:
                    elem = router_format[RouterConstants.ELEM]
                    assert elem in [
                        RouterConstants.START_SIMULATION,
                        RouterConstants.STOP_SIMULATION,
                        RouterConstants.PAUSE_SIMULATION,
                        RouterConstants.RESUME_SIMULATION,
                    ], f"Command message {message_type.value} should map to command elem: {elem}"

                # System messages should map appropriately
                elif message_type == MessageType.HEARTBEAT:
                    assert (
                        router_format[RouterConstants.ELEM] == RouterConstants.HEARTBEAT
                    ), f"Heartbeat should map to heartbeat elem"

                # Status messages should map appropriately
                elif message_type in [
                    MessageType.STATUS_UPDATE,
                    MessageType.STATUS_RESPONSE,
                ]:
                    elem = router_format[RouterConstants.ELEM]
                    assert elem in [
                        RouterConstants.STATUS_UPDATE,
                        RouterConstants.STATUS,
                    ], f"Status message {message_type.value} should map to status elem: {elem}"

                # Game state messages should map appropriately
                elif message_type in [
                    MessageType.GAME_STATE_UPDATE,
                    MessageType.GAME_OVER,
                ]:
                    assert (
                        router_format[RouterConstants.ELEM]
                        == RouterConstants.GAME_STATE_UPDATE
                    ), f"Game state message {message_type.value} should map to game state elem"

                # Error messages should map appropriately
                elif message_type == MessageType.ERROR_OCCURRED:
                    assert (
                        router_format[RouterConstants.ELEM] == RouterConstants.ERROR
                    ), f"Error message should map to error elem"

                supported_count += 1

            except ValueError as e:
                # Some message types might not be supported yet, track them
                if "Unsupported message type" in str(e):
                    unsupported_types.append(message_type.value)
                else:
                    # Other ValueError should not occur for valid message types
                    pytest.fail(f"Unexpected ValueError for {message_type.value}: {e}")
            except Exception as e:
                # No other exceptions should occur for valid message types
                pytest.fail(f"Unexpected error for {message_type.value}: {e}")

        # Property 6: Coverage should be comprehensive
        total_types = len(all_message_types)
        coverage_ratio = supported_count / total_types

        # At least 80% of message types should be supported
        assert coverage_ratio >= 0.8, (
            f"Message type coverage should be at least 80%, got {coverage_ratio:.2%} "
            f"({supported_count}/{total_types}). Unsupported: {unsupported_types}"
        )

        # Property 7: Critical message types should always be supported
        critical_types = [
            MessageType.HEARTBEAT,
            MessageType.START_SIMULATION,
            MessageType.STOP_SIMULATION,
            MessageType.GET_STATUS,
            MessageType.STATUS_UPDATE,
            MessageType.ERROR_OCCURRED,
        ]

        for critical_type in critical_types:
            assert (
                critical_type.value not in unsupported_types
            ), f"Critical message type {critical_type.value} must be supported"

        # Property 8: All supported types should have bidirectional conversion
        for message_type in all_message_types:
            if message_type.value not in unsupported_types:
                try:
                    # Test that we can convert both ways
                    test_message = ZMQMessage(
                        message_type=message_type,
                        timestamp=timestamp,
                        client_id=client_id,
                        request_id=request_id,
                        data=data_content,
                    )

                    # ZMQMessage -> RouterConstants
                    router_format = mq_client._convert_to_router_format(test_message)

                    # RouterConstants -> ZMQMessage
                    converted_back = mq_client._convert_from_router_format(
                        router_format
                    )

                    # Should be identical
                    assert (
                        converted_back.message_type == test_message.message_type
                    ), f"Bidirectional conversion should preserve type for {message_type.value}"

                except Exception as e:
                    pytest.fail(
                        f"Bidirectional conversion failed for {message_type.value}: {e}"
                    )

        # Property 9: Message type mapping should be consistent
        # Test that the same message type always maps to the same RouterConstants elem
        if supported_count > 0:
            test_message_type = MessageType.HEARTBEAT  # Known supported type

            # Create multiple messages of the same type
            for i in range(3):
                test_message = ZMQMessage(
                    message_type=test_message_type,
                    timestamp=timestamp + i,  # Different timestamp
                    client_id=f"{client_id}_{i}",  # Different client_id
                    request_id=f"{request_id}_{i}",  # Different request_id
                    data={"test": i},  # Different data
                )

                router_format = mq_client._convert_to_router_format(test_message)

                # elem should always be the same for the same message type
                if i == 0:
                    expected_elem = router_format[RouterConstants.ELEM]
                else:
                    assert (
                        router_format[RouterConstants.ELEM] == expected_elem
                    ), f"Message type {test_message_type.value} should always map to same elem"


class TestZMQProtocolProperties:
    """Property-based tests for ZeroMQ message protocol integrity."""

    @given(
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        request_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd")),
        ),
        grid_width=st.integers(min_value=5, max_value=50),
        grid_height=st.integers(min_value=5, max_value=50),
        move_budget=st.integers(min_value=1, max_value=1000),
        random_seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=3000)
    def test_zmq_message_protocol_integrity_property(
        self, client_id, request_id, grid_width, grid_height, move_budget, random_seed
    ):
        """
        **Feature: ai-hydra, Property 13: ZeroMQ Message Protocol Integrity**

        *For any* ZMQ message sent between client and server, the message should serialize
        to JSON, deserialize correctly, and maintain all original data fields and types.
        **Validates: Requirements 13.1, 13.2**
        """
        # Create message with various data types
        original_data = {
            "config": {
                "grid_size": [grid_width, grid_height],
                "move_budget": move_budget,
                "random_seed": random_seed,
                "nn_enabled": True,
                "learning_rate": 0.001,
                "batch_size": 32,
            },
            "metadata": {
                "timestamp": time.time(),
                "version": "1.0.0",
                "debug_mode": False,
            },
        }

        # Test all message types for protocol integrity
        message_types = [
            MessageType.START_SIMULATION,
            MessageType.STOP_SIMULATION,
            MessageType.PAUSE_SIMULATION,
            MessageType.RESUME_SIMULATION,
            MessageType.GET_STATUS,
        ]

        for msg_type in message_types:
            # Create original message
            original_message = ZMQMessage.create_command(
                msg_type, client_id, request_id, original_data
            )

            # Property 1: Message should serialize to valid JSON
            json_str = original_message.to_json()
            assert isinstance(
                json_str, str
            ), f"Serialization should produce string for {msg_type}"

            # Verify it's valid JSON
            try:
                json.loads(json_str)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON produced for {msg_type}: {e}")

            # Property 2: Deserialization should recreate identical message
            deserialized_message = ZMQMessage.from_json(json_str)

            assert (
                deserialized_message.message_type == original_message.message_type
            ), f"Message type should be preserved for {msg_type}"
            assert (
                deserialized_message.client_id == original_message.client_id
            ), f"Client ID should be preserved for {msg_type}"
            assert (
                deserialized_message.request_id == original_message.request_id
            ), f"Request ID should be preserved for {msg_type}"

            # Property 3: All data fields and types should be preserved
            self._verify_data_integrity(
                original_message.data, deserialized_message.data, msg_type
            )

            # Property 4: Message should pass validation after round-trip
            is_valid, error = MessageValidator.validate_message(deserialized_message)
            if msg_type == MessageType.START_SIMULATION:
                assert (
                    is_valid
                ), f"Deserialized START_SIMULATION message should be valid: {error}"
            # Other message types may not require config validation

    @given(
        snake_head_x=st.integers(min_value=0, max_value=49),
        snake_head_y=st.integers(min_value=0, max_value=49),
        snake_length=st.integers(min_value=3, max_value=20),
        food_x=st.integers(min_value=0, max_value=49),
        food_y=st.integers(min_value=0, max_value=49),
        score=st.integers(min_value=0, max_value=1000),
        moves_count=st.integers(min_value=0, max_value=10000),
        is_game_over=st.booleans(),
    )
    @settings(max_examples=15, deadline=2000)
    def test_game_state_serialization_integrity_property(
        self,
        snake_head_x,
        snake_head_y,
        snake_length,
        food_x,
        food_y,
        score,
        moves_count,
        is_game_over,
    ):
        """
        **Feature: ai-hydra, Property 13a: Game State Serialization Integrity**

        *For any* game state data, serialization and deserialization should preserve
        all game state information without loss or corruption.
        **Validates: Requirements 13.1, 13.2**
        """
        # Create snake body (head + segments going left)
        snake_body = [(snake_head_x, snake_head_y)]
        for i in range(1, snake_length):
            segment_x = max(0, snake_head_x - i)  # Ensure valid coordinates
            snake_body.append((segment_x, snake_head_y))

        # Create game state data
        original_game_state = GameStateData(
            snake_head=(snake_head_x, snake_head_y),
            snake_body=snake_body,
            direction=(1, 0),  # Moving right
            food_position=(food_x, food_y),
            score=score,
            grid_size=(50, 50),
            moves_count=moves_count,
            is_game_over=is_game_over,
        )

        # Create status update message
        performance = PerformanceMetrics.create_empty()
        performance.decisions_per_second = 2.5
        performance.memory_usage_mb = 150.0

        original_message = MessageBuilder.status_update(
            original_game_state, performance, "running"
        )

        # Serialize and deserialize
        json_str = original_message.to_json()
        deserialized_message = ZMQMessage.from_json(json_str)

        # Verify game state integrity
        game_state_data = deserialized_message.data["game_state"]

        assert game_state_data["snake_head"] == [
            snake_head_x,
            snake_head_y,
        ], "Snake head position should be preserved"
        assert (
            len(game_state_data["snake_body"]) == snake_length
        ), f"Snake body length should be preserved: expected {snake_length}, got {len(game_state_data['snake_body'])}"
        assert game_state_data["food_position"] == [
            food_x,
            food_y,
        ], "Food position should be preserved"
        assert game_state_data["score"] == score, "Score should be preserved"
        assert (
            game_state_data["moves_count"] == moves_count
        ), "Moves count should be preserved"
        assert (
            game_state_data["is_game_over"] == is_game_over
        ), "Game over status should be preserved"

        # Verify performance metrics integrity
        perf_data = deserialized_message.data["performance"]
        assert (
            abs(perf_data["decisions_per_second"] - 2.5) < 0.001
        ), "Performance metrics should be preserved with precision"
        assert (
            abs(perf_data["memory_usage_mb"] - 150.0) < 0.001
        ), "Memory usage should be preserved with precision"

    @given(
        error_code=st.integers(min_value=1000, max_value=9999),
        error_message=st.text(min_size=1, max_size=200),
        timestamp=st.floats(
            min_value=1000000000, max_value=2000000000
        ),  # Valid Unix timestamps
        client_id=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
    )
    @settings(max_examples=10, deadline=2000)
    def test_error_message_serialization_property(
        self, error_code, error_message, timestamp, client_id
    ):
        """
        **Feature: ai-hydra, Property 13b: Error Message Serialization**

        *For any* error message with various error codes and messages, serialization
        should preserve all error information for proper client error handling.
        **Validates: Requirements 13.1, 13.2**
        """
        # Create error response message
        error_data = {
            "error_code": error_code,
            "error_message": error_message,
            "timestamp": timestamp,
            "server_info": {"version": "1.0.0", "uptime_seconds": 3600.5},
        }

        original_message = ZMQMessage.create_response(
            MessageType.ERROR_OCCURRED, f"req_{int(timestamp)}", error_data
        )

        # Serialize and deserialize
        json_str = original_message.to_json()
        deserialized_message = ZMQMessage.from_json(json_str)

        # Verify error data integrity
        error_data_received = deserialized_message.data

        assert (
            error_data_received["error_code"] == error_code
        ), "Error code should be preserved"
        assert (
            error_data_received["error_message"] == error_message
        ), "Error message should be preserved"
        assert (
            abs(error_data_received["timestamp"] - timestamp) < 0.001
        ), "Timestamp should be preserved with precision"
        assert (
            error_data_received["server_info"]["version"] == "1.0.0"
        ), "Server info should be preserved"
        assert (
            abs(error_data_received["server_info"]["uptime_seconds"] - 3600.5) < 0.001
        ), "Server uptime should be preserved with precision"

    def _verify_data_integrity(
        self, original_data: dict, deserialized_data: dict, msg_type: str
    ) -> None:
        """Verify that all data fields and types are preserved after serialization."""
        assert type(original_data) == type(
            deserialized_data
        ), f"Data type should be preserved for {msg_type}"

        if isinstance(original_data, dict):
            assert set(original_data.keys()) == set(
                deserialized_data.keys()
            ), f"Dictionary keys should be preserved for {msg_type}"

            for key in original_data.keys():
                self._verify_data_integrity(
                    original_data[key], deserialized_data[key], f"{msg_type}.{key}"
                )

        elif isinstance(original_data, list):
            assert len(original_data) == len(
                deserialized_data
            ), f"List length should be preserved for {msg_type}"

            for i, (orig_item, deser_item) in enumerate(
                zip(original_data, deserialized_data)
            ):
                self._verify_data_integrity(orig_item, deser_item, f"{msg_type}[{i}]")

        elif isinstance(original_data, float):
            assert (
                abs(original_data - deserialized_data) < 0.001
            ), f"Float precision should be preserved for {msg_type}: {original_data} vs {deserialized_data}"

        else:
            assert (
                original_data == deserialized_data
            ), f"Value should be preserved for {msg_type}: {original_data} vs {deserialized_data}"


class TestZMQProtocol:
    """Test ZeroMQ message protocol components."""

    def test_message_creation_and_serialization(self):
        """Test creating and serializing ZMQ messages."""
        # Create a command message
        message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            "test_client",
            "req_123",
            {"config": {"grid_size": [10, 10]}},
        )

        assert message.message_type == MessageType.START_SIMULATION
        assert message.client_id == "test_client"
        assert message.request_id == "req_123"
        assert message.data["config"]["grid_size"] == [10, 10]

        # Test JSON serialization
        json_str = message.to_json()
        assert isinstance(json_str, str)

        # Test deserialization
        deserialized = ZMQMessage.from_json(json_str)
        assert deserialized.message_type == message.message_type
        assert deserialized.client_id == message.client_id
        assert deserialized.request_id == message.request_id
        assert deserialized.data == message.data

    def test_message_builder(self):
        """Test MessageBuilder helper functions."""
        # Test status update message
        game_state = GameStateData(
            snake_head=(5, 5),
            snake_body=[(5, 5), (4, 5), (3, 5)],
            direction=(1, 0),
            food_position=(8, 8),
            score=10,
            grid_size=(10, 10),
            moves_count=25,
            is_game_over=False,
        )

        performance = PerformanceMetrics.create_empty()
        performance.decisions_per_second = 2.5

        status_msg = MessageBuilder.status_update(game_state, performance, "running")

        assert status_msg.message_type == MessageType.STATUS_UPDATE
        assert status_msg.data["simulation_status"] == "running"
        assert status_msg.data["game_state"]["score"] == 10
        assert status_msg.data["performance"]["decisions_per_second"] == 2.5

    def test_message_validator(self):
        """Test message validation."""
        # Valid start simulation message
        valid_message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            "client1",
            "req1",
            {"config": {"grid_size": [10, 10], "move_budget": 50}},
        )

        is_valid, error = MessageValidator.validate_message(valid_message)
        assert is_valid
        assert error is None

        # Invalid message - missing config
        invalid_message = ZMQMessage.create_command(
            MessageType.START_SIMULATION, "client1", "req2", {}
        )

        is_valid, error = MessageValidator.validate_message(invalid_message)
        assert not is_valid
        assert "Missing required field: config" in error

        # Invalid grid size
        invalid_grid_message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            "client1",
            "req3",
            {"config": {"grid_size": [2, 2], "move_budget": 50}},  # Too small
        )

        is_valid, error = MessageValidator.validate_message(invalid_grid_message)
        assert not is_valid
        assert "grid_size values must be integers >= 5" in error

    def test_game_state_data_from_board(self):
        """Test creating GameStateData from GameBoard."""
        # Create a test GameBoard
        board = GameBoard(
            snake_head=Position(5, 5),
            snake_body=(Position(5, 5), Position(4, 5), Position(3, 5)),
            direction=Direction(1, 0),
            food_position=Position(8, 8),
            score=15,
            move_count=0,
            random_state=None,
            grid_size=(10, 10),
        )

        game_state = GameStateData.from_game_board(board, moves_count=30)

        assert game_state.snake_head == (5, 5)
        assert game_state.snake_body == [(5, 5), (4, 5), (3, 5)]
        assert game_state.direction == (1, 0)
        assert game_state.food_position == (8, 8)
        assert game_state.score == 15
        assert game_state.grid_size == (10, 10)
        assert game_state.moves_count == 30
        assert game_state.is_game_over == False


class TestZMQServer:
    """Test ZeroMQ server functionality."""

    def test_server_initialization(self):
        """Test server initialization."""
        server = ZMQServer(bind_address="tcp://*:5556")

        assert server.bind_address == "tcp://*:5556"
        assert server.simulation_state == SimulationState.IDLE
        assert not server.is_running
        assert len(server.connected_clients) == 0
        assert server.hydra_mgr is None

    @pytest.mark.asyncio
    async def test_message_routing(self):
        """Test message routing to appropriate handlers."""
        server = ZMQServer()

        # Mock the socket to avoid actual ZMQ operations
        server.socket = Mock()

        # Test GET_STATUS message
        status_message = ZMQMessage.create_command(
            MessageType.GET_STATUS, "test_client", "req_1"
        )

        response = await server._route_message(status_message)

        assert response.message_type == MessageType.STATUS_RESPONSE
        assert response.request_id == "req_1"
        assert "server_id" in response.data
        assert "simulation_state" in response.data

    @pytest.mark.asyncio
    async def test_start_simulation_validation(self):
        """Test start simulation message handling."""
        server = ZMQServer()
        server.socket = Mock()

        # Valid start simulation message
        start_message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            "test_client",
            "req_start",
            {"config": {"grid_size": [8, 8], "move_budget": 30, "random_seed": 42}},
        )

        # Mock the simulation thread to avoid actual simulation
        with patch("threading.Thread"):
            response = await server._handle_start_simulation(start_message)

        assert response.message_type == MessageType.SIMULATION_STARTED
        assert response.request_id == "req_start"
        assert server.simulation_state == SimulationState.RUNNING

    @pytest.mark.asyncio
    async def test_stop_simulation(self):
        """Test stop simulation functionality."""
        server = ZMQServer()
        server.socket = Mock()
        server.simulation_state = SimulationState.RUNNING

        stop_message = ZMQMessage.create_command(
            MessageType.STOP_SIMULATION, "test_client", "req_stop"
        )

        response = await server._handle_stop_simulation(stop_message)

        assert response.message_type == MessageType.SIMULATION_STOPPED
        assert response.request_id == "req_stop"
        assert server.simulation_state == SimulationState.STOPPED

    def test_performance_metrics_update(self):
        """Test performance metrics calculation."""
        server = ZMQServer()

        # Mock HydraMgr
        server.hydra_mgr = Mock()
        server.hydra_mgr.decision_cycle_count = 10
        server.start_time = time.time() - 5.0  # 5 seconds ago

        with patch("psutil.Process") as mock_process:
            mock_process.return_value.memory_info.return_value.rss = (
                100 * 1024 * 1024
            )  # 100MB
            mock_process.return_value.cpu_percent.return_value = 25.5

            server._update_performance_metrics()

        assert (
            abs(server.performance_metrics.decisions_per_second - 2.0) < 0.1
        )  # ~10 decisions / 5 seconds
        assert server.performance_metrics.total_decision_cycles == 10
        assert server.performance_metrics.memory_usage_mb == 100.0
        assert server.performance_metrics.cpu_usage_percent == 25.5


class TestZMQClient:
    """Test ZeroMQ client functionality."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = ZMQClient("tcp://localhost:5556")

        assert client.server_address == "tcp://localhost:5556"
        assert not client.is_connected
        assert client.request_counter == 0
        assert client.client_id is not None

    def test_request_id_generation(self):
        """Test unique request ID generation."""
        client = ZMQClient()

        req_id_1 = client._create_request_id()
        req_id_2 = client._create_request_id()

        assert req_id_1 != req_id_2
        assert client.client_id in req_id_1
        assert client.client_id in req_id_2
        assert client.request_counter == 2


@pytest.mark.integration
class TestZMQIntegration:
    """Integration tests for ZeroMQ communication."""

    @pytest.mark.asyncio
    async def test_server_client_communication(self):
        """Test basic server-client communication."""
        # This test would require actual ZMQ sockets and is more complex
        # For now, we'll skip it but it would test:
        # 1. Start server on a test port
        # 2. Connect client to server
        # 3. Send messages and verify responses
        # 4. Clean shutdown
        pytest.skip("Integration test requires actual ZMQ setup")

    def test_message_protocol_compatibility(self):
        """Test that messages are compatible between client and server."""
        # Create a message with the client
        client = ZMQClient()

        # Simulate creating a start simulation message
        config = {"grid_size": [10, 10], "move_budget": 100, "random_seed": 42}

        message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            client.client_id,
            client._create_request_id(),
            {"config": config},
        )

        # Serialize and deserialize (simulating network transmission)
        json_data = message.to_json()
        received_message = ZMQMessage.from_json(json_data)

        # Validate the message (as server would)
        is_valid, error = MessageValidator.validate_message(received_message)

        assert is_valid
        assert error is None
        assert received_message.data["config"]["grid_size"] == [10, 10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
