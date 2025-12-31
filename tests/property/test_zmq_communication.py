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
