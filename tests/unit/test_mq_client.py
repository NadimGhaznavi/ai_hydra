"""
Unit tests for AI Hydra MQClient.
"""

import asyncio
import pytest
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from ai_hydra.mq_client import MQClient
from ai_hydra.router_constants import RouterConstants
from ai_hydra.zmq_protocol import MessageType, ZMQMessage


class TestMQClient:
    """Unit tests for MQClient class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.router_address = "tcp://localhost:5556"
        self.client_type = "TestClient"
        self.heartbeat_interval = 1.0
        self.client_id = "test-client-1234"

    def test_client_initialization(self):
        """Test that MQClient initializes correctly."""
        client = MQClient(
            router_address=self.router_address,
            client_type=self.client_type,
            heartbeat_interval=self.heartbeat_interval,
            client_id=self.client_id,
        )

        assert client.router_address == self.router_address
        assert client.client_type == self.client_type
        assert client.heartbeat_interval == self.heartbeat_interval
        assert client.client_id == self.client_id
        assert not client.is_connected
        assert client.heartbeat_task is None

    def test_client_id_generation(self):
        """Test automatic client ID generation."""
        client = MQClient(
            router_address=self.router_address, client_type=self.client_type
        )

        assert client.client_id.startswith(self.client_type)
        assert len(client.client_id) > len(self.client_type)

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to router."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            # Mock successful connection
            result = await client.connect()

            assert result is True
            assert client.is_connected is True
            mock_socket.connect.assert_called_once_with(self.router_address)
            assert client.heartbeat_task is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_socket.connect.side_effect = Exception("Connection failed")
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            result = await client.connect()

            assert result is False
            assert client.is_connected is False
            assert client.heartbeat_task is None

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection from router."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            # Connect first
            await client.connect()
            assert client.is_connected is True

            # Then disconnect
            await client.disconnect()

            assert client.is_connected is False
            mock_socket.disconnect.assert_called_once_with(self.router_address)
            mock_socket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message through router."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            await client.connect()

            # Create test message
            message = ZMQMessage.create_command(
                message_type=MessageType.GET_STATUS,
                client_id=self.client_id,
                request_id=str(uuid.uuid4()),
                data={"test": "data"},
            )

            await client.send_message(message)

            # Verify message was sent
            mock_socket.send_json.assert_called_once()
            sent_data = mock_socket.send_json.call_args[0][0]

            assert sent_data["sender"] == self.client_type
            assert sent_data["client_id"] == self.client_id
            assert sent_data["message_type"] == MessageType.GET_STATUS.value
            assert sent_data["data"] == {"test": "data"}

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self):
        """Test sending message when not connected raises error."""
        client = MQClient(
            router_address=self.router_address,
            client_type=self.client_type,
            client_id=self.client_id,
        )

        message = ZMQMessage.create_command(
            message_type=MessageType.GET_STATUS,
            client_id=self.client_id,
            request_id=str(uuid.uuid4()),
            data={},
        )

        with pytest.raises(ConnectionError, match="Not connected to router"):
            await client.send_message(message)

    @pytest.mark.asyncio
    async def test_receive_message_non_blocking(self):
        """Test non-blocking message receive."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_socket.poll.return_value = True
            mock_socket.recv_json.return_value = {"test": "message"}
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            await client.connect()

            message = await client.receive_message()

            assert message == {"test": "message"}
            mock_socket.poll.assert_called_once_with(timeout=0)
            mock_socket.recv_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_message_no_message(self):
        """Test receive when no message available."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_socket.poll.return_value = False
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            await client.connect()

            message = await client.receive_message()

            assert message is None
            mock_socket.recv_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_receive_message_blocking_with_timeout(self):
        """Test blocking message receive with timeout."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_socket.poll.return_value = True
            mock_socket.recv_json.return_value = {"test": "message"}
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            await client.connect()

            message = await client.receive_message_blocking(timeout=5.0)

            assert message == {"test": "message"}
            mock_socket.poll.assert_called_once_with(timeout=5000)  # Convert to ms

    @pytest.mark.asyncio
    async def test_send_command_with_response(self):
        """Test send command and wait for response."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_socket.poll.return_value = True

            # Mock response message
            request_id = str(uuid.uuid4())
            response_message = {
                "request_id": request_id,
                "message_type": "STATUS_UPDATE",
                "data": {"status": "ok"},
            }
            mock_socket.recv_json.return_value = response_message
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            await client.connect()

            # Mock the request_id generation to match response
            with patch(
                "uuid.uuid4", return_value=MagicMock(hex=request_id.replace("-", ""))
            ):
                response = await client.send_command(
                    MessageType.GET_STATUS, {"test": "data"}, timeout=5.0
                )

            assert response == response_message
            mock_socket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_command_timeout(self):
        """Test send command timeout."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_socket.poll.return_value = False  # No response
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            await client.connect()

            response = await client.send_command(
                MessageType.GET_STATUS, {"test": "data"}, timeout=0.1  # Short timeout
            )

            assert response is None

    @pytest.mark.asyncio
    async def test_heartbeat_sending(self):
        """Test heartbeat message sending."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
                heartbeat_interval=0.1,  # Fast heartbeat for testing
            )

            await client.connect()

            # Wait for at least one heartbeat
            await asyncio.sleep(0.2)

            # Check that heartbeat was sent
            assert mock_socket.send_json.call_count >= 1

            # Check heartbeat message format
            heartbeat_calls = [
                call
                for call in mock_socket.send_json.call_args_list
                if "heartbeat" in str(call)
            ]
            assert len(heartbeat_calls) >= 1

            await client.disconnect()

    def test_context_manager(self):
        """Test MQClient as context manager."""
        with patch("zmq.asyncio.Context"):
            client = MQClient(
                router_address=self.router_address,
                client_type=self.client_type,
                client_id=self.client_id,
            )

            # Test context manager entry
            with client as ctx_client:
                assert ctx_client is client

            # Context manager exit should handle disconnection
            # (actual disconnection testing requires async context)
