"""
Integration tests for AI Hydra Router system.
"""

import asyncio
import pytest
import time
from unittest.mock import patch

from ai_hydra.router import HydraRouter
from ai_hydra.mq_client import MQClient
from ai_hydra.router_constants import RouterConstants
from ai_hydra.zmq_protocol import MessageType


class TestRouterIntegration:
    """Integration tests for router and MQClient interaction."""

    @pytest.mark.asyncio
    async def test_client_server_communication_through_router(self):
        """Test complete communication flow through router."""
        # This test would require actual ZMQ sockets, so we'll mock the key parts

        with patch("zmq.asyncio.Context") as mock_context:
            # Mock router setup
            mock_router_socket = MockAsyncSocket()
            mock_context.return_value.socket.return_value = mock_router_socket

            router = HydraRouter(
                router_address="127.0.0.1", router_port=5556, log_level="DEBUG"
            )

            # Mock client setup
            mock_client_socket = MockAsyncSocket()

            client = MQClient(
                router_address="tcp://127.0.0.1:5556",
                client_type=RouterConstants.HYDRA_CLIENT,
                client_id="test-client",
            )

            # Replace client socket with mock
            client.socket = mock_client_socket
            client.context = mock_context.return_value

            # Test connection
            client.is_connected = True  # Simulate successful connection

            # Test message sending
            from ai_hydra.zmq_protocol import ZMQMessage

            message = ZMQMessage.create_command(
                message_type=MessageType.GET_STATUS,
                client_id="test-client",
                request_id="test-request",
                data={"test": "data"},
            )

            await client.send_message(message)

            # Verify message was formatted correctly
            assert mock_client_socket.sent_messages
            sent_data = mock_client_socket.sent_messages[0]
            assert sent_data["sender"] == RouterConstants.HYDRA_CLIENT
            assert sent_data["client_id"] == "test-client"

    @pytest.mark.asyncio
    async def test_heartbeat_flow(self):
        """Test heartbeat message flow."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = MockAsyncSocket()
            mock_context.return_value.socket.return_value = mock_socket

            client = MQClient(
                router_address="tcp://127.0.0.1:5556",
                client_type=RouterConstants.HYDRA_CLIENT,
                client_id="test-client",
                heartbeat_interval=0.1,  # Fast heartbeat for testing
            )

            # Replace socket with mock
            client.socket = mock_socket
            client.context = mock_context.return_value
            client.is_connected = True

            # Start heartbeat
            heartbeat_task = asyncio.create_task(client._send_heartbeat())

            # Wait for heartbeats
            await asyncio.sleep(0.3)

            # Stop heartbeat
            client.stop_event.set()
            await heartbeat_task

            # Check that heartbeats were sent
            heartbeat_messages = [
                msg
                for msg in mock_socket.sent_messages
                if msg.get("elem") == RouterConstants.HEARTBEAT
            ]
            assert len(heartbeat_messages) >= 2  # Should have sent multiple heartbeats

    @pytest.mark.asyncio
    async def test_router_client_registration(self):
        """Test client registration with router."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = MockAsyncSocket()
            mock_context.return_value.socket.return_value = mock_socket

            router = HydraRouter(
                router_address="127.0.0.1", router_port=5556, log_level="DEBUG"
            )

            # Simulate receiving heartbeat from client
            current_time = time.time()

            # Manually add client (simulating heartbeat processing)
            async with router.clients_lock:
                router.clients["test-client"] = (
                    RouterConstants.HYDRA_CLIENT,
                    current_time,
                )

            # Test client pruning doesn't remove active client
            await router.prune_dead_clients()

            assert "test-client" in router.clients
            assert router.client_count == 1

    @pytest.mark.asyncio
    async def test_message_routing_logic(self):
        """Test router message routing logic."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = MockAsyncSocket()
            mock_context.return_value.socket.return_value = mock_socket

            router = HydraRouter(
                router_address="127.0.0.1", router_port=5556, log_level="DEBUG"
            )

            # Add test clients and server
            router.clients = {
                "client1": (RouterConstants.HYDRA_CLIENT, time.time()),
                "client2": (RouterConstants.HYDRA_CLIENT, time.time()),
                "server1": (RouterConstants.HYDRA_SERVER, time.time()),
            }

            # Test client to server forwarding
            await router.forward_to_server(
                elem=RouterConstants.START_SIMULATION,
                data={"config": "test"},
                sender=b"client1",
            )

            # Should send to server and acknowledge client
            assert len(mock_socket.sent_multiparts) == 2

            # Test server to clients broadcasting
            await router.broadcast_to_clients(
                elem=RouterConstants.STATUS_UPDATE,
                data={"status": "running"},
                sender_id="server1",
            )

            # Should broadcast to both clients
            broadcast_calls = len(
                [
                    call
                    for call in mock_socket.sent_multiparts
                    if call[0] in [b"client1", b"client2"]
                ]
            )
            assert broadcast_calls == 2

    @pytest.mark.asyncio
    async def test_error_handling_no_server(self):
        """Test error handling when no server is available."""
        with patch("zmq.asyncio.Context") as mock_context:
            mock_socket = MockAsyncSocket()
            mock_context.return_value.socket.return_value = mock_socket

            router = HydraRouter(
                router_address="127.0.0.1", router_port=5556, log_level="DEBUG"
            )

            # Only clients, no server
            router.clients = {"client1": (RouterConstants.HYDRA_CLIENT, time.time())}

            await router.forward_to_server(
                elem=RouterConstants.START_SIMULATION,
                data={"config": "test"},
                sender=b"client1",
            )

            # Should send error message to client
            assert len(mock_socket.sent_multiparts) == 1

            # Check error message content
            error_call = mock_socket.sent_multiparts[0]
            assert error_call[0] == b"client1"  # Sent to requesting client


class MockAsyncSocket:
    """Mock async socket for testing."""

    def __init__(self):
        self.sent_messages = []
        self.sent_multiparts = []
        self.bound_addresses = []
        self.connected_addresses = []

    def bind(self, address):
        """Mock bind method."""
        self.bound_addresses.append(address)

    def connect(self, address):
        """Mock connect method."""
        self.connected_addresses.append(address)

    def setsockopt(self, option, value):
        """Mock setsockopt method."""
        pass

    async def send_json(self, data):
        """Mock send_json method."""
        self.sent_messages.append(data)

    async def send_multipart(self, parts):
        """Mock send_multipart method."""
        self.sent_multiparts.append(parts)

    async def recv_json(self):
        """Mock recv_json method."""
        return {"test": "response"}

    async def recv_multipart(self):
        """Mock recv_multipart method."""
        return [b"test_identity", b'{"test": "message"}']

    async def poll(self, timeout=None):
        """Mock poll method."""
        return True

    def close(self, linger=None):
        """Mock close method."""
        pass

    def disconnect(self, address):
        """Mock disconnect method."""
        pass
