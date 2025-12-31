"""
End-to-end tests for AI Hydra Router system.
"""

import asyncio
import pytest
import time
from unittest.mock import patch, AsyncMock

from ai_hydra.router import HydraRouter
from ai_hydra.mq_client import MQClient
from ai_hydra.router_constants import RouterConstants
from ai_hydra.zmq_protocol import MessageType, ZMQMessage


class TestRouterSystemE2E:
    """End-to-end tests for the complete router system."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_complete_client_server_workflow(self):
        """Test complete workflow from client connection to message exchange."""
        # This test simulates the complete workflow without actual network sockets
        
        with patch('zmq.asyncio.Context') as mock_context:
            # Setup mock sockets
            router_socket = MockE2ESocket()
            client_socket = MockE2ESocket()
            server_socket = MockE2ESocket()
            
            mock_context.return_value.socket.return_value = router_socket
            
            # Create router
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="DEBUG"
            )
            
            # Create client
            client = MQClient(
                router_address="tcp://127.0.0.1:5556",
                client_type=RouterConstants.HYDRA_CLIENT,
                client_id="test-client"
            )
            client.socket = client_socket
            client.context = mock_context.return_value
            
            # Create server client
            server = MQClient(
                router_address="tcp://127.0.0.1:5556",
                client_type=RouterConstants.HYDRA_SERVER,
                client_id="test-server"
            )
            server.socket = server_socket
            server.context = mock_context.return_value
            
            # Simulate connections
            client.is_connected = True
            server.is_connected = True
            
            # Register clients with router
            router.clients = {
                "test-client": (RouterConstants.HYDRA_CLIENT, time.time()),
                "test-server": (RouterConstants.HYDRA_SERVER, time.time())
            }
            
            # Test 1: Client sends command to server
            command_message = ZMQMessage.create_command(
                message_type=MessageType.START_SIMULATION,
                client_id="test-client",
                request_id="req-001",
                data={"config": {"grid_size": [10, 10]}}
            )
            
            await client.send_message(command_message)
            
            # Verify client sent message
            assert len(client_socket.sent_messages) == 1
            sent_msg = client_socket.sent_messages[0]
            assert sent_msg["sender"] == RouterConstants.HYDRA_CLIENT
            assert sent_msg["message_type"] == MessageType.START_SIMULATION.value
            
            # Test 2: Simulate router forwarding to server
            await router.forward_to_server(
                elem=RouterConstants.START_SIMULATION,
                data={"config": {"grid_size": [10, 10]}},
                sender=b"test-client"
            )
            
            # Verify router sent to server and acknowledged client
            assert len(router_socket.sent_multiparts) == 2
            
            # Test 3: Server sends response back
            response_message = ZMQMessage.create_response(
                message_type=MessageType.SIMULATION_STARTED,
                client_id="test-server",
                request_id="req-001",
                data={"status": "started", "simulation_id": "sim-001"}
            )
            
            await server.send_message(response_message)
            
            # Verify server sent response
            assert len(server_socket.sent_messages) == 1
            response_msg = server_socket.sent_messages[0]
            assert response_msg["sender"] == RouterConstants.HYDRA_SERVER
            assert response_msg["message_type"] == MessageType.SIMULATION_STARTED.value
            
            # Test 4: Simulate router broadcasting to clients
            await router.broadcast_to_clients(
                elem=RouterConstants.SIMULATION_STARTED,
                data={"status": "started", "simulation_id": "sim-001"},
                sender_id="test-server"
            )
            
            # Verify broadcast was sent to client
            broadcast_calls = [call for call in router_socket.sent_multiparts 
                             if call[0] == b"test-client"]
            assert len(broadcast_calls) >= 1
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(20)
    async def test_heartbeat_lifecycle(self):
        """Test complete heartbeat lifecycle."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockE2ESocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            # Create router
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="DEBUG"
            )
            
            # Create client with fast heartbeat
            client = MQClient(
                router_address="tcp://127.0.0.1:5556",
                client_type=RouterConstants.HYDRA_CLIENT,
                client_id="heartbeat-client",
                heartbeat_interval=0.1  # Fast for testing
            )
            client.socket = mock_socket
            client.context = mock_context.return_value
            client.is_connected = True
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(client._send_heartbeat())
            
            # Let it run for a short time
            await asyncio.sleep(0.3)
            
            # Stop heartbeat
            client.stop_event.set()
            await heartbeat_task
            
            # Verify heartbeats were sent
            heartbeat_messages = [msg for msg in mock_socket.sent_messages 
                                if msg.get("message_type") == "HEARTBEAT"]
            assert len(heartbeat_messages) >= 2
            
            # Verify heartbeat message format
            for hb_msg in heartbeat_messages:
                assert hb_msg["sender"] == RouterConstants.HYDRA_CLIENT
                assert hb_msg["client_id"] == "heartbeat-client"
                assert "timestamp" in hb_msg
                assert "request_id" in hb_msg
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_client_disconnection_cleanup(self):
        """Test client disconnection and cleanup."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockE2ESocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="DEBUG"
            )
            
            # Add clients with different activity levels
            current_time = time.time()
            old_time = current_time - (RouterConstants.HEARTBEAT_INTERVAL * 4)
            
            router.clients = {
                "active-client": (RouterConstants.HYDRA_CLIENT, current_time),
                "inactive-client": (RouterConstants.HYDRA_CLIENT, old_time),
                "active-server": (RouterConstants.HYDRA_SERVER, current_time)
            }
            
            # Run pruning
            await router.prune_dead_clients()
            
            # Verify inactive client was removed
            assert "active-client" in router.clients
            assert "inactive-client" not in router.clients
            assert "active-server" in router.clients
            
            # Verify counts are correct
            assert router.client_count == 1
            assert router.server_count == 1
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_error_handling_no_server(self):
        """Test error handling when no server is available."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockE2ESocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="DEBUG"
            )
            
            # Only client, no server
            router.clients = {
                "lonely-client": (RouterConstants.HYDRA_CLIENT, time.time())
            }
            
            # Try to forward message
            await router.forward_to_server(
                elem=RouterConstants.START_SIMULATION,
                data={"config": "test"},
                sender=b"lonely-client"
            )
            
            # Should send error to client
            assert len(mock_socket.sent_multiparts) == 1
            error_call = mock_socket.sent_multiparts[0]
            assert error_call[0] == b"lonely-client"
            
            # Verify error message content
            import zmq.utils.jsonapi
            error_msg = zmq.utils.jsonapi.loads(error_call[1])
            assert RouterConstants.ERROR in error_msg
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_multiple_clients_single_server(self):
        """Test multiple clients communicating with single server."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockE2ESocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="DEBUG"
            )
            
            # Setup multiple clients and one server
            current_time = time.time()
            router.clients = {
                "client-1": (RouterConstants.HYDRA_CLIENT, current_time),
                "client-2": (RouterConstants.HYDRA_CLIENT, current_time),
                "client-3": (RouterConstants.HYDRA_CLIENT, current_time),
                "server-1": (RouterConstants.HYDRA_SERVER, current_time)
            }
            
            # Test server broadcasting to all clients
            await router.broadcast_to_clients(
                elem=RouterConstants.STATUS_UPDATE,
                data={"status": "running", "epoch": 42},
                sender_id="server-1"
            )
            
            # Should send to all 3 clients
            client_calls = [call for call in mock_socket.sent_multiparts 
                          if call[0] in [b"client-1", b"client-2", b"client-3"]]
            assert len(client_calls) == 3
            
            # Verify message content
            for call in client_calls:
                import zmq.utils.jsonapi
                msg = zmq.utils.jsonapi.loads(call[1])
                assert msg[RouterConstants.SENDER] == RouterConstants.HYDRA_SERVER
                assert msg[RouterConstants.ELEM] == RouterConstants.STATUS_UPDATE
                assert msg[RouterConstants.DATA]["epoch"] == 42


class MockE2ESocket:
    """Enhanced mock socket for end-to-end testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.sent_multiparts = []
        self.received_messages = []
        self.is_bound = False
        self.is_connected = False
    
    def bind(self, address):
        self.is_bound = True
    
    def connect(self, address):
        self.is_connected = True
    
    def setsockopt(self, option, value):
        pass
    
    async def send_json(self, data):
        self.sent_messages.append(data)
    
    async def send_multipart(self, parts):
        self.sent_multiparts.append(parts)
    
    async def recv_json(self):
        if self.received_messages:
            return self.received_messages.pop(0)
        return {"test": "response"}
    
    async def recv_multipart(self):
        return [b"test_identity", b'{"test": "message"}']
    
    async def poll(self, timeout=None):
        return len(self.received_messages) > 0 or timeout == 0
    
    def close(self, linger=None):
        self.is_connected = False
        self.is_bound = False
    
    def disconnect(self, address):
        self.is_connected = False
    
    def add_received_message(self, message):
        """Helper method to simulate receiving messages."""
        self.received_messages.append(message)