"""
Unit tests for AI Hydra Router.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from ai_hydra.router import HydraRouter
from ai_hydra.router_constants import RouterConstants


class TestHydraRouter:
    """Unit tests for HydraRouter class."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.router_address = "127.0.0.1"
        self.router_port = 5556
        self.log_level = "INFO"
    
    def test_router_initialization(self):
        """Test that HydraRouter initializes correctly."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MagicMock()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            assert router.clients == {}
            assert router.client_count == 0
            assert router.server_count == 0
            mock_socket.bind.assert_called_once_with(f"tcp://{self.router_address}:{self.router_port}")
    
    def test_router_initialization_bind_failure(self):
        """Test router initialization with bind failure."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MagicMock()
            mock_socket.bind.side_effect = Exception("Bind failed")
            mock_context.return_value.socket.return_value = mock_socket
            
            with pytest.raises(Exception, match="Bind failed"):
                HydraRouter(
                    router_address=self.router_address,
                    router_port=self.router_port,
                    log_level=self.log_level
                )
    
    @pytest.mark.asyncio
    async def test_broadcast_to_clients(self):
        """Test broadcasting messages to clients."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            # Add some test clients
            router.clients = {
                "client1": (RouterConstants.HYDRA_CLIENT, time.time()),
                "client2": (RouterConstants.HYDRA_CLIENT, time.time()),
                "server1": (RouterConstants.HYDRA_SERVER, time.time())
            }
            
            await router.broadcast_to_clients("test_elem", {"test": "data"}, "sender1")
            
            # Should send to both clients but not server
            assert mock_socket.send_multipart.call_count == 2
            
            # Check message format
            calls = mock_socket.send_multipart.call_args_list
            for call in calls:
                identity, message_bytes = call[0][0]
                assert identity in [b"client1", b"client2"]
    
    @pytest.mark.asyncio
    async def test_broadcast_to_clients_no_clients(self):
        """Test broadcasting when no clients are connected."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            await router.broadcast_to_clients("test_elem", {"test": "data"}, "sender1")
            
            # Should not send any messages
            mock_socket.send_multipart.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_forward_to_server(self):
        """Test forwarding client messages to server."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            # Add a test server
            router.clients = {
                "server1": (RouterConstants.HYDRA_SERVER, time.time())
            }
            
            await router.forward_to_server("test_elem", {"test": "data"}, b"client1")
            
            # Should send message to server and acknowledge client
            assert mock_socket.send_multipart.call_count == 2
            
            # Check server message
            server_call = mock_socket.send_multipart.call_args_list[0]
            server_identity, server_message = server_call[0][0]
            assert server_identity == b"server1"
            
            # Check client acknowledgment
            client_call = mock_socket.send_multipart.call_args_list[1]
            client_identity, client_message = client_call[0][0]
            assert client_identity == b"client1"
    
    @pytest.mark.asyncio
    async def test_forward_to_server_no_server(self):
        """Test forwarding when no server is connected."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            # No servers in clients dict
            router.clients = {}
            
            await router.forward_to_server("test_elem", {"test": "data"}, b"client1")
            
            # Should send error message to client
            mock_socket.send_multipart.assert_called_once()
            
            call_args = mock_socket.send_multipart.call_args[0][0]
            client_identity, error_message = call_args
            assert client_identity == b"client1"
    
    @pytest.mark.asyncio
    async def test_prune_dead_clients(self):
        """Test pruning of inactive clients."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = AsyncMock()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            # Add clients with different timestamps
            current_time = time.time()
            old_time = current_time - (RouterConstants.HEARTBEAT_INTERVAL * 4)  # Too old
            
            router.clients = {
                "active_client": (RouterConstants.HYDRA_CLIENT, current_time),
                "dead_client": (RouterConstants.HYDRA_CLIENT, old_time),
                "active_server": (RouterConstants.HYDRA_SERVER, current_time)
            }
            
            await router.prune_dead_clients()
            
            # Dead client should be removed
            assert "dead_client" not in router.clients
            assert "active_client" in router.clients
            assert "active_server" in router.clients
            
            # Counts should be updated
            assert router.client_count == 1
            assert router.server_count == 1
    
    @pytest.mark.asyncio
    async def test_handle_heartbeat_message(self):
        """Test handling of heartbeat messages."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = AsyncMock()
            
            # Mock receiving a heartbeat message
            heartbeat_msg = {
                RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
                RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                RouterConstants.DATA: {}
            }
            
            mock_socket.recv_multipart.return_value = [
                b"client1",
                b'{"sender": "HydraClient", "elem": "heartbeat", "data": {}}'
            ]
            
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            # Simulate processing one message
            frames = await mock_socket.recv_multipart()
            identity = frames[0]
            identity_str = identity.decode()
            msg_bytes = frames[1]
            
            import zmq.utils.jsonapi
            msg = zmq.utils.jsonapi.loads(msg_bytes)
            
            # Process heartbeat
            if msg.get(RouterConstants.ELEM) == RouterConstants.HEARTBEAT:
                async with router.clients_lock:
                    router.clients[identity_str] = (msg.get(RouterConstants.SENDER), time.time())
            
            # Verify client was registered
            assert "client1" in router.clients
            assert router.clients["client1"][0] == RouterConstants.HYDRA_CLIENT
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test router shutdown."""
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MagicMock()
            mock_ctx = MagicMock()
            mock_context.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address=self.router_address,
                router_port=self.router_port,
                log_level=self.log_level
            )
            
            await router.shutdown()
            
            mock_socket.close.assert_called_once_with(linger=0)
            mock_ctx.term.assert_called_once()
    
    def test_router_constants(self):
        """Test that router constants are properly defined."""
        assert RouterConstants.HYDRA_CLIENT == "HydraClient"
        assert RouterConstants.HYDRA_SERVER == "HydraServer"
        assert RouterConstants.HYDRA_ROUTER == "HydraRouter"
        assert RouterConstants.HEARTBEAT == "heartbeat"
        assert RouterConstants.HEARTBEAT_INTERVAL == 5
        assert RouterConstants.DEFAULT_ROUTER_PORT == 5556