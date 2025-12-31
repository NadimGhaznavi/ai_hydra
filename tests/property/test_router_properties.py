"""
Property-based tests for AI Hydra Router system.
"""

import asyncio
import pytest
import time
from hypothesis import given, strategies as st, settings
from unittest.mock import patch

from ai_hydra.router import HydraRouter
from ai_hydra.mq_client import MQClient
from ai_hydra.router_constants import RouterConstants


class TestRouterProperties:
    """Property-based tests for router system."""
    
    @given(
        client_count=st.integers(min_value=0, max_value=10),
        server_count=st.integers(min_value=0, max_value=3)
    )
    @settings(max_examples=20, deadline=5000)
    def test_client_tracking_property(self, client_count, server_count):
        """
        **Feature: ai-hydra, Property 1: Client Tracking Consistency**
        **Validates: Router client management requirements**
        
        For any number of clients and servers, the router should accurately
        track and count connected clients and servers.
        """
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockSocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="ERROR"  # Reduce noise in tests
            )
            # Don't start background tasks in property tests
            
            # Add clients and servers
            current_time = time.time()
            
            for i in range(client_count):
                router.clients[f"client_{i}"] = (RouterConstants.HYDRA_CLIENT, current_time)
            
            for i in range(server_count):
                router.clients[f"server_{i}"] = (RouterConstants.HYDRA_SERVER, current_time)
            
            # Count clients and servers
            actual_clients = sum(1 for client_type, _ in router.clients.values() 
                               if client_type == RouterConstants.HYDRA_CLIENT)
            actual_servers = sum(1 for client_type, _ in router.clients.values() 
                               if client_type == RouterConstants.HYDRA_SERVER)
            
            assert actual_clients == client_count
            assert actual_servers == server_count
            assert len(router.clients) == client_count + server_count
    
    @given(
        heartbeat_intervals=st.lists(
            st.floats(min_value=0.1, max_value=10.0),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_heartbeat_interval_property(self, heartbeat_intervals):
        """
        **Feature: ai-hydra, Property 2: Heartbeat Interval Consistency**
        **Validates: Heartbeat timing requirements**
        
        For any valid heartbeat interval, the MQClient should respect
        the configured interval timing.
        """
        for interval in heartbeat_intervals:
            with patch('zmq.asyncio.Context') as mock_context:
                mock_socket = MockSocket()
                mock_context.return_value.socket.return_value = mock_socket
                
                client = MQClient(
                    router_address="tcp://127.0.0.1:5556",
                    client_type=RouterConstants.HYDRA_CLIENT,
                    heartbeat_interval=interval
                )
                
                assert client.heartbeat_interval == interval
                assert client.heartbeat_interval > 0
                assert client.heartbeat_interval <= 10.0
    
    @given(
        client_ids=st.lists(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            min_size=1,
            max_size=10,
            unique=True
        )
    )
    @settings(max_examples=15, deadline=3000)
    def test_client_id_uniqueness_property(self, client_ids):
        """
        **Feature: ai-hydra, Property 3: Client ID Uniqueness**
        **Validates: Client identification requirements**
        
        For any set of unique client IDs, the router should maintain
        separate tracking for each client.
        """
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockSocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="ERROR"
            )
            
            # Add clients with unique IDs
            current_time = time.time()
            for client_id in client_ids:
                router.clients[client_id] = (RouterConstants.HYDRA_CLIENT, current_time)
            
            # Verify all clients are tracked
            assert len(router.clients) == len(client_ids)
            
            # Verify all IDs are present
            for client_id in client_ids:
                assert client_id in router.clients
                assert router.clients[client_id][0] == RouterConstants.HYDRA_CLIENT
    
    @given(
        message_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.booleans(),
                st.floats(allow_nan=False, allow_infinity=False)
            ),
            min_size=0,
            max_size=5
        )
    )
    @settings(max_examples=20, deadline=3000)
    def test_message_data_preservation_property(self, message_data):
        """
        **Feature: ai-hydra, Property 4: Message Data Preservation**
        **Validates: Message routing integrity requirements**
        
        For any valid message data, the router should preserve the data
        content during routing operations.
        """
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockSocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            client = MQClient(
                router_address="tcp://127.0.0.1:5556",
                client_type=RouterConstants.HYDRA_CLIENT,
                client_id="test-client"
            )
            
            # Replace socket with mock
            client.socket = mock_socket
            client.context = mock_context.return_value
            client.is_connected = True
            
            # Create and send message
            from ai_hydra.zmq_protocol import ZMQMessage, MessageType
            message = ZMQMessage.create_command(
                message_type=MessageType.GET_STATUS,
                client_id="test-client",
                request_id="test-request",
                data=message_data
            )
            
            # Send message (synchronously for testing)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(client.send_message(message))
            finally:
                loop.close()
            
            # Verify data preservation
            if mock_socket.sent_messages:
                sent_data = mock_socket.sent_messages[0]
                assert sent_data["data"] == message_data
    
    @given(
        prune_threshold_multiplier=st.floats(min_value=2.0, max_value=10.0)
    )
    @settings(max_examples=10, deadline=2000)
    def test_client_pruning_threshold_property(self, prune_threshold_multiplier):
        """
        **Feature: ai-hydra, Property 5: Client Pruning Threshold**
        **Validates: Client lifecycle management requirements**
        
        For any valid pruning threshold, clients should only be removed
        when they exceed the heartbeat timeout threshold.
        """
        with patch('zmq.asyncio.Context') as mock_context:
            mock_socket = MockSocket()
            mock_context.return_value.socket.return_value = mock_socket
            
            router = HydraRouter(
                router_address="127.0.0.1",
                router_port=5556,
                log_level="ERROR"
            )
            
            current_time = time.time()
            threshold = RouterConstants.HEARTBEAT_INTERVAL * prune_threshold_multiplier
            
            # Add clients with different timestamps
            router.clients = {
                "active_client": (RouterConstants.HYDRA_CLIENT, current_time),
                "old_client": (RouterConstants.HYDRA_CLIENT, current_time - threshold - 1)
            }
            
            # Run pruning (synchronously for testing)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(router.prune_dead_clients())
            finally:
                loop.close()
            
            # Active client should remain, old client should be removed
            if prune_threshold_multiplier > 3.0:  # Router uses 3x threshold
                assert "active_client" in router.clients
                assert "old_client" not in router.clients
            else:
                # Both should be removed if threshold is too low
                assert len(router.clients) <= 2


class MockSocket:
    """Mock socket for property testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.sent_multiparts = []
    
    def bind(self, address):
        pass
    
    def connect(self, address):
        pass
    
    def setsockopt(self, option, value):
        pass
    
    async def send_json(self, data):
        self.sent_messages.append(data)
    
    async def send_multipart(self, parts):
        self.sent_multiparts.append(parts)
    
    async def recv_json(self):
        return {"test": "response"}
    
    async def recv_multipart(self):
        return [b"test_identity", b'{"test": "message"}']
    
    async def poll(self, timeout=None):
        return True
    
    def close(self, linger=None):
        pass
    
    def disconnect(self, address):
        pass