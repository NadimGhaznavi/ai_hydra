"""
Integration tests for TUI epoch display with ZeroMQ communication.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
import zmq.asyncio

from ai_hydra.tui.client import HydraClient
from ai_hydra.zmq_protocol import ZMQMessage, MessageType


class TestTUIEpochIntegration:
    """Integration tests for TUI epoch display functionality."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.client = HydraClient(server_address="tcp://localhost:5555")
    
    @pytest.mark.asyncio
    async def test_epoch_display_end_to_end_workflow(self):
        """Test complete epoch display workflow from server communication to UI update."""
        # Mock ZeroMQ components
        with patch('zmq.asyncio.Context') as mock_context_class, \
             patch.object(self.client, 'query_one') as mock_query:
            
            # Setup mock context and socket
            mock_context = Mock()
            mock_socket = Mock()
            mock_context_class.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            
            # Mock socket operations
            mock_socket.connect = Mock()
            mock_socket.poll = AsyncMock(return_value=True)
            mock_socket.recv_string = AsyncMock()
            mock_socket.send_string = AsyncMock()
            
            # Mock UI components
            mock_epoch_label = Mock()
            mock_query.return_value = mock_epoch_label
            
            # Mock game board
            self.client.game_board = Mock()
            self.client.game_board.update_game_state = AsyncMock()
            
            # Simulate server response with epoch data
            server_response = {
                "message_type": "STATUS_RESPONSE",
                "timestamp": 1234567890.0,
                "client_id": "test-client",
                "request_id": "test-request",
                "data": {
                    "simulation_status": "running",
                    "game_state": {
                        "score": 50,
                        "snake_body": [[5, 5], [4, 5], [3, 5]],
                        "moves_count": 10,
                        "epoch": 3
                    }
                }
            }
            
            mock_socket.recv_string.return_value = json.dumps(server_response)
            
            # Connect to server
            await self.client.connect_to_server()
            
            # Simulate receiving status update
            response = await self.client.send_command(MessageType.GET_STATUS, {})
            
            if response:
                await self.client.process_status_update(response.data)
            
            # Verify epoch was updated
            assert self.client.current_epoch == 3
            
            # Trigger UI update
            self.client.watch_current_epoch(0, 3)
            
            # Verify UI was updated
            mock_epoch_label.update.assert_called_with("3")
    
    @pytest.mark.asyncio
    async def test_epoch_increment_across_multiple_games(self):
        """Test epoch increments correctly across multiple game cycles."""
        # Mock ZeroMQ components
        with patch('zmq.asyncio.Context') as mock_context_class, \
             patch.object(self.client, 'query_one') as mock_query:
            
            # Setup mocks
            mock_context = Mock()
            mock_socket = Mock()
            mock_context_class.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            
            mock_socket.connect = Mock()
            mock_socket.poll = AsyncMock(return_value=True)
            mock_socket.recv_string = AsyncMock()
            mock_socket.send_string = AsyncMock()
            
            # Mock UI components
            mock_epoch_label = Mock()
            mock_query.return_value = mock_epoch_label
            
            # Mock game board
            self.client.game_board = Mock()
            self.client.game_board.update_game_state = AsyncMock()
            
            # Connect to server
            await self.client.connect_to_server()
            
            # Simulate multiple game cycles with increasing epochs
            epochs = [1, 2, 3, 4, 5]
            
            for epoch in epochs:
                server_response = {
                    "message_type": "STATUS_RESPONSE",
                    "timestamp": 1234567890.0,
                    "client_id": "test-client",
                    "request_id": f"test-request-{epoch}",
                    "data": {
                        "simulation_status": "running",
                        "game_state": {
                            "score": epoch * 10,
                            "snake_body": [[5, 5], [4, 5], [3, 5]],
                            "moves_count": epoch * 5,
                            "epoch": epoch
                        }
                    }
                }
                
                mock_socket.recv_string.return_value = json.dumps(server_response)
                
                # Get status and process update
                response = await self.client.send_command(MessageType.GET_STATUS, {})
                if response:
                    await self.client.process_status_update(response.data)
                
                # Verify epoch progression
                assert self.client.current_epoch == epoch
                
                # Trigger UI update
                previous_epoch = epoch - 1 if epoch > 1 else 0
                self.client.watch_current_epoch(previous_epoch, epoch)
            
            # Verify all UI updates occurred
            assert mock_epoch_label.update.call_count == len(epochs)
            
            # Verify final epoch value
            assert self.client.current_epoch == 5
    
    @pytest.mark.asyncio
    async def test_epoch_reset_integration(self):
        """Test epoch reset integration with server communication."""
        # Mock ZeroMQ components
        with patch('zmq.asyncio.Context') as mock_context_class, \
             patch.object(self.client, 'query_one') as mock_query:
            
            # Setup mocks
            mock_context = Mock()
            mock_socket = Mock()
            mock_context_class.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            
            mock_socket.connect = Mock()
            mock_socket.poll = AsyncMock(return_value=True)
            mock_socket.recv_string = AsyncMock()
            mock_socket.send_string = AsyncMock()
            
            # Mock UI components
            mock_log = Mock()
            mock_log.clear = Mock()
            mock_query.return_value = mock_log
            
            # Mock game board
            self.client.game_board = Mock()
            self.client.game_board.reset = AsyncMock()
            self.client.game_board.update_game_state = AsyncMock()
            
            # Connect to server
            await self.client.connect_to_server()
            
            # Set initial epoch to non-zero value
            self.client.current_epoch = 10
            
            # Mock reset response
            reset_response = {
                "message_type": "SIMULATION_RESET",
                "timestamp": 1234567890.0,
                "client_id": "test-client",
                "request_id": "reset-request",
                "data": {}
            }
            
            mock_socket.recv_string.return_value = json.dumps(reset_response)
            
            # Perform reset
            await self.client.reset_simulation()
            
            # Verify epoch was reset to 0
            assert self.client.current_epoch == 0
            
            # Verify other values were also reset
            assert self.client.game_score == 0
            assert self.client.moves_count == 0
            assert self.client.snake_length == 3
            assert self.client.runtime_seconds == 0
    
    @pytest.mark.asyncio
    async def test_epoch_display_with_connection_errors(self):
        """Test epoch display behavior when connection errors occur."""
        # Mock ZeroMQ components with connection failure
        with patch('zmq.asyncio.Context') as mock_context_class, \
             patch.object(self.client, 'log_message') as mock_log:
            
            # Setup mock context that fails
            mock_context = Mock()
            mock_socket = Mock()
            mock_context_class.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            
            # Mock connection failure
            mock_socket.connect.side_effect = Exception("Connection failed")
            
            # Attempt to connect
            result = await self.client.connect_to_server()
            
            # Verify connection failed
            assert result is False
            assert self.client.is_connected is False
            
            # Verify error was logged
            mock_log.assert_called()
            args, kwargs = mock_log.call_args
            assert "Connection failed" in args[0]
            assert kwargs.get("level") == "error"
            
            # Epoch should remain at default value
            assert self.client.current_epoch == 0
    
    @pytest.mark.asyncio
    async def test_epoch_display_with_malformed_server_data(self):
        """Test epoch display handles malformed server data gracefully."""
        # Mock ZeroMQ components
        with patch('zmq.asyncio.Context') as mock_context_class, \
             patch.object(self.client, 'log_message') as mock_log:
            
            # Setup mocks
            mock_context = Mock()
            mock_socket = Mock()
            mock_context_class.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            
            mock_socket.connect = Mock()
            mock_socket.poll = AsyncMock(return_value=True)
            mock_socket.recv_string = AsyncMock()
            mock_socket.send_string = AsyncMock()
            
            # Mock game board
            self.client.game_board = Mock()
            self.client.game_board.update_game_state = AsyncMock()
            
            # Connect to server
            await self.client.connect_to_server()
            
            # Test various malformed data scenarios
            malformed_responses = [
                # Invalid JSON
                "invalid json data",
                # Missing required fields
                json.dumps({"message_type": "STATUS_RESPONSE"}),
                # Invalid game_state type
                json.dumps({
                    "message_type": "STATUS_RESPONSE",
                    "data": {"game_state": "invalid_type"}
                }),
                # Invalid epoch type
                json.dumps({
                    "message_type": "STATUS_RESPONSE",
                    "data": {
                        "game_state": {
                            "score": 10,
                            "epoch": "invalid_epoch_type"
                        }
                    }
                })
            ]
            
            for malformed_data in malformed_responses:
                mock_socket.recv_string.return_value = malformed_data
                
                # Should handle errors gracefully
                try:
                    response = await self.client.send_command(MessageType.GET_STATUS, {})
                    if response and hasattr(response, 'data'):
                        await self.client.process_status_update(response.data)
                except Exception:
                    # Some malformed data might cause exceptions in parsing
                    pass
                
                # Epoch should remain valid (non-negative integer)
                assert isinstance(self.client.current_epoch, int)
                assert self.client.current_epoch >= 0