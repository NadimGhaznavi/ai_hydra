"""
Unit tests for TUI status display functionality including epoch tracking.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from textual.widgets import Label

from ai_hydra.tui.client import HydraClient


class TestTUIStatusDisplay:
    """Unit tests for TUI status display widget."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.client = HydraClient(server_address="tcp://localhost:5555")
        
    def test_status_display_initialization(self):
        """Test that status display initializes with correct default values."""
        assert self.client.simulation_state == "idle"
        assert self.client.game_score == 0
        assert self.client.snake_length == 3
        assert self.client.moves_count == 0
        assert self.client.runtime_seconds == 0
        assert self.client.current_epoch == 0
    
    def test_epoch_reactive_variable_update(self):
        """Test that epoch reactive variable updates correctly."""
        # Test initial value
        assert self.client.current_epoch == 0
        
        # Test updating epoch
        self.client.current_epoch = 5
        assert self.client.current_epoch == 5
        
        # Test epoch increment
        self.client.current_epoch += 1
        assert self.client.current_epoch == 6
    
    @patch('ai_hydra.tui.client.HydraClient.query_one')
    def test_epoch_watcher_updates_ui(self, mock_query_one):
        """Test that epoch watcher updates the UI label correctly."""
        # Mock the epoch label
        mock_epoch_label = Mock(spec=Label)
        mock_query_one.return_value = mock_epoch_label
        
        # Trigger epoch watcher
        self.client.watch_current_epoch(0, 5)
        
        # Verify UI was updated
        mock_query_one.assert_called_once_with("#current_epoch", Label)
        mock_epoch_label.update.assert_called_once_with("5")
    
    @patch('ai_hydra.tui.client.HydraClient.query_one')
    def test_epoch_watcher_handles_ui_errors(self, mock_query_one):
        """Test that epoch watcher handles UI errors gracefully."""
        # Mock query_one to raise an exception
        mock_query_one.side_effect = Exception("UI not ready")
        
        # Should not raise exception
        self.client.watch_current_epoch(0, 5)
        
        # Verify query was attempted
        mock_query_one.assert_called_once_with("#current_epoch", Label)
    
    @pytest.mark.asyncio
    async def test_process_status_update_with_epoch(self):
        """Test processing status update that includes epoch information."""
        # Mock game board
        self.client.game_board = Mock()
        self.client.game_board.update_game_state = AsyncMock()
        
        # Test status data with epoch
        status_data = {
            "simulation_status": "running",
            "game_state": {
                "score": 150,
                "snake_body": [[1, 2], [2, 2], [3, 2], [4, 2]],
                "moves_count": 25,
                "epoch": 3
            }
        }
        
        # Process the update
        await self.client.process_status_update(status_data)
        
        # Verify all values were updated
        assert self.client.simulation_state == "running"
        assert self.client.game_score == 150
        assert self.client.snake_length == 4
        assert self.client.moves_count == 25
        assert self.client.current_epoch == 3
    
    @pytest.mark.asyncio
    async def test_process_status_update_without_epoch(self):
        """Test processing status update that doesn't include epoch information."""
        # Mock game board
        self.client.game_board = Mock()
        self.client.game_board.update_game_state = AsyncMock()
        
        # Test status data without epoch
        status_data = {
            "simulation_status": "running",
            "game_state": {
                "score": 100,
                "snake_body": [[1, 2], [2, 2], [3, 2]],
                "moves_count": 15
                # No epoch field
            }
        }
        
        # Process the update
        await self.client.process_status_update(status_data)
        
        # Verify epoch defaults to 0 when not provided
        assert self.client.current_epoch == 0
        assert self.client.game_score == 100
        assert self.client.snake_length == 3
        assert self.client.moves_count == 15
    
    @pytest.mark.asyncio
    async def test_reset_simulation_resets_epoch(self):
        """Test that reset simulation resets epoch to 0."""
        # Set initial values
        self.client.current_epoch = 5
        self.client.game_score = 100
        self.client.moves_count = 50
        
        # Mock the send_command method to return successful reset
        with patch.object(self.client, 'send_command') as mock_send:
            from ai_hydra.zmq_protocol import ZMQMessage, MessageType
            
            mock_response = Mock()
            mock_response.message_type = MessageType.SIMULATION_RESET
            mock_send.return_value = mock_response
            
            # Mock UI components
            with patch.object(self.client, 'query_one') as mock_query:
                mock_log = Mock()
                mock_log.clear = Mock()
                mock_query.return_value = mock_log
                
                # Mock game board
                self.client.game_board = Mock()
                self.client.game_board.reset = AsyncMock()
                
                # Reset simulation
                await self.client.reset_simulation()
        
        # Verify all values were reset including epoch
        assert self.client.current_epoch == 0
        assert self.client.game_score == 0
        assert self.client.moves_count == 0
        assert self.client.snake_length == 3
        assert self.client.runtime_seconds == 0
    
    def test_format_runtime_display(self):
        """Test runtime formatting for display."""
        # Test various runtime values
        assert self.client.format_runtime(0) == "00:00:00"
        assert self.client.format_runtime(59) == "00:00:59"
        assert self.client.format_runtime(60) == "00:01:00"
        assert self.client.format_runtime(3661) == "01:01:01"
        assert self.client.format_runtime(7200) == "02:00:00"
    
    @pytest.mark.asyncio
    async def test_status_update_error_handling(self):
        """Test error handling in status update processing."""
        # Test with malformed status data
        malformed_data = {
            "game_state": "invalid_data_type"  # Should be dict, not string
        }
        
        # Mock log_message to capture error
        with patch.object(self.client, 'log_message') as mock_log:
            await self.client.process_status_update(malformed_data)
            
            # Verify error was logged
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert "Error processing status update" in args[0]
            assert kwargs.get("level") == "error"