"""
Property-based tests for TUI epoch display functionality.
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock, patch

from ai_hydra.tui.client import HydraClient


class TestTUIEpochDisplayProperties:
    """Property-based tests for TUI epoch display."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.client = HydraClient(server_address="tcp://localhost:5555")
    
    @given(epoch_value=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=1000)
    def test_epoch_display_property(self, epoch_value):
        """
        **Feature: tui-client, Property 5: Status Information Display**
        **Validates: Requirements 3.5, 3.6**
        
        For any valid epoch value, the status display should show the current 
        epoch number and update the UI correctly.
        """
        # Mock the UI label
        with patch.object(self.client, 'query_one') as mock_query:
            mock_epoch_label = Mock()
            mock_query.return_value = mock_epoch_label
            
            # Update epoch value
            self.client.current_epoch = epoch_value
            
            # Trigger the watcher
            self.client.watch_current_epoch(0, epoch_value)
            
            # Verify UI was updated with correct value
            mock_query.assert_called_once_with("#current_epoch", Mock)
            mock_epoch_label.update.assert_called_once_with(str(epoch_value))
    
    @given(
        game_states=st.lists(
            st.fixed_dictionaries({
                "score": st.integers(min_value=0, max_value=1000),
                "snake_body": st.lists(
                    st.tuples(st.integers(0, 19), st.integers(0, 19)),
                    min_size=3, max_size=20
                ),
                "moves_count": st.integers(min_value=0, max_value=500),
                "epoch": st.integers(min_value=1, max_value=100)
            }),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=50, deadline=2000)
    @pytest.mark.asyncio
    async def test_epoch_increment_property(self, game_states):
        """
        **Feature: tui-client, Property 5: Status Information Display**
        **Validates: Requirements 3.5, 3.6**
        
        For any sequence of game states with increasing epochs, the epoch 
        display should correctly track and display the progression.
        """
        # Mock game board
        self.client.game_board = Mock()
        self.client.game_board.update_game_state = AsyncMock()
        
        previous_epoch = 0
        
        for game_state in game_states:
            # Ensure epoch is monotonically increasing
            if game_state["epoch"] <= previous_epoch:
                game_state["epoch"] = previous_epoch + 1
            
            status_data = {
                "simulation_status": "running",
                "game_state": game_state
            }
            
            # Process the status update
            await self.client.process_status_update(status_data)
            
            # Verify epoch was updated correctly
            assert self.client.current_epoch == game_state["epoch"]
            assert self.client.current_epoch > previous_epoch
            
            previous_epoch = game_state["epoch"]
    
    @given(
        status_updates=st.lists(
            st.one_of(
                # Valid status with epoch
                st.fixed_dictionaries({
                    "simulation_status": st.sampled_from(["idle", "running", "paused", "stopped"]),
                    "game_state": st.fixed_dictionaries({
                        "score": st.integers(min_value=0, max_value=1000),
                        "snake_body": st.lists(
                            st.tuples(st.integers(0, 19), st.integers(0, 19)),
                            min_size=1, max_size=20
                        ),
                        "moves_count": st.integers(min_value=0, max_value=500),
                        "epoch": st.integers(min_value=0, max_value=100)
                    })
                }),
                # Valid status without epoch
                st.fixed_dictionaries({
                    "simulation_status": st.sampled_from(["idle", "running", "paused", "stopped"]),
                    "game_state": st.fixed_dictionaries({
                        "score": st.integers(min_value=0, max_value=1000),
                        "snake_body": st.lists(
                            st.tuples(st.integers(0, 19), st.integers(0, 19)),
                            min_size=1, max_size=20
                        ),
                        "moves_count": st.integers(min_value=0, max_value=500)
                        # No epoch field
                    })
                }),
                # Status with None game_state
                st.fixed_dictionaries({
                    "simulation_status": st.sampled_from(["idle", "running", "paused", "stopped"]),
                    "game_state": st.none()
                })
            ),
            min_size=1, max_size=20
        )
    )
    @settings(max_examples=50, deadline=2000)
    @pytest.mark.asyncio
    async def test_epoch_robustness_property(self, status_updates):
        """
        **Feature: tui-client, Property 5: Status Information Display**
        **Validates: Requirements 3.5, 3.6**
        
        For any sequence of status updates (valid or missing epoch data), 
        the epoch display should handle all cases gracefully without errors.
        """
        # Mock game board
        self.client.game_board = Mock()
        self.client.game_board.update_game_state = AsyncMock()
        
        for status_data in status_updates:
            # Should not raise any exceptions
            await self.client.process_status_update(status_data)
            
            # Epoch should always be a non-negative integer
            assert isinstance(self.client.current_epoch, int)
            assert self.client.current_epoch >= 0
            
            # If game_state exists and has epoch, it should be used
            if (status_data.get("game_state") is not None and 
                isinstance(status_data["game_state"], dict) and
                "epoch" in status_data["game_state"]):
                assert self.client.current_epoch == status_data["game_state"]["epoch"]
    
    @given(
        initial_epoch=st.integers(min_value=0, max_value=100),
        reset_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=2000)
    @pytest.mark.asyncio
    async def test_epoch_reset_property(self, initial_epoch, reset_count):
        """
        **Feature: tui-client, Property 5: Status Information Display**
        **Validates: Requirements 3.5, 3.6**
        
        For any initial epoch value, resetting the simulation should always 
        set the epoch back to 0, regardless of how many times reset is called.
        """
        # Set initial epoch
        self.client.current_epoch = initial_epoch
        
        # Mock the send_command method and UI components
        with patch.object(self.client, 'send_command') as mock_send, \
             patch.object(self.client, 'query_one') as mock_query:
            
            from ai_hydra.zmq_protocol import MessageType
            
            # Mock successful reset response
            mock_response = Mock()
            mock_response.message_type = MessageType.SIMULATION_RESET
            mock_send.return_value = mock_response
            
            # Mock log widget
            mock_log = Mock()
            mock_log.clear = Mock()
            mock_query.return_value = mock_log
            
            # Mock game board
            self.client.game_board = Mock()
            self.client.game_board.reset = AsyncMock()
            
            # Perform multiple resets
            for _ in range(reset_count):
                await self.client.reset_simulation()
                
                # Epoch should always be 0 after reset
                assert self.client.current_epoch == 0
    
    @given(
        epoch_sequence=st.lists(
            st.integers(min_value=0, max_value=1000),
            min_size=1, max_size=50
        )
    )
    @settings(max_examples=50, deadline=2000)
    def test_epoch_ui_update_property(self, epoch_sequence):
        """
        **Feature: tui-client, Property 5: Status Information Display**
        **Validates: Requirements 3.5, 3.6**
        
        For any sequence of epoch values, the UI should be updated correctly 
        for each value change, displaying the epoch as a string.
        """
        with patch.object(self.client, 'query_one') as mock_query:
            mock_epoch_label = Mock()
            mock_query.return_value = mock_epoch_label
            
            previous_epoch = 0
            
            for epoch in epoch_sequence:
                # Update epoch
                self.client.current_epoch = epoch
                
                # Trigger watcher
                self.client.watch_current_epoch(previous_epoch, epoch)
                
                # Verify UI update was called with string representation
                mock_epoch_label.update.assert_called_with(str(epoch))
                
                previous_epoch = epoch
            
            # Verify UI was updated for each epoch change
            assert mock_epoch_label.update.call_count == len(epoch_sequence)