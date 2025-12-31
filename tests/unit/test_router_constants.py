"""
Unit tests for AI Hydra Router Constants.
"""

import pytest

from ai_hydra.router_constants import RouterConstants, RouterLabels


class TestRouterConstants:
    """Unit tests for RouterConstants class."""
    
    def test_client_server_types(self):
        """Test client and server type constants."""
        assert RouterConstants.HYDRA_CLIENT == "HydraClient"
        assert RouterConstants.HYDRA_SERVER == "HydraServer"
        assert RouterConstants.HYDRA_ROUTER == "HydraRouter"
        
        # Ensure they are different
        assert RouterConstants.HYDRA_CLIENT != RouterConstants.HYDRA_SERVER
        assert RouterConstants.HYDRA_CLIENT != RouterConstants.HYDRA_ROUTER
        assert RouterConstants.HYDRA_SERVER != RouterConstants.HYDRA_ROUTER
    
    def test_message_structure_keys(self):
        """Test message structure key constants."""
        expected_keys = [
            "SENDER", "ELEM", "DATA", "CLIENT_ID", 
            "TIMESTAMP", "REQUEST_ID", "MESSAGE_TYPE"
        ]
        
        for key in expected_keys:
            assert hasattr(RouterConstants, key)
            assert isinstance(getattr(RouterConstants, key), str)
            assert len(getattr(RouterConstants, key)) > 0
    
    def test_system_messages(self):
        """Test system message constants."""
        assert RouterConstants.HEARTBEAT == "heartbeat"
        assert RouterConstants.STATUS == "status"
        assert RouterConstants.ERROR == "error"
        assert RouterConstants.OK == "ok"
        
        # Ensure they are unique
        system_messages = [
            RouterConstants.HEARTBEAT,
            RouterConstants.STATUS,
            RouterConstants.ERROR,
            RouterConstants.OK
        ]
        assert len(system_messages) == len(set(system_messages))
    
    def test_simulation_control_commands(self):
        """Test simulation control command constants."""
        commands = [
            RouterConstants.START_SIMULATION,
            RouterConstants.STOP_SIMULATION,
            RouterConstants.PAUSE_SIMULATION,
            RouterConstants.RESUME_SIMULATION,
            RouterConstants.RESET_SIMULATION
        ]
        
        # All should be strings
        for command in commands:
            assert isinstance(command, str)
            assert len(command) > 0
        
        # All should be unique
        assert len(commands) == len(set(commands))
        
        # Check specific values
        assert RouterConstants.START_SIMULATION == "start_simulation"
        assert RouterConstants.STOP_SIMULATION == "stop_simulation"
        assert RouterConstants.PAUSE_SIMULATION == "pause_simulation"
        assert RouterConstants.RESUME_SIMULATION == "resume_simulation"
        assert RouterConstants.RESET_SIMULATION == "reset_simulation"
    
    def test_simulation_status_messages(self):
        """Test simulation status message constants."""
        status_messages = [
            RouterConstants.SIMULATION_STARTED,
            RouterConstants.SIMULATION_STOPPED,
            RouterConstants.SIMULATION_PAUSED,
            RouterConstants.SIMULATION_RESUMED,
            RouterConstants.SIMULATION_RESET
        ]
        
        # All should be strings
        for message in status_messages:
            assert isinstance(message, str)
            assert len(message) > 0
        
        # All should be unique
        assert len(status_messages) == len(set(status_messages))
        
        # Check specific values
        assert RouterConstants.SIMULATION_STARTED == "simulation_started"
        assert RouterConstants.SIMULATION_STOPPED == "simulation_stopped"
        assert RouterConstants.SIMULATION_PAUSED == "simulation_paused"
        assert RouterConstants.SIMULATION_RESUMED == "simulation_resumed"
        assert RouterConstants.SIMULATION_RESET == "simulation_reset"
    
    def test_data_messages(self):
        """Test data message constants."""
        data_messages = [
            RouterConstants.GET_STATUS,
            RouterConstants.STATUS_UPDATE,
            RouterConstants.GAME_STATE_UPDATE,
            RouterConstants.PERFORMANCE_UPDATE
        ]
        
        for message in data_messages:
            assert isinstance(message, str)
            assert len(message) > 0
        
        assert len(data_messages) == len(set(data_messages))
    
    def test_configuration_messages(self):
        """Test configuration message constants."""
        config_messages = [
            RouterConstants.GET_CONFIG,
            RouterConstants.SET_CONFIG,
            RouterConstants.CONFIG_UPDATE
        ]
        
        for message in config_messages:
            assert isinstance(message, str)
            assert len(message) > 0
        
        assert len(config_messages) == len(set(config_messages))
    
    def test_timing_constants(self):
        """Test timing-related constants."""
        assert isinstance(RouterConstants.HEARTBEAT_INTERVAL, int)
        assert RouterConstants.HEARTBEAT_INTERVAL > 0
        assert RouterConstants.HEARTBEAT_INTERVAL == 5
    
    def test_network_constants(self):
        """Test network-related constants."""
        assert isinstance(RouterConstants.DEFAULT_ROUTER_PORT, int)
        assert RouterConstants.DEFAULT_ROUTER_PORT > 0
        assert RouterConstants.DEFAULT_ROUTER_PORT == 5556
        
        assert isinstance(RouterConstants.DEFAULT_SERVER_PORT, int)
        assert RouterConstants.DEFAULT_SERVER_PORT > 0
        assert RouterConstants.DEFAULT_SERVER_PORT == 5555
        
        assert RouterConstants.PROTOCOL == "tcp"
        
        # Ports should be different
        assert RouterConstants.DEFAULT_ROUTER_PORT != RouterConstants.DEFAULT_SERVER_PORT


class TestRouterLabels:
    """Unit tests for RouterLabels class."""
    
    def test_message_labels_exist(self):
        """Test that all expected labels exist."""
        expected_labels = [
            "STARTUP_MSG", "SHUTDOWN_MSG", "MALFORMED_MESSAGE",
            "ROUTER_ERROR", "UNKNOWN_SENDER", "NO_SERVER_CONNECTED",
            "CLIENT_DISCONNECTED", "SERVER_DISCONNECTED"
        ]
        
        for label in expected_labels:
            assert hasattr(RouterLabels, label)
            assert isinstance(getattr(RouterLabels, label), str)
            assert len(getattr(RouterLabels, label)) > 0
    
    def test_message_label_formatting(self):
        """Test that labels support string formatting where expected."""
        # Test labels that expect formatting
        assert "%s" in RouterLabels.STARTUP_MSG
        assert "%s" in RouterLabels.ROUTER_ERROR
        
        # Test formatting works
        formatted_startup = RouterLabels.STARTUP_MSG % "tcp://127.0.0.1:5556"
        assert "tcp://127.0.0.1:5556" in formatted_startup
        
        formatted_error = RouterLabels.ROUTER_ERROR % "Test error message"
        assert "Test error message" in formatted_error
    
    def test_label_content(self):
        """Test specific label content."""
        assert "Router running" in RouterLabels.STARTUP_MSG
        assert "shutting down" in RouterLabels.SHUTDOWN_MSG.lower()
        assert "malformed" in RouterLabels.MALFORMED_MESSAGE.lower()
        assert "error" in RouterLabels.ROUTER_ERROR.lower()
        assert "unknown sender" in RouterLabels.UNKNOWN_SENDER.lower()
        assert "no" in RouterLabels.NO_SERVER_CONNECTED.lower()
        assert "server" in RouterLabels.NO_SERVER_CONNECTED.lower()
    
    def test_labels_are_human_readable(self):
        """Test that labels are human-readable."""
        labels = [
            RouterLabels.STARTUP_MSG,
            RouterLabels.SHUTDOWN_MSG,
            RouterLabels.MALFORMED_MESSAGE,
            RouterLabels.UNKNOWN_SENDER,
            RouterLabels.NO_SERVER_CONNECTED,
            RouterLabels.CLIENT_DISCONNECTED,
            RouterLabels.SERVER_DISCONNECTED
        ]
        
        for label in labels:
            # Should contain spaces (human readable)
            if "%s" not in label:  # Skip format strings
                assert " " in label or label.isupper()  # Either spaced or constant-style
            
            # Should not be empty
            assert len(label.strip()) > 0
            
            # Should not contain obvious code artifacts
            assert "{" not in label
            assert "}" not in label
            assert "def " not in label
            assert "class " not in label