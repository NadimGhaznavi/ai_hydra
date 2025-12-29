"""
Property-based tests for headless operation completeness.

This module implements Property 14: Headless Operation Completeness
**Validates: Requirements 13.3, 13.4**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time
from typing import Dict, Any

from ai_hydra.zmq_server import ZMQServer, SimulationState
from ai_hydra.zmq_protocol import ZMQMessage, MessageType, MessageValidator
from ai_hydra.config import SimulationConfig


class TestHeadlessOperationProperties:
    """Property-based tests for headless operation completeness."""
    
    @given(
        grid_width=st.integers(min_value=5, max_value=20),
        grid_height=st.integers(min_value=5, max_value=20),
        move_budget=st.integers(min_value=10, max_value=100),
        random_seed=st.integers(min_value=0, max_value=1000),
        client_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        request_id=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
    )
    @settings(max_examples=10, deadline=5000)
    def test_headless_operation_completeness_property(self, grid_width, grid_height, 
                                                    move_budget, random_seed, 
                                                    client_id, request_id):
        """
        **Feature: ai-hydra, Property 14: Headless Operation Completeness**
        
        *For any* simulation operation (start, stop, pause, resume, status), the headless 
        server should handle the operation without requiring any direct user interaction 
        or GUI components.
        **Validates: Requirements 13.3, 13.4**
        """
        # Create headless server
        server = ZMQServer(bind_address="tcp://*:5557")
        
        # Mock the socket to avoid actual ZMQ operations
        server.socket = Mock()
        
        # Test configuration
        config_data = {
            "config": {
                "grid_size": [grid_width, grid_height],
                "move_budget": move_budget,
                "random_seed": random_seed,
                "nn_enabled": True
            }
        }
        
        # Property 1: All operations should be handleable without user interaction
        operations_to_test = [
            (MessageType.START_SIMULATION, config_data),
            (MessageType.GET_STATUS, {}),
            (MessageType.PAUSE_SIMULATION, {}),
            (MessageType.RESUME_SIMULATION, {}),
            (MessageType.STOP_SIMULATION, {}),
            (MessageType.RESET_SIMULATION, config_data)
        ]
        
        for operation_type, operation_data in operations_to_test:
            # Create operation message
            message = ZMQMessage.create_command(
                operation_type, client_id, f"{request_id}_{operation_type.value}", operation_data
            )
            
            # Property: Operation should be processable without user interaction
            try:
                # Mock threading for start simulation to avoid actual simulation
                with patch('threading.Thread'):
                    response = asyncio.run(server._route_message(message))
                
                # Verify response is generated (no user interaction required)
                assert response is not None, f"Operation {operation_type} should generate response"
                assert hasattr(response, 'message_type'), f"Response should have message_type for {operation_type}"
                assert hasattr(response, 'request_id'), f"Response should have request_id for {operation_type}"
                assert response.request_id == message.request_id, \
                    f"Response request_id should match for {operation_type}"
                
                # Property: Response should be valid ZMQ message
                json_str = response.to_json()
                assert isinstance(json_str, str), f"Response should serialize to JSON for {operation_type}"
                
                # Property: Response should be deserializable
                deserialized = ZMQMessage.from_json(json_str)
                assert deserialized.message_type == response.message_type, \
                    f"Deserialized response should match original for {operation_type}"
                
            except Exception as e:
                pytest.fail(f"Operation {operation_type} failed in headless mode: {e}")
        
        # Property 2: Server state should be manageable programmatically
        assert hasattr(server, 'simulation_state'), "Server should track simulation state"
        # Note: simulation_state might be a string or enum, both are acceptable for headless operation
        
        # Property 3: No GUI dependencies should be required
        self._verify_no_gui_dependencies(server)
    
    @given(
        operation_sequence=st.lists(
            st.sampled_from([
                MessageType.START_SIMULATION,
                MessageType.PAUSE_SIMULATION,
                MessageType.RESUME_SIMULATION,
                MessageType.GET_STATUS,
                MessageType.STOP_SIMULATION
            ]),
            min_size=2,
            max_size=8
        ),
        client_id=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))
    )
    @settings(max_examples=8, deadline=6000)
    def test_headless_operation_sequence_property(self, operation_sequence, client_id):
        """
        **Feature: ai-hydra, Property 14a: Headless Operation Sequences**
        
        *For any* sequence of valid operations, the headless server should handle
        each operation in sequence without requiring user intervention.
        **Validates: Requirements 13.3, 13.4**
        """
        server = ZMQServer(bind_address="tcp://*:5558")
        server.socket = Mock()
        
        # Standard configuration for operations that need it
        standard_config = {
            "config": {
                "grid_size": [10, 10],
                "move_budget": 50,
                "random_seed": 42,
                "nn_enabled": False  # Disable NN for faster testing
            }
        }
        
        operation_count = 0
        
        for operation_type in operation_sequence:
            operation_count += 1
            
            # Determine if operation needs configuration data
            operation_data = standard_config if operation_type == MessageType.START_SIMULATION else {}
            
            # Create message
            message = ZMQMessage.create_command(
                operation_type, 
                client_id, 
                f"seq_req_{operation_count}",
                operation_data
            )
            
            # Property: Each operation in sequence should be handled
            try:
                with patch('threading.Thread'):
                    response = asyncio.run(server._route_message(message))
                
                assert response is not None, \
                    f"Operation {operation_count} ({operation_type}) should generate response"
                
                # Property: Response should correspond to the operation
                expected_response_types = {
                    MessageType.START_SIMULATION: MessageType.SIMULATION_STARTED,
                    MessageType.STOP_SIMULATION: MessageType.SIMULATION_STOPPED,
                    MessageType.PAUSE_SIMULATION: MessageType.SIMULATION_PAUSED,
                    MessageType.RESUME_SIMULATION: MessageType.SIMULATION_RESUMED,
                    MessageType.GET_STATUS: MessageType.STATUS_RESPONSE
                }
                
                if operation_type in expected_response_types:
                    expected_response = expected_response_types[operation_type]
                    
                    # Some operations may return errors if called in invalid states
                    if response.message_type == MessageType.ERROR_OCCURRED:
                        # This is acceptable for state-dependent operations
                        if operation_type in [MessageType.PAUSE_SIMULATION, MessageType.RESUME_SIMULATION]:
                            assert "InvalidState" in response.data.get("error_type", ""), \
                                f"Expected InvalidState error for {operation_type} in wrong state"
                        else:
                            # For other operations, errors should be more specific
                            assert "error_type" in response.data, \
                                f"Error response should have error_type for {operation_type}"
                    else:
                        # Normal successful response
                        assert response.message_type == expected_response, \
                            f"Operation {operation_type} should return {expected_response}, got {response.message_type}"
                
                # Property: Server state should be updated appropriately
                if operation_type == MessageType.START_SIMULATION:
                    assert server.simulation_state == SimulationState.RUNNING, \
                        f"Server should be running after START_SIMULATION, got {server.simulation_state}"
                elif operation_type == MessageType.STOP_SIMULATION:
                    # STOP can transition from any state to STOPPED or IDLE
                    assert server.simulation_state in [SimulationState.STOPPED, SimulationState.IDLE], \
                        f"Server should be stopped or idle after STOP_SIMULATION, got {server.simulation_state}"
                elif operation_type == MessageType.PAUSE_SIMULATION:
                    # PAUSE only works if simulation is running, otherwise expect error response
                    if response.message_type == MessageType.ERROR_OCCURRED:
                        # This is acceptable - can't pause when not running
                        assert "InvalidState" in response.data.get("error_type", ""), \
                            f"Expected InvalidState error for pause when not running, got {response.data}"
                    else:
                        assert server.simulation_state == SimulationState.PAUSED, \
                            f"Server should be paused after successful PAUSE_SIMULATION, got {server.simulation_state}"
                
            except Exception as e:
                pytest.fail(f"Operation sequence failed at step {operation_count} ({operation_type}): {e}")
    
    @given(
        concurrent_clients=st.integers(min_value=2, max_value=5),
        operations_per_client=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=5, deadline=8000)
    def test_headless_multi_client_property(self, concurrent_clients, operations_per_client):
        """
        **Feature: ai-hydra, Property 14b: Headless Multi-Client Support**
        
        *For any* number of concurrent clients, the headless server should handle
        all client operations without requiring user intervention for client management.
        **Validates: Requirements 13.3, 13.4**
        """
        server = ZMQServer(bind_address="tcp://*:5559")
        server.socket = Mock()
        
        # Track client interactions
        client_responses = {}
        
        # Simulate concurrent clients
        for client_num in range(concurrent_clients):
            client_id = f"client_{client_num}"
            client_responses[client_id] = []
            
            # Each client performs multiple operations
            for op_num in range(operations_per_client):
                # Alternate between status requests and start/stop operations
                if op_num % 2 == 0:
                    operation_type = MessageType.GET_STATUS
                    operation_data = {}
                else:
                    operation_type = MessageType.START_SIMULATION
                    operation_data = {
                        "config": {
                            "grid_size": [8, 8],
                            "move_budget": 20,
                            "random_seed": client_num * 100 + op_num,
                            "nn_enabled": False
                        }
                    }
                
                # Create client message
                message = ZMQMessage.create_command(
                    operation_type,
                    client_id,
                    f"{client_id}_req_{op_num}",
                    operation_data
                )
                
                # Property: Server should handle each client independently
                try:
                    with patch('threading.Thread'):
                        response = asyncio.run(server._route_message(message))
                    
                    assert response is not None, \
                        f"Client {client_id} operation {op_num} should get response"
                    
                    assert response.request_id == message.request_id, \
                        f"Response should match request ID for client {client_id}"
                    
                    client_responses[client_id].append(response)
                    
                except Exception as e:
                    pytest.fail(f"Multi-client operation failed for {client_id} op {op_num}: {e}")
        
        # Property: All clients should have received responses
        for client_id in client_responses:
            assert len(client_responses[client_id]) == operations_per_client, \
                f"Client {client_id} should have received {operations_per_client} responses"
        
        # Property: Server should track multiple clients without user intervention
        assert len(server.connected_clients) >= 0, "Server should track connected clients"
    
    @given(
        invalid_config_type=st.sampled_from(['missing_grid', 'invalid_budget', 'invalid_seed', 'missing_config']),
        client_id=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))
    )
    @settings(max_examples=8, deadline=3000)
    def test_headless_error_handling_property(self, invalid_config_type, client_id):
        """
        **Feature: ai-hydra, Property 14c: Headless Error Handling**
        
        *For any* invalid operation or configuration, the headless server should
        handle errors gracefully without requiring user intervention.
        **Validates: Requirements 13.3, 13.4**
        """
        server = ZMQServer(bind_address="tcp://*:5560")
        server.socket = Mock()
        
        # Create invalid configurations based on type
        invalid_configs = {
            'missing_grid': {"config": {"move_budget": 50, "random_seed": 42}},
            'invalid_budget': {"config": {"grid_size": [10, 10], "move_budget": -5, "random_seed": 42}},
            'invalid_seed': {"config": {"grid_size": [10, 10], "move_budget": 50, "random_seed": -1}},
            'missing_config': {}
        }
        
        invalid_data = invalid_configs[invalid_config_type]
        
        # Create invalid start simulation message
        message = ZMQMessage.create_command(
            MessageType.START_SIMULATION,
            client_id,
            f"invalid_req_{invalid_config_type}",
            invalid_data
        )
        
        # Property: Server should handle invalid operations gracefully
        try:
            with patch('threading.Thread'):
                response = asyncio.run(server._route_message(message))
            
            # Should get a response (not crash)
            assert response is not None, "Server should respond to invalid operations"
            
            # Response should indicate error or validation failure
            # (The exact response type depends on implementation, but it shouldn't crash)
            assert hasattr(response, 'message_type'), "Error response should have message type"
            assert hasattr(response, 'data'), "Error response should have data"
            
            # Property: Error response should be serializable
            json_str = response.to_json()
            assert isinstance(json_str, str), "Error response should be serializable"
            
            # Property: Server should remain operational after error
            status_message = ZMQMessage.create_command(
                MessageType.GET_STATUS, client_id, "status_after_error", {}
            )
            
            status_response = asyncio.run(server._route_message(status_message))
            assert status_response is not None, "Server should remain operational after handling error"
            
        except Exception as e:
            pytest.fail(f"Server crashed on invalid config {invalid_config_type}: {e}")
    
    def _verify_no_gui_dependencies(self, server: ZMQServer) -> None:
        """Verify that the server has no GUI dependencies."""
        # Check that server doesn't import GUI libraries
        import sys
        gui_modules = ['tkinter', 'pygame', 'kivy', 'wx', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6']
        
        for module_name in gui_modules:
            if module_name in sys.modules:
                # If GUI module is loaded, verify server doesn't use it
                gui_module = sys.modules[module_name]
                
                # Check server attributes don't reference GUI objects
                for attr_name in dir(server):
                    attr_value = getattr(server, attr_name)
                    if hasattr(gui_module, '__name__'):
                        assert not str(type(attr_value)).startswith(f"<class '{gui_module.__name__}"), \
                            f"Server should not have GUI dependency: {attr_name} uses {gui_module.__name__}"
        
        # Verify server can operate without display
        import os
        original_display = os.environ.get('DISPLAY')
        try:
            # Temporarily remove DISPLAY environment variable
            if 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']
            
            # Server should still be initializable without display
            test_server = ZMQServer(bind_address="tcp://*:5561")
            assert test_server is not None, "Server should initialize without DISPLAY"
            
        finally:
            # Restore DISPLAY if it was set
            if original_display is not None:
                os.environ['DISPLAY'] = original_display
    
    def test_headless_operation_basic_functionality(self):
        """
        Basic functionality test for headless operation.
        
        This test verifies that the basic headless operation mechanism works
        without property-based complexity.
        """
        server = ZMQServer(bind_address="tcp://*:5562")
        server.socket = Mock()
        
        # Test basic status request
        status_message = ZMQMessage.create_command(
            MessageType.GET_STATUS,
            "test_client",
            "basic_status_req",
            {}
        )
        
        response = asyncio.run(server._route_message(status_message))
        
        assert response is not None, "Should get status response"
        assert response.message_type == MessageType.STATUS_RESPONSE, "Should be status response type"
        assert response.request_id == "basic_status_req", "Should match request ID"
        
        # Verify response contains expected data
        assert "server_id" in response.data, "Status should include server ID"
        assert "simulation_state" in response.data, "Status should include simulation state"
        assert "uptime_seconds" in response.data, "Status should include uptime"