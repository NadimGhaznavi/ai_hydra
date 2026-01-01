"""
Integration tests for Hydra Router system.

Tests the integration between HydraRouter, MQClient, and message validation
components to ensure they work together correctly.
"""

import asyncio
import pytest
import time
from typing import Dict, Any

from ai_hydra.router import HydraRouter
from ai_hydra.mq_client import MQClient
from ai_hydra.zmq_protocol import ZMQMessage, MessageType
from ai_hydra.router_constants import RouterConstants


class TestRouterIntegration:
    """Integration tests for router system components."""

    @pytest.mark.asyncio
    async def test_router_startup_and_shutdown(self):
        """Test router can start up and shut down cleanly."""
        router = HydraRouter(
            router_address="127.0.0.1", router_port=0, log_level="INFO"
        )

        try:
            # Start the router
            await router.start()
            assert router.is_running is True

            # Verify components are initialized
            assert router.validator is not None
            assert router.client_registry is not None
            assert router.message_router is not None
            assert router.task_manager is not None

        finally:
            # Clean shutdown
            await router.shutdown()
            assert router.is_running is False

    @pytest.mark.asyncio
    async def test_client_registry_operations(self):
        """Test client registry operations."""
        router = HydraRouter(
            router_address="127.0.0.1", router_port=0, log_level="INFO"
        )

        try:
            await router.start()

            # Test client registration
            await router.client_registry.register_client(
                "test-client-1", RouterConstants.HYDRA_CLIENT
            )
            await router.client_registry.register_client(
                "test-server-1", RouterConstants.HYDRA_SERVER
            )

            # Test getting clients by type
            clients = await router.client_registry.get_clients_by_type(
                RouterConstants.HYDRA_CLIENT
            )
            assert "test-client-1" in clients

            servers = await router.client_registry.get_clients_by_type(
                RouterConstants.HYDRA_SERVER
            )
            assert "test-server-1" in servers

            # Test client counts
            counts = await router.client_registry.get_client_count()
            assert counts.get(RouterConstants.HYDRA_CLIENT, 0) == 1
            assert counts.get(RouterConstants.HYDRA_SERVER, 0) == 1

            # Test heartbeat update
            await router.client_registry.update_heartbeat("test-client-1")

            # Test client removal
            await router.client_registry.remove_client("test-client-1")
            clients = await router.client_registry.get_clients_by_type(
                RouterConstants.HYDRA_CLIENT
            )
            assert "test-client-1" not in clients

        finally:
            await router.shutdown()

    @pytest.mark.asyncio
    async def test_message_validation_integration(self):
        """Test message validation integration with router."""
        router = HydraRouter(
            router_address="127.0.0.1", router_port=0, log_level="INFO"
        )

        try:
            await router.start()

            # Test valid message validation
            valid_message = {
                RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
                RouterConstants.ELEM: RouterConstants.HEARTBEAT,
                RouterConstants.DATA: {},
                RouterConstants.CLIENT_ID: "test-client",
                RouterConstants.TIMESTAMP: time.time(),
            }

            is_valid, error_msg = router.validator.validate_router_message(
                valid_message
            )
            assert is_valid is True
            assert error_msg == ""

            # Test invalid message validation
            invalid_message = {"invalid": "message"}
            is_valid, error_msg = router.validator.validate_router_message(
                invalid_message
            )
            assert is_valid is False
            assert "Missing required fields" in error_msg

        finally:
            await router.shutdown()

    def test_mq_client_format_conversion(self):
        """Test MQClient message format conversion."""
        client = MQClient(
            client_type=RouterConstants.HYDRA_CLIENT, client_id="test-client"
        )

        # Create test ZMQMessage
        zmq_message = ZMQMessage(
            message_type=MessageType.START_SIMULATION,
            timestamp=time.time(),
            client_id="test-client",
            data={"config": "test"},
        )

        # Test conversion to RouterConstants format
        router_message = client._convert_to_router_format(zmq_message)

        assert router_message[RouterConstants.SENDER] == RouterConstants.HYDRA_CLIENT
        assert router_message[RouterConstants.ELEM] == RouterConstants.START_SIMULATION
        assert router_message[RouterConstants.DATA] == {"config": "test"}
        assert router_message[RouterConstants.CLIENT_ID] == "test-client"

        # Test conversion back to ZMQMessage
        converted_back = client._convert_from_router_format(router_message)

        assert converted_back.message_type == MessageType.START_SIMULATION
        assert converted_back.client_id == "test-client"
        assert converted_back.data == {"config": "test"}

    def test_message_type_mapping_completeness(self):
        """Test that all message types have proper mapping."""
        client = MQClient(client_type=RouterConstants.HYDRA_CLIENT)

        # Test all message types in the mapping
        for zmq_type, router_elem in client.MESSAGE_TYPE_MAPPING.items():
            # Test forward mapping
            mapped_elem = client._map_message_type_to_elem(zmq_type)
            assert mapped_elem == router_elem

            # Test reverse mapping
            mapped_type = client._map_elem_to_message_type(router_elem)
            assert mapped_type.value == zmq_type

    @pytest.mark.asyncio
    async def test_background_task_management(self):
        """Test background task management."""
        router = HydraRouter(
            router_address="127.0.0.1", router_port=0, log_level="INFO"
        )

        try:
            await router.start()

            # Verify background tasks are running
            assert len(router.task_manager.tasks) > 0

            # Register a client and wait briefly
            await router.client_registry.register_client(
                "test-client", RouterConstants.HYDRA_CLIENT
            )

            # Wait a short time for background tasks to process
            await asyncio.sleep(0.1)

            # Verify client is still registered (not pruned immediately)
            clients = await router.client_registry.get_clients_by_type(
                RouterConstants.HYDRA_CLIENT
            )
            assert "test-client" in clients

        finally:
            await router.shutdown()

            # Verify tasks are cleaned up
            assert len(router.task_manager.tasks) == 0


class TestRouterErrorHandling:
    """Test error handling in router integration."""

    def test_invalid_message_format_handling(self):
        """Test handling of invalid message formats."""
        client = MQClient(client_type=RouterConstants.HYDRA_CLIENT)

        # Test invalid ZMQMessage type
        with pytest.raises(TypeError):
            client._convert_to_router_format("not a ZMQMessage")

        # Test invalid router message type
        with pytest.raises(TypeError):
            client._convert_from_router_format("not a dict")

    def test_unsupported_message_type_handling(self):
        """Test handling of unsupported message types."""
        client = MQClient(client_type=RouterConstants.HYDRA_CLIENT)

        # Test unsupported message type
        with pytest.raises(ValueError, match="Unsupported message type"):
            client._map_message_type_to_elem("unsupported_type")

        # Test unsupported router element
        with pytest.raises(ValueError, match="Unsupported router element"):
            client._map_elem_to_message_type("unsupported_element")

    @pytest.mark.asyncio
    async def test_client_registration_error_handling(self):
        """Test client registration error handling."""
        router = HydraRouter(
            router_address="127.0.0.1", router_port=0, log_level="INFO"
        )

        try:
            await router.start()

            # Test invalid client ID
            with pytest.raises(Exception):  # Should raise ClientRegistrationError
                await router.client_registry.register_client(
                    "", RouterConstants.HYDRA_CLIENT
                )

            # Test invalid client type
            with pytest.raises(Exception):  # Should raise ClientRegistrationError
                await router.client_registry.register_client(
                    "test-client", "InvalidType"
                )

        finally:
            await router.shutdown()


class TestRouterPerformance:
    """Basic performance tests for router system."""

    @pytest.mark.asyncio
    async def test_multiple_client_registration_performance(self):
        """Test performance with multiple client registrations."""
        router = HydraRouter(
            router_address="127.0.0.1", router_port=0, log_level="WARNING"
        )  # Reduce logging

        try:
            await router.start()

            # Register multiple clients quickly
            start_time = time.time()

            for i in range(100):
                await router.client_registry.register_client(
                    f"client-{i}", RouterConstants.HYDRA_CLIENT
                )

            registration_time = time.time() - start_time

            # Should complete within reasonable time (less than 1 second)
            assert (
                registration_time < 1.0
            ), f"Registration took too long: {registration_time:.2f}s"

            # Verify all clients are registered
            clients = await router.client_registry.get_clients_by_type(
                RouterConstants.HYDRA_CLIENT
            )
            assert len(clients) == 100

        finally:
            await router.shutdown()

    def test_message_conversion_performance(self):
        """Test message conversion performance."""
        client = MQClient(client_type=RouterConstants.HYDRA_CLIENT)

        # Create test message
        zmq_message = ZMQMessage(
            message_type=MessageType.HEARTBEAT,
            timestamp=time.time(),
            client_id="test-client",
            data={"test": "data"},
        )

        # Test conversion performance
        start_time = time.time()

        for _ in range(1000):
            router_message = client._convert_to_router_format(zmq_message)
            converted_back = client._convert_from_router_format(router_message)

        conversion_time = time.time() - start_time

        # Should complete within reasonable time (less than 1 second for 1000 conversions)
        assert (
            conversion_time < 1.0
        ), f"Conversion took too long: {conversion_time:.2f}s"

        # Average time per conversion should be very fast
        avg_time = conversion_time / 2000  # 2000 total conversions (to and from)
        assert avg_time < 0.001, f"Average conversion time too slow: {avg_time:.4f}s"
