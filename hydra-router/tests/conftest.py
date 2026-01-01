"""
Shared test configuration and fixtures for Hydra Router tests.
"""

import asyncio
import pytest
import logging
from typing import AsyncGenerator, Generator

from hydra_router import HydraRouter, MQClient, RouterConstants


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def router() -> AsyncGenerator[HydraRouter, None]:
    """Create a test router instance."""
    router = HydraRouter(
        router_address="127.0.0.1",
        router_port=0,  # Let system choose port
        log_level="DEBUG",
    )

    await router.start()

    yield router

    await router.shutdown()


@pytest.fixture
async def client() -> AsyncGenerator[MQClient, None]:
    """Create a test client instance."""
    client = MQClient(
        router_address="tcp://127.0.0.1:5556",
        client_type="TestClient",
        client_id="test-client-001",
    )

    yield client

    if client.is_connected:
        await client.disconnect()


@pytest.fixture
def sample_router_message():
    """Sample RouterConstants format message for testing."""
    return {
        RouterConstants.SENDER: RouterConstants.HYDRA_CLIENT,
        RouterConstants.ELEM: RouterConstants.HEARTBEAT,
        RouterConstants.DATA: {},
        RouterConstants.CLIENT_ID: "test-client",
        RouterConstants.TIMESTAMP: 1640995200.0,
        RouterConstants.REQUEST_ID: "test-request-001",
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
