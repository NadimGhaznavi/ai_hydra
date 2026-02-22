# Hydra Router

A standalone ZeroMQ-based message routing system for distributed applications.

## Overview

Hydra Router provides reliable communication between multiple clients and servers through a centralized router pattern. It features automatic client discovery, heartbeat monitoring, message format standardization, and comprehensive error handling.

## Features

- **Centralized Message Routing**: Routes messages between multiple clients and servers
- **Generic MQClient Library**: Reusable client library for any application
- **Message Format Conversion**: Automatic conversion between internal and router formats
- **Heartbeat Monitoring**: Automatic client health tracking and cleanup
- **Comprehensive Validation**: Detailed message validation and error reporting
- **Flexible Configuration**: Configurable for different deployment scenarios

## Installation

```bash
pip install hydra-router
```

## Quick Start

### Start the Router

```bash
# Start router on default port (5556)
ai-hydra-router

# Start with custom configuration
ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO
```

### Basic Example

Run the basic client-server example:

```bash
# Terminal 1: Start the router
ai-hydra-router --log-level INFO

# Terminal 2: Run the example
python examples/basic_client_server.py
```

### Client Usage

```python
import asyncio
from hydra_router import MQClient, MessageType, ZMQMessage

async def main():
    # Create client
    client = MQClient(
        router_address="tcp://localhost:5556",
        client_type="MyApp",
        client_id="client-001"
    )
    
    # Connect to router
    await client.connect()
    
    # Send a message
    message = ZMQMessage.create_command(
        message_type=MessageType.GET_STATUS,
        client_id="client-001",
        request_id="req-001",
        data={"request": "status"}
    )
    await client.send_message(message)
    
    # Receive messages
    response = await client.receive_message_blocking(timeout=5.0)
    print(f"Received: {response}")
    
    # Cleanup
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

The Hydra Router system consists of three main components:

1. **HydraRouter**: Central message routing component
2. **MQClient**: Generic client library for applications
3. **RouterConstants**: Message format definitions and constants

### Message Flow

```
Client App 1 ──┐
               ├──► Hydra Router ──► Server App
Client App 2 ──┘       │
                        └──► Broadcast to all clients
```

## Configuration

### Router Configuration

```bash
ai-hydra-router --help
```

Options:
- `--address, -a`: IP address to bind (default: 0.0.0.0)
- `--port, -p`: Port to bind (default: 5556)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Client Configuration

```python
client = MQClient(
    router_address="tcp://localhost:5556",  # Router address
    client_type="HydraClient",              # Client type identifier
    heartbeat_interval=5.0,                 # Heartbeat frequency
    client_id="my-client-001"               # Unique client ID
)
```

## Message Format

The router uses a standardized message format:

```python
{
    "sender": "HydraClient",        # Client type
    "elem": "heartbeat",            # Message type
    "data": {},                     # Message payload
    "client_id": "client-001",      # Client identifier
    "timestamp": 1640995200.0,      # Message timestamp
    "request_id": "req-001"         # Optional request ID
}
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/ai-hydra/hydra-router.git
cd hydra-router
pip install -e ".[dev]"
```

### Examples

The `examples/` directory contains practical demonstrations:

- **`basic_client_server.py`**: Complete client-server communication example
- **`simple_heartbeat_test.py`**: Heartbeat mechanism demonstration

Run examples:

```bash
# Start router first
ai-hydra-router --log-level INFO

# Run basic client-server example
python examples/basic_client_server.py

# Run heartbeat test
python examples/simple_heartbeat_test.py
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hydra_router --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/property/      # Property-based tests
pytest tests/integration/   # Integration tests
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Support

- Documentation: https://hydra-router.readthedocs.io
- Issues: https://github.com/ai-hydra/hydra-router/issues
- Discussions: https://github.com/ai-hydra/hydra-router/discussions