# AI Hydra Router Implementation

## Overview

This document describes the implementation of the AI Hydra router system, which provides message routing between AI Hydra clients (TUI) and servers. The implementation is based on the ai_snake_lab SimRouter pattern and provides a robust, scalable messaging infrastructure.

## Architecture

### Components

1. **HydraRouter** (`ai_hydra/router.py`)
   - Central message router using ZeroMQ ROUTER socket
   - Handles client registration and heartbeat management
   - Routes messages between clients and servers
   - Manages client lifecycle and cleanup

2. **MQClient** (`ai_hydra/mq_client.py`)
   - Generic ZeroMQ client for connecting to the router
   - Supports both blocking and non-blocking message operations
   - Automatic heartbeat management
   - Command/response pattern support

3. **RouterConstants** (`ai_hydra/router_constants.py`)
   - Centralized constants for message types and routing
   - Network configuration constants
   - Human-readable labels for logging

### Message Flow

```
[TUI Client] <---> [HydraRouter] <---> [Headless Server]
     ^                   ^                      ^
     |                   |                      |
  MQClient          ROUTER Socket          MQClient
```

## Implementation Details

### Router Features

- **Client Registration**: Automatic client registration via heartbeat messages
- **Message Routing**: Intelligent routing based on sender type and message content
- **Heartbeat Management**: Automatic detection and removal of inactive clients
- **Error Handling**: Graceful error handling with informative error messages
- **Scalability**: Support for multiple clients connecting to single server

### MQClient Features

- **Connection Management**: Automatic connection and reconnection handling
- **Heartbeat**: Automatic heartbeat sending to maintain connection
- **Message Types**: Support for commands, responses, and broadcasts
- **Timeout Handling**: Configurable timeouts for operations
- **Context Management**: Support for Python context manager pattern

### Updated Components

1. **Headless Server** (`ai_hydra/headless_server.py`)
   - Updated to use MQClient instead of direct ZMQ
   - Connects to router instead of binding directly
   - Handles router messages and forwards to internal ZMQ server

2. **TUI Client** (`ai_hydra/tui/client.py`)
   - Updated to use MQClient for router communication
   - Improved error handling and connection management
   - Maintains all existing UI functionality

## Configuration

### Router Configuration

- **Default Port**: 5556 (RouterConstants.DEFAULT_ROUTER_PORT)
- **Heartbeat Interval**: 5 seconds (RouterConstants.HEARTBEAT_INTERVAL)
- **Client Timeout**: 3x heartbeat interval (15 seconds)

### Network Topology

```
Router:     tcp://localhost:5556  (ai-hydra-router)
Server:     Connects to router    (ai-hydra-server --router tcp://localhost:5556)
TUI Client: Connects to router    (ai-hydra-tui --router tcp://localhost:5556)
```

## CLI Commands

### New Commands

- `ai-hydra-router`: Start the AI Hydra router
  ```bash
  ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO
  ```

### Updated Commands

- `ai-hydra-server`: Now connects to router instead of binding
  ```bash
  ai-hydra-server --router tcp://localhost:5556 --log-level INFO
  ```

- `ai-hydra-tui`: Now connects to router instead of server directly
  ```bash
  ai-hydra-tui --router tcp://localhost:5556
  ```

## Usage Examples

### Starting the Complete System

1. **Start Router**:
   ```bash
   ai-hydra-router --log-level INFO
   ```

2. **Start Server**:
   ```bash
   ai-hydra-server --router tcp://localhost:5556
   ```

3. **Start TUI Client**:
   ```bash
   ai-hydra-tui --router tcp://localhost:5556
   ```

### Remote Deployment

1. **Router on dedicated machine**:
   ```bash
   ai-hydra-router --address 0.0.0.0 --port 5556
   ```

2. **Server connecting to remote router**:
   ```bash
   ai-hydra-server --router tcp://192.168.1.100:5556
   ```

3. **Multiple clients**:
   ```bash
   ai-hydra-tui --router tcp://192.168.1.100:5556
   ```

## Testing

### Test Coverage

The implementation includes comprehensive tests:

1. **Unit Tests**:
   - `tests/unit/test_mq_client.py`: MQClient functionality
   - `tests/unit/test_router.py`: HydraRouter functionality  
   - `tests/unit/test_router_constants.py`: Constants validation

2. **Property-Based Tests**:
   - `tests/property/test_router_properties.py`: Universal properties validation

3. **Integration Tests**:
   - `tests/integration/test_router_integration.py`: Component interaction

4. **End-to-End Tests**:
   - `tests/e2e/test_router_system.py`: Complete workflow validation

### Running Tests

```bash
# Run all router tests
python -m pytest tests/ -k "router" -v

# Run specific test categories
python -m pytest tests/unit/test_router_constants.py -v
python -m pytest tests/property/test_router_properties.py -v
python -m pytest tests/integration/test_router_integration.py -v
```

## Message Protocol

### Message Structure

All messages follow this structure:
```json
{
  "sender": "HydraClient|HydraServer",
  "client_id": "unique-client-id",
  "message_type": "command_type",
  "timestamp": 1234567890.123,
  "request_id": "uuid-string",
  "data": {...}
}
```

### Message Types

#### Control Commands
- `start_simulation`: Start simulation with config
- `stop_simulation`: Stop running simulation
- `pause_simulation`: Pause running simulation
- `resume_simulation`: Resume paused simulation
- `reset_simulation`: Reset simulation state

#### Status Messages
- `get_status`: Request current status
- `status_update`: Status information broadcast
- `game_state_update`: Game state changes
- `performance_update`: Performance metrics

#### System Messages
- `heartbeat`: Keep-alive message
- `error`: Error notification
- `ok`: Success acknowledgment

## Error Handling

### Router Error Handling

- **Connection Failures**: Graceful handling of client disconnections
- **Malformed Messages**: Validation and error reporting
- **No Server Available**: Informative error messages to clients
- **Resource Cleanup**: Proper cleanup on shutdown

### Client Error Handling

- **Connection Loss**: Automatic reconnection attempts
- **Timeout Handling**: Configurable timeouts with fallback behavior
- **Message Validation**: Input validation before sending
- **Graceful Degradation**: Continued operation during temporary issues

## Performance Considerations

### Scalability

- **Multiple Clients**: Router supports multiple concurrent clients
- **Message Throughput**: Efficient message routing with minimal latency
- **Memory Management**: Automatic cleanup of inactive clients
- **Resource Usage**: Minimal resource overhead per client

### Optimization

- **Heartbeat Efficiency**: Configurable heartbeat intervals
- **Message Batching**: Support for message batching where appropriate
- **Connection Pooling**: Efficient connection management
- **Async Operations**: Full async/await support for non-blocking operations

## Security Considerations

### Network Security

- **Bind Address Control**: Configurable bind addresses for security
- **Client Authentication**: Framework for future authentication features
- **Message Validation**: Input validation to prevent malformed messages
- **Resource Limits**: Protection against resource exhaustion

### Deployment Security

- **Firewall Configuration**: Proper port management
- **Network Isolation**: Support for isolated network deployments
- **Logging**: Comprehensive logging for security monitoring
- **Error Information**: Controlled error information disclosure

## Future Enhancements

### Planned Features

1. **Authentication**: Client authentication and authorization
2. **Encryption**: Message encryption for secure communication
3. **Load Balancing**: Multiple server support with load balancing
4. **Monitoring**: Enhanced monitoring and metrics collection
5. **Configuration**: Dynamic configuration updates
6. **Clustering**: Router clustering for high availability

### Extension Points

- **Message Handlers**: Pluggable message handling
- **Client Types**: Support for additional client types
- **Protocols**: Support for additional transport protocols
- **Persistence**: Message persistence and replay capabilities

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if router is running and accessible
2. **Heartbeat Timeouts**: Verify network connectivity and timing
3. **Message Routing**: Check client registration and message format
4. **Port Conflicts**: Ensure ports are available and not blocked

### Debugging

- **Log Levels**: Use DEBUG log level for detailed information
- **Message Tracing**: Enable message tracing for debugging
- **Client Status**: Monitor client registration and heartbeats
- **Network Tools**: Use network tools to verify connectivity

### Monitoring

- **Client Counts**: Monitor connected client and server counts
- **Message Rates**: Track message throughput and latency
- **Error Rates**: Monitor error rates and types
- **Resource Usage**: Monitor memory and CPU usage

## Conclusion

The AI Hydra router implementation provides a robust, scalable messaging infrastructure that enables distributed AI Hydra deployments. The system maintains the simplicity of the original direct connection while adding the flexibility and scalability of a router-based architecture.

The implementation follows established patterns from ai_snake_lab while adapting to AI Hydra's specific requirements. Comprehensive testing ensures reliability, and the modular design allows for future enhancements and extensions.