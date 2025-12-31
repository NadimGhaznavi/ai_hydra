ZeroMQ Router-Based Communication Protocol
==========================================

The AI Hydra system provides a comprehensive ZeroMQ router-based communication protocol that enables distributed, scalable operation with multiple clients connecting to a centralized AI server through an intelligent message routing system.

Overview
--------

The router-based protocol enables:

* **Distributed Architecture**: Router and AI server can run on different machines
* **Multiple Client Support**: Multiple TUI clients, monitoring tools, and API clients can connect simultaneously
* **Intelligent Message Routing**: Messages are routed based on sender type and client registration
* **Automatic Client Management**: Heartbeat-based client registration and lifecycle management
* **Fault Tolerance**: Automatic detection and cleanup of inactive clients
* **Scalable Deployment**: Support for local and remote deployment scenarios

Architecture
------------

The communication architecture follows a router-based pattern with centralized message routing:

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   TUI Client    │    │ Monitoring Tool │    │   API Client    │
    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                           ┌─────────▼───────┐
                           │  HydraRouter    │
                           │   (Port 5556)   │ ← Central Message Router
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐
                           │ Headless Server │
                           │ (AI Agent)      │ ← AI Processing Engine
                           └─────────────────┘

**Network Flow:**

1. **Client Connection**: Clients connect to HydraRouter at port 5556
2. **Registration**: Clients register via heartbeat messages every 5 seconds
3. **Message Routing**: Router intelligently routes messages between clients and server
4. **AI Processing**: Headless server processes AI decisions and sends responses back through router
5. **Broadcast Distribution**: Router distributes status updates and broadcasts to all registered clients

Router Components
-----------------

HydraRouter
~~~~~~~~~~~

The central message routing component that:

* **Connection Management**: Handles multiple client connections using ZeroMQ ROUTER socket
* **Client Registration**: Manages client lifecycle through heartbeat-based registration
* **Message Routing**: Routes messages based on sender type (HydraClient/HydraServer)
* **Inactive Client Detection**: Automatically removes clients after 15-second timeout
* **Background Tasks**: Manages async tasks for heartbeat processing and cleanup

MQClient
~~~~~~~~

Generic client class for router communication that:

* **Unified Interface**: Provides consistent API for both client and server roles
* **Connection Management**: Automatic connection and reconnection support
* **Message Protocol**: Structured JSON message handling with timeout management
* **Context Management**: Python context manager support for resource cleanup
* **Error Recovery**: Graceful error handling and automatic cleanup
* **Format Conversion**: Automatic conversion between ZMQMessage and RouterConstants formats

Message Format Adapter
~~~~~~~~~~~~~~~~~~~~~~~

The MQClient includes a built-in message format adapter that:

* **Transparent Conversion**: Automatically converts between ZMQMessage and RouterConstants formats
* **Bidirectional Support**: Handles both outgoing and incoming message conversion
* **Validation**: Validates message format compliance before processing
* **Error Handling**: Provides detailed error messages for format validation failures
* **Backward Compatibility**: Maintains compatibility with existing internal components

**Conversion Process:**

.. code-block:: python

    # Outgoing: ZMQMessage → RouterConstants
    zmq_message = {
        "message_type": "HEARTBEAT",
        "client_id": "client_001",
        "timestamp": 1640995200.123,
        "data": {"status": "active"}
    }
    
    # Converted to RouterConstants format
    router_message = {
        "sender": "HydraClient",
        "elem": "HEARTBEAT",
        "client_id": "client_001", 
        "timestamp": 1640995200.123,
        "data": {"status": "active"}
    }

**Error Handling:**

The format adapter provides comprehensive error handling:

* **Missing Fields**: Detailed error messages for missing required fields
* **Invalid Types**: Type validation with specific error descriptions
* **Unknown Message Types**: Clear errors for unsupported message types
* **Conversion Failures**: Graceful handling with fallback mechanisms

Client Registration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic client management through:

* **Heartbeat Messages**: Clients send heartbeat every 5 seconds to maintain registration
* **Timeout Detection**: Router removes clients inactive for more than 15 seconds
* **Background Processing**: Async task handles client lifecycle management
* **Resource Cleanup**: Automatic cleanup of inactive client resources

Message Structure
-----------------

The AI Hydra system uses two message formats depending on the communication context:

RouterConstants Format (Router Communication)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Messages sent to and from the router use the RouterConstants format with an ``elem`` field:

.. code-block:: python

    {
        "sender": "HydraClient",           # Sender type for routing
        "elem": "START_SIMULATION",        # Message type (RouterConstants format)
        "client_id": "client_001",         # Unique client identifier
        "timestamp": 1640995200.123,       # Message timestamp
        "request_id": "req_12345",         # Request correlation ID
        "data": {                          # Message payload
            "grid_size": [8, 8],
            "move_budget": 50,
            "nn_enabled": true
        }
    }

ZMQMessage Format (Internal Communication)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internal components use the ZMQMessage format with a ``message_type`` field:

.. code-block:: python

    {
        "message_type": "START_SIMULATION", # Message type (ZMQMessage format)
        "client_id": "client_001",         # Unique client identifier
        "timestamp": 1640995200.123,       # Message timestamp
        "request_id": "req_12345",         # Request correlation ID
        "data": {                          # Message payload
            "grid_size": [8, 8],
            "move_budget": 50,
            "nn_enabled": true
        }
    }

Message Format Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

The MQClient automatically handles format conversion between these two formats:

* **Outgoing Messages**: Converts ZMQMessage format to RouterConstants format when sending to router
* **Incoming Messages**: Converts RouterConstants format to ZMQMessage format for internal processing
* **Backward Compatibility**: Internal components continue using ZMQMessage format unchanged
* **Transparent Operation**: Format conversion is handled automatically by MQClient

**Message Routing Fields:**

* ``sender``: Identifies sender type (HydraClient, HydraServer) for intelligent routing
* ``elem``: Message type in RouterConstants format (used for router communication)
* ``message_type``: Message type in ZMQMessage format (used for internal communication)
* ``client_id``: Unique identifier for client connection tracking and response routing
* ``timestamp``: Message creation timestamp for ordering and timeout detection
* ``request_id``: Correlation ID for matching requests with responses
* ``data``: Structured payload containing message-specific information

**Format Conversion Process:**

The MQClient handles automatic conversion between message formats:

1. **Internal to Router**: ZMQMessage ``message_type`` → RouterConstants ``elem``
2. **Router to Internal**: RouterConstants ``elem`` → ZMQMessage ``message_type``
3. **Field Mapping**: All other fields (client_id, timestamp, data) are preserved
4. **Validation**: Both formats are validated before processing

Message Types
-------------

Command Messages
~~~~~~~~~~~~~~~~

**START_SIMULATION**
    Initialize and start a new simulation with specified configuration.

    Request Data:
        * ``grid_size``: [width, height] - Game grid dimensions
        * ``move_budget``: int - Budget for tree search exploration
        * ``nn_enabled``: bool - Enable neural network integration
        * ``random_seed``: int - Seed for deterministic behavior
        * ``collision_penalty``: int - Penalty for collisions (default: -10)
        * ``food_reward``: int - Reward for eating food (default: 10)

    Response:
        * ``status``: "SIMULATION_STARTED"
        * ``simulation_id``: Unique identifier for the simulation

**STOP_SIMULATION**
    Stop the currently running simulation.

    Request Data: None

    Response:
        * ``status``: "SIMULATION_STOPPED"
        * ``final_results``: Final game statistics

**PAUSE_SIMULATION**
    Pause the current simulation while maintaining state.

    Request Data: None

    Response:
        * ``status``: "SIMULATION_PAUSED"

**RESUME_SIMULATION**
    Resume a paused simulation.

    Request Data: None

    Response:
        * ``status``: "SIMULATION_RESUMED"

**GET_STATUS**
    Query current simulation status and performance metrics.

    Request Data: None

    Response:
        * ``status``: Current simulation state
        * ``game_state``: Current game board state
        * ``performance_metrics``: System performance data

**RESET_SIMULATION**
    Reset all simulation state and stop current execution.

    Request Data: None

    Response:
        * ``status``: "SIMULATION_RESET"

System Messages
~~~~~~~~~~~~~~~

**HEARTBEAT**
    Client registration and keep-alive message sent every 5 seconds.

    RouterConstants Format:
        * ``sender``: Client type (HydraClient, HydraServer)
        * ``elem``: "HEARTBEAT"
        * ``client_id``: Unique client identifier
        * ``timestamp``: Message timestamp
        * ``data``: Client capabilities and status

    Response:
        * ``status``: "HEARTBEAT_ACK"
        * ``server_time``: Server timestamp for synchronization

    **Note**: Heartbeat messages use RouterConstants format to ensure proper router processing.

**ERROR**
    Error notification with detailed information and recovery guidance.

    Data:
        * ``error_code``: Standardized error code
        * ``error_message``: Human-readable error description
        * ``error_context``: Additional context information
        * ``recovery_suggestions``: Recommended recovery actions

**OK**
    Acknowledgment message for successful operations.

    Data:
        * ``operation``: Operation that was acknowledged
        * ``result_summary``: Brief summary of operation result

Broadcast Messages
~~~~~~~~~~~~~~~~~~

**STATUS_UPDATE**
    Real-time status updates sent automatically during simulation.

    Data:
        * ``current_score``: Current game score
        * ``moves_count``: Total moves executed
        * ``is_game_over``: Game termination status
        * ``performance_metrics``: Real-time performance data

**DECISION_CYCLE_COMPLETE**
    Notification when a decision cycle completes.

    Data:
        * ``cycle_number``: Decision cycle identifier
        * ``move_selected``: Move chosen by the system
        * ``budget_used``: Budget consumed in this cycle
        * ``paths_evaluated``: Number of exploration paths
        * ``nn_accuracy``: Neural network prediction accuracy

**GAME_OVER**
    Final results when simulation completes.

    Data:
        * ``final_score``: Final game score
        * ``total_moves``: Total moves in the game
        * ``collision_avoidance_rate``: Success rate avoiding collisions
        * ``simulation_duration``: Total simulation time
        * ``performance_summary``: Complete performance statistics

**ERROR_OCCURRED**
    Error notifications with recovery information.

    Data:
        * ``error_type``: Category of error
        * ``error_message``: Detailed error description
        * ``recovery_action``: Suggested recovery steps
        * ``can_continue``: Whether simulation can continue

Data Structures
---------------

Game State Data
~~~~~~~~~~~~~~~

.. code-block:: python

    {
        "snake_head": [x, y],
        "snake_body": [[x1, y1], [x2, y2], ...],
        "direction": [dx, dy],
        "food_position": [x, y],
        "score": 15,
        "grid_size": [8, 8],
        "moves_count": 42,
        "is_game_over": false
    }

Performance Metrics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    {
        "decisions_per_second": 2.5,
        "budget_utilization": 0.85,
        "average_tree_depth": 4.2,
        "neural_network_accuracy": 0.73,
        "memory_usage_mb": 156.7,
        "cpu_usage_percent": 45.2,
        "total_decision_cycles": 28,
        "paths_evaluated_per_cycle": 12.4
    }

Client Implementation
---------------------

Router-Based Client Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ai_hydra.mq_client import MQClient
    import asyncio

    class RouterClient:
        def __init__(self, router_address="tcp://localhost:5556"):
            self.client = MQClient(
                router_address=router_address,
                client_id="example_client",
                sender_type="HydraClient"
            )
        
        async def connect(self):
            """Connect to router and start heartbeat."""
            await self.client.connect()
            await self.client.start_heartbeat()
        
        async def start_simulation(self, config):
            """Start simulation with configuration."""
            response = await self.client.send_command(
                "start_simulation", 
                config
            )
            return response
        
        async def get_status(self):
            """Get current simulation status."""
            response = await self.client.send_command("get_status")
            return response
        
        async def stop_simulation(self):
            """Stop current simulation."""
            response = await self.client.send_command("stop_simulation")
            return response
        
        async def disconnect(self):
            """Disconnect from router."""
            await self.client.disconnect()

    # Usage example
    async def main():
        client = RouterClient()
        
        try:
            await client.connect()
            
            # Start simulation
            config = {
                "grid_size": [8, 8],
                "move_budget": 50,
                "random_seed": 42
            }
            result = await client.start_simulation(config)
            print(f"Simulation started: {result}")
            
            # Monitor status
            while True:
                status = await client.get_status()
                if status.get("is_game_over"):
                    break
                await asyncio.sleep(1)
                
        finally:
            await client.disconnect()

Legacy Direct Client Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import zmq
    import json
    import time

    class DirectZMQClient:
        """Legacy direct connection client (deprecated)."""
        def __init__(self, server_address="tcp://localhost:5555"):
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(server_address)
        
        def send_command(self, message_type, data=None):
            message = {
                "message_type": message_type,
                "timestamp": time.time(),
                "client_id": "example_client",
                "request_id": f"req_{int(time.time())}",
                "data": data or {}
            }
            
            self.socket.send_string(json.dumps(message))
            response = self.socket.recv_string()
            return json.loads(response)

Monitoring Client Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import zmq
    import json
    import threading

    class MonitoringClient:
        def __init__(self, server_address="tcp://localhost:5556"):
            self.context = zmq.Context()
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect(server_address)
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all
            self.running = False
        
        def start_monitoring(self):
            self.running = True
            thread = threading.Thread(target=self._monitor_loop)
            thread.start()
            return thread
        
        def _monitor_loop(self):
            while self.running:
                try:
                    message = self.subscriber.recv_string(zmq.NOBLOCK)
                    data = json.loads(message)
                    self._handle_broadcast(data)
                except zmq.Again:
                    time.sleep(0.1)
        
        def _handle_broadcast(self, data):
            message_type = data.get("message_type")
            if message_type == "STATUS_UPDATE":
                self._handle_status_update(data)
            elif message_type == "DECISION_CYCLE_COMPLETE":
                self._handle_decision_complete(data)
            elif message_type == "GAME_OVER":
                self._handle_game_over(data)

Router and Server Configuration
-------------------------------

Router Configuration
~~~~~~~~~~~~~~~~~~~~~

The HydraRouter can be configured and deployed independently:

.. code-block:: bash

    # Start router with default settings
    ai-hydra-router
    
    # Start router with custom configuration
    ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO
    
    # Start router for remote deployment
    ai-hydra-router --address 0.0.0.0 --port 5556 --log-level DEBUG

.. code-block:: python

    from ai_hydra.router import HydraRouter
    import asyncio
    
    # Programmatic router configuration
    async def start_router():
        router = HydraRouter(
            bind_address="0.0.0.0",
            port=5556,
            heartbeat_interval=5,      # Heartbeat interval in seconds
            client_timeout=15,         # Client timeout in seconds
            log_level="INFO"
        )
        
        await router.start()
        print("Router started and listening for connections")
        
        try:
            await router.run_forever()
        except KeyboardInterrupt:
            await router.stop()

Server Configuration
~~~~~~~~~~~~~~~~~~~~

The headless server connects to the router instead of binding directly:

.. code-block:: bash

    # Connect server to local router
    ai-hydra-server --router tcp://localhost:5556
    
    # Connect server to remote router
    ai-hydra-server --router tcp://production-router:5556 --log-level INFO

.. code-block:: python

    from ai_hydra.headless_server import HeadlessServer
    
    # Server configuration for router connection
    server = HeadlessServer(
        router_address="tcp://localhost:5556",
        client_id="ai_server_001",
        log_level="INFO",
        log_file="ai_agent.log"
    )

TUI Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

The TUI client connects to the router for distributed operation:

.. code-block:: bash

    # Connect TUI to local router
    ai-hydra-tui --router tcp://localhost:5556
    
    # Connect TUI to remote router
    ai-hydra-tui --router tcp://production-router:5556

Legacy Direct Server Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ai_hydra.headless_server import HeadlessServer
    
    # Legacy direct binding (deprecated)
    server = HeadlessServer(
        command_port=5555,      # Port for command messages
        broadcast_port=5556,    # Port for status broadcasts
        log_level="INFO",       # Logging level
        log_file="ai_agent.log" # Log file path
    )

Error Handling
--------------

The protocol includes comprehensive error handling:

Connection Errors
~~~~~~~~~~~~~~~~~

* **Connection Timeout**: Client connection attempts timeout after 30 seconds
* **Server Unavailable**: Graceful handling when server is not running
* **Network Issues**: Automatic retry with exponential backoff

Message Errors
~~~~~~~~~~~~~~

* **Invalid JSON**: Malformed message handling with error responses
* **Missing Fields**: Validation of required message fields
* **Type Errors**: Data type validation and conversion

Simulation Errors
~~~~~~~~~~~~~~~~~

* **Configuration Errors**: Invalid simulation parameters
* **Runtime Errors**: Errors during simulation execution
* **Resource Errors**: Memory or CPU resource exhaustion

Security Considerations
-----------------------

* **No Authentication**: Current implementation does not include authentication
* **Local Network**: Designed for trusted local network environments
* **Message Validation**: All messages are validated for structure and content
* **Resource Limits**: Built-in limits prevent resource exhaustion attacks

Deployment Examples
-------------------

Local Development with Router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Terminal 1: Start router
    ai-hydra-router --address localhost --port 5556 --log-level DEBUG
    
    # Terminal 2: Start headless server
    ai-hydra-server --router tcp://localhost:5556 --log-level INFO
    
    # Terminal 3: Start TUI client
    ai-hydra-tui --router tcp://localhost:5556

Distributed Production Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Machine 1 (Router): Start central router
    ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO
    
    # Machine 2 (AI Server): Connect to remote router
    ai-hydra-server --router tcp://router-machine:5556 --log-level INFO
    
    # Machine 3 (Client): Connect TUI to remote router
    ai-hydra-tui --router tcp://router-machine:5556
    
    # Machine 4 (Monitoring): Connect monitoring client
    python monitoring_client.py --router tcp://router-machine:5556

Docker Deployment
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # docker-compose.yml
    version: '3.8'
    services:
      router:
        image: ai-hydra:latest
        command: ai-hydra-router --address 0.0.0.0 --port 5556
        ports:
          - "5556:5556"
        networks:
          - ai-hydra-net
      
      ai-server:
        image: ai-hydra:latest
        command: ai-hydra-server --router tcp://router:5556
        depends_on:
          - router
        networks:
          - ai-hydra-net
      
      tui-client:
        image: ai-hydra:latest
        command: ai-hydra-tui --router tcp://router:5556
        depends_on:
          - router
          - ai-server
        networks:
          - ai-hydra-net
        stdin_open: true
        tty: true
    
    networks:
      ai-hydra-net:
        driver: bridge

Legacy Local Development
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Legacy direct connection (deprecated)
    python -m ai_hydra.headless_server

    # Connect with legacy client
    python -m ai_hydra.zmq_client_example --mode interactive

Remote Monitoring with Router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Connect monitoring tools from any machine
    python monitoring_client.py --router tcp://production-router:5556 --client-id monitor_001

This protocol provides a robust foundation for integrating the AI Hydra agent into larger systems and enables flexible deployment scenarios.