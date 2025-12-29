ZeroMQ Communication Protocol
=============================

The AI Hydra agent provides a comprehensive ZeroMQ-based communication protocol for headless operation, enabling complete separation of AI logic from presentation layers.

Overview
--------

The ZeroMQ protocol enables:

* **Complete Headless Operation**: AI agent runs without GUI dependencies
* **Message-Based Control**: All operations controlled via structured messages
* **Real-time Monitoring**: Live status updates and performance metrics
* **Multi-Client Support**: Multiple clients can connect simultaneously
* **Language Agnostic**: Clients can be written in any language with ZeroMQ support

Architecture
------------

The communication architecture follows a client-server pattern:

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   TUI Client    │    │ Monitoring Tool │    │   API Client    │
    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                           ┌─────────▼───────┐
                           │  ZeroMQ Server  │
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐
                           │ Headless AI     │
                           │ Agent           │
                           └─────────────────┘

Message Structure
-----------------

All messages follow a standardized JSON structure:

.. code-block:: python

    {
        "message_type": "START_SIMULATION",
        "timestamp": 1640995200.123,
        "client_id": "client_001",
        "request_id": "req_12345",
        "data": {
            "grid_size": [8, 8],
            "move_budget": 50,
            "nn_enabled": true
        }
    }

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

Basic Client Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import zmq
    import json
    import time

    class ZMQClient:
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
        
        def start_simulation(self, config):
            return self.send_command("START_SIMULATION", config)
        
        def get_status(self):
            return self.send_command("GET_STATUS")
        
        def stop_simulation(self):
            return self.send_command("STOP_SIMULATION")

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

Server Configuration
--------------------

The ZeroMQ server supports various configuration options:

.. code-block:: python

    from ai_hydra.headless_server import HeadlessServer
    
    # Basic configuration
    server = HeadlessServer(
        command_port=5555,      # Port for command messages
        broadcast_port=5556,    # Port for status broadcasts
        log_level="INFO",       # Logging level
        log_file="ai_agent.log" # Log file path
    )
    
    # Advanced configuration
    server = HeadlessServer(
        command_port=5555,
        broadcast_port=5556,
        bind_address="0.0.0.0", # Bind to all interfaces
        max_clients=10,         # Maximum concurrent clients
        heartbeat_interval=5,   # Heartbeat interval in seconds
        message_timeout=30,     # Message timeout in seconds
        log_level="DEBUG",
        log_file="/var/log/snake_ai.log"
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

Local Development
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Start headless server
    python -m ai_hydra.headless_server

    # Connect with example client
    python -m ai_hydra.zmq_client_example --mode interactive

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run as daemon with logging
    python -m ai_hydra.headless_server \
        --bind "0.0.0.0" \
        --command-port 5555 \
        --broadcast-port 5556 \
        --log-file /var/log/snake_ai.log \
        --log-level INFO \
        --daemon

Remote Monitoring
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Connect from remote machine
    python -m ai_hydra.zmq_client_example \
        --server "tcp://production-server:5555" \
        --monitor "tcp://production-server:5556" \
        --mode demo

This protocol provides a robust foundation for integrating the AI Hydra agent into larger systems and enables flexible deployment scenarios.