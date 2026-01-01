Hydra Router Configuration Guide
================================

This guide provides comprehensive configuration instructions for the Hydra Router system, including router setup, client configuration, and deployment options.

Overview
--------

The Hydra Router system provides flexible configuration options for different deployment scenarios, from local development to distributed production environments. The system supports multiple clients connecting to a single server through the centralized router, with configuration managed through command-line arguments, configuration files, and environment variables.

Router Configuration
--------------------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The Hydra Router provides a CLI command with configurable options:

.. code-block:: bash

    ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO

**Available Options:**

* ``--address``: Network binding address (default: localhost)
* ``--port``: Network binding port (default: 5556)
* ``--log-level``: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
* ``--config``: Configuration file path (optional)
* ``--heartbeat-interval``: Client heartbeat interval in seconds (default: 5)
* ``--client-timeout``: Client timeout in seconds (default: 15)
* ``--max-connections``: Maximum concurrent connections (default: 100)

Configuration File Format
~~~~~~~~~~~~~~~~~~~~~~~~~

Router configuration can be specified using JSON or YAML files:

.. code-block:: yaml

    # router_config.yaml
    network:
      address: "0.0.0.0"
      port: 5556
      max_connections: 200
    
    heartbeat:
      interval: 5
      timeout: 15
    
    logging:
      level: "INFO"
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      file: "/var/log/hydra-router.log"
    
    routing:
      enable_client_to_client: false
      enable_server_to_server: false  # Reserved for future multiple server support
      message_queue_size: 1000
    
    monitoring:
      enable_metrics: true
      metrics_port: 9090
      health_check_port: 8080

**Configuration Sections:**

**Network Configuration:**
  - ``address``: Binding address for router socket
  - ``port``: Binding port for client connections
  - ``max_connections``: Maximum concurrent client connections

**Heartbeat Configuration:**
  - ``interval``: How often clients send heartbeat messages
  - ``timeout``: How long to wait before considering client inactive

**Logging Configuration:**
  - ``level``: Log verbosity level
  - ``format``: Log message format string
  - ``file``: Log file path (optional, defaults to stdout)

**Routing Configuration:**
  - ``enable_client_to_client``: Allow direct client-to-client communication
  - ``enable_server_to_server``: Reserved for future multiple server support
  - ``message_queue_size``: Maximum queued messages per client

**Monitoring Configuration:**
  - ``enable_metrics``: Enable metrics collection
  - ``metrics_port``: Port for metrics endpoint
  - ``health_check_port``: Port for health check endpoint

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Router configuration can also be set using environment variables:

.. code-block:: bash

    export HYDRA_ROUTER_ADDRESS="0.0.0.0"
    export HYDRA_ROUTER_PORT="5556"
    export HYDRA_ROUTER_LOG_LEVEL="INFO"
    export HYDRA_ROUTER_MAX_CONNECTIONS="200"
    export HYDRA_ROUTER_HEARTBEAT_INTERVAL="5"
    export HYDRA_ROUTER_CLIENT_TIMEOUT="15"

**Environment Variable Precedence:**
1. Command line arguments (highest priority)
2. Configuration file settings
3. Environment variables
4. Default values (lowest priority)

Client Configuration
--------------------

MQClient Configuration
~~~~~~~~~~~~~~~~~~~~~~

Clients configure connection to the router using MQClient parameters:

.. code-block:: python

    from ai_hydra.mq_client import MQClient
    
    # Basic configuration
    client = MQClient(
        router_address="tcp://localhost:5556",
        client_type="HydraClient",
        client_id="client_001",
        timeout=30.0,
        heartbeat_interval=5.0,
        max_retries=3,
        retry_delay=1.0
    )

**Configuration Parameters:**

* ``router_address``: Router connection string (e.g., "tcp://localhost:5556")
* ``client_type``: Client classification (HydraClient, HydraServer, etc.)
* ``client_id``: Unique client identifier
* ``timeout``: Message operation timeout in seconds
* ``heartbeat_interval``: Heartbeat message frequency
* ``max_retries``: Maximum connection retry attempts
* ``retry_delay``: Delay between retry attempts

Advanced Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Advanced configuration with custom settings
    client_config = {
        "router_address": "tcp://router.example.com:5556",
        "client_type": "HydraServer",
        "client_id": "ai_server_001",
        "timeout": 60.0,
        "heartbeat_interval": 3.0,
        "max_retries": 5,
        "retry_delay": 2.0,
        "enable_compression": True,
        "message_buffer_size": 10000,
        "enable_encryption": False,
        "log_level": "DEBUG"
    }
    
    client = MQClient(**client_config)

**Advanced Parameters:**

* ``enable_compression``: Enable message compression for large payloads
* ``message_buffer_size``: Internal message buffer size
* ``enable_encryption``: Enable message encryption (requires key configuration)
* ``log_level``: Client-specific logging level

Server Configuration
~~~~~~~~~~~~~~~~~~~~

AI Hydra server configuration for router connection:

.. code-block:: bash

    # Start server with router connection
    ai-hydra-server --router tcp://localhost:5556 --log-level INFO

.. code-block:: python

    # Server configuration in code
    from ai_hydra.headless_server import HeadlessServer
    
    server_config = {
        "router_address": "tcp://localhost:5556",
        "client_id": "hydra_server_main",
        "simulation_config": simulation_config,
        "enable_status_broadcasting": True,
        "status_update_interval": 1.0
    }
    
    server = HeadlessServer(**server_config)

TUI Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Terminal UI client configuration:

.. code-block:: bash

    # Start TUI with router connection
    ai-hydra-tui --router tcp://localhost:5556

.. code-block:: python

    # TUI configuration
    tui_config = {
        "router_address": "tcp://localhost:5556",
        "client_id": "tui_client_001",
        "auto_connect": True,
        "refresh_rate": 10,  # Updates per second
        "enable_keyboard_shortcuts": True
    }

Deployment Configurations
-------------------------

Local Development
~~~~~~~~~~~~~~~~~

Configuration for local development and testing:

.. code-block:: yaml

    # dev_config.yaml
    network:
      address: "localhost"
      port: 5556
      max_connections: 10
    
    heartbeat:
      interval: 2
      timeout: 6
    
    logging:
      level: "DEBUG"
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    routing:
      enable_client_to_client: true
      enable_server_to_server: true
      message_queue_size: 100

**Development Features:**
- Verbose logging for debugging
- Shorter timeouts for faster feedback
- Smaller connection limits for resource efficiency
- Enabled cross-communication for testing

Production Environment
~~~~~~~~~~~~~~~~~~~~~~

Configuration for production deployment:

.. code-block:: yaml

    # prod_config.yaml
    network:
      address: "0.0.0.0"
      port: 5556
      max_connections: 500
    
    heartbeat:
      interval: 5
      timeout: 15
    
    logging:
      level: "INFO"
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
      file: "/var/log/hydra-router.log"
    
    routing:
      enable_client_to_client: false
      enable_server_to_server: false
      message_queue_size: 5000
    
    monitoring:
      enable_metrics: true
      metrics_port: 9090
      health_check_port: 8080
    
    security:
      enable_access_control: true
      allowed_networks: ["10.0.0.0/8", "192.168.0.0/16"]
      rate_limit_per_client: 1000  # messages per minute

**Production Features:**
- Optimized for high throughput
- Comprehensive monitoring
- Security controls
- Resource limits and rate limiting

Docker Deployment
~~~~~~~~~~~~~~~~~

Docker configuration for containerized deployment:

.. code-block:: dockerfile

    # Dockerfile
    FROM python:3.11-slim
    
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    COPY . .
    
    EXPOSE 5556 9090 8080
    
    CMD ["ai-hydra-router", "--config", "/app/config/router_config.yaml"]

.. code-block:: yaml

    # docker-compose.yml
    version: '3.8'
    
    services:
      hydra-router:
        build: .
        ports:
          - "5556:5556"
          - "9090:9090"
          - "8080:8080"
        volumes:
          - ./config:/app/config
          - ./logs:/var/log
        environment:
          - HYDRA_ROUTER_LOG_LEVEL=INFO
        restart: unless-stopped
      
      hydra-server:
        build: .
        command: ["ai-hydra-server", "--router", "tcp://hydra-router:5556"]
        depends_on:
          - hydra-router
        restart: unless-stopped

Kubernetes Deployment
~~~~~~~~~~~~~~~~~~~~~~

Kubernetes configuration for cloud deployment:

.. code-block:: yaml

    # k8s-deployment.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: hydra-router
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: hydra-router
      template:
        metadata:
          labels:
            app: hydra-router
        spec:
          containers:
          - name: hydra-router
            image: ai-hydra/router:latest
            ports:
            - containerPort: 5556
            - containerPort: 9090
            - containerPort: 8080
            env:
            - name: HYDRA_ROUTER_ADDRESS
              value: "0.0.0.0"
            - name: HYDRA_ROUTER_PORT
              value: "5556"
            volumeMounts:
            - name: config-volume
              mountPath: /app/config
          volumes:
          - name: config-volume
            configMap:
              name: hydra-router-config
    
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: hydra-router-service
    spec:
      selector:
        app: hydra-router
      ports:
      - name: router
        port: 5556
        targetPort: 5556
      - name: metrics
        port: 9090
        targetPort: 9090
      - name: health
        port: 8080
        targetPort: 8080
      type: LoadBalancer

Security Configuration
----------------------

Access Control
~~~~~~~~~~~~~~

Configure network-based access control:

.. code-block:: yaml

    security:
      enable_access_control: true
      allowed_networks:
        - "10.0.0.0/8"
        - "192.168.0.0/16"
        - "172.16.0.0/12"
      denied_networks:
        - "0.0.0.0/0"  # Deny all by default
      
      rate_limiting:
        enable: true
        messages_per_minute: 1000
        burst_limit: 100
        
      authentication:
        enable: false  # Future feature
        method: "token"  # Future feature

Message Encryption
~~~~~~~~~~~~~~~~~~

Configure message encryption (future feature):

.. code-block:: yaml

    encryption:
      enable: false  # Not yet implemented
      method: "AES-256-GCM"
      key_rotation_interval: 3600  # seconds
      key_storage: "file"  # or "vault"

Monitoring Configuration
------------------------

Metrics Collection
~~~~~~~~~~~~~~~~~~

Configure metrics collection and export:

.. code-block:: yaml

    monitoring:
      enable_metrics: true
      metrics_port: 9090
      metrics_path: "/metrics"
      
      # Prometheus configuration
      prometheus:
        enable: true
        job_name: "hydra-router"
        scrape_interval: "15s"
      
      # Custom metrics
      custom_metrics:
        - name: "message_processing_time"
          type: "histogram"
          buckets: [0.001, 0.01, 0.1, 1.0, 10.0]
        
        - name: "active_connections"
          type: "gauge"
        
        - name: "message_throughput"
          type: "counter"

Health Checks
~~~~~~~~~~~~~

Configure health check endpoints:

.. code-block:: yaml

    health_checks:
      enable: true
      port: 8080
      path: "/health"
      
      checks:
        - name: "router_socket"
          type: "socket_binding"
          critical: true
        
        - name: "client_registry"
          type: "component_health"
          critical: true
        
        - name: "message_processing"
          type: "performance"
          critical: false
          threshold: 1000  # messages per second

Troubleshooting Configuration
-----------------------------

Debug Configuration
~~~~~~~~~~~~~~~~~~~

Enhanced debugging configuration:

.. code-block:: yaml

    debug:
      enable: true
      
      logging:
        level: "DEBUG"
        enable_trace: true
        log_message_content: true  # Security risk in production
        log_client_details: true
      
      performance:
        enable_profiling: true
        profile_output: "/tmp/router_profile.prof"
        
      network:
        log_socket_events: true
        log_connection_details: true

Common Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Port Conflicts:**
  Ensure the configured port is not in use by other services.

**Network Binding:**
  Use "0.0.0.0" for external access, "localhost" for local-only access.

**Resource Limits:**
  Adjust connection limits based on available system resources.

**Timeout Values:**
  Balance between responsiveness and network reliability.

**Log File Permissions:**
  Ensure the router process has write access to log directories.

Message Validation and Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hydra Router system includes comprehensive message validation with detailed error reporting and debugging support.

**Message Validation Framework**

The router uses the MessageValidator class to ensure all messages comply with RouterConstants format:

.. code-block:: python

    from hydra_router.validation import validate_message, validate_message_strict
    
    # Lenient validation (returns boolean)
    is_valid, error_msg = validate_message(message)
    if not is_valid:
        print(f"Validation failed: {error_msg}")
    
    # Strict validation (raises exception)
    try:
        validate_message_strict(message)
        print("Message is valid")
    except MessageValidationError as e:
        print(f"Validation error: {e}")
        print(f"Expected format: {e.expected_format}")

**Validation Features**

* **Required Field Validation**: Ensures presence of sender, elem, data, client_id, timestamp
* **Type Validation**: Validates correct data types for all fields
* **Sender Type Validation**: Checks sender against allowed types (HydraClient, HydraServer, HydraRouter)
* **Message Element Validation**: Validates elem against supported message types
* **Timestamp Validation**: Ensures reasonable timestamp values
* **Format Detection**: Detects common format mismatches (ZMQMessage vs RouterConstants)

**Error Debugging**

The validation system provides detailed debugging information:

.. code-block:: python

    from hydra_router.validation import get_validator
    
    validator = get_validator()
    error_details = validator.get_validation_error_details(invalid_message)
    print(f"Validation details: {error_details}")

**Configuration Validation**
-------------------------

The router provides built-in configuration validation:

.. code-block:: bash

    # Validate configuration file
    ai-hydra-router --config router_config.yaml --validate-only

**Validation Checks:**
- Required parameter presence
- Value range validation
- Network address format
- File path accessibility
- Resource limit reasonableness

This configuration guide provides comprehensive setup instructions for all deployment scenarios, from development to production environments.

Router Demo Examples
---------------------

The AI Hydra project includes comprehensive examples demonstrating router functionality:

**Complete Router Demo**
  The ``examples/test_router_demo.py`` script provides a complete demonstration of the router system including:
  
  - Router startup and configuration
  - Multiple client connections (demo-client-001, demo-client-002)
  - Server connection (demo-server-001)
  - Message routing between clients and server
  - Heartbeat monitoring and client lifecycle
  - Proper cleanup and shutdown procedures
  
  Run the demo with:
  
  .. code-block:: bash
  
     python examples/test_router_demo.py

**Basic Router Testing**
  For simpler router testing, use:
  
  .. code-block:: bash
  
     # Basic router functionality test
     python examples/simple_router_test.py
     
     # Router debugging and validation
     python examples/debug_test.py

These examples demonstrate real-world usage patterns and serve as templates for building your own router-based applications.