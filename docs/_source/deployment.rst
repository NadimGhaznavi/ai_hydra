Deployment Guide
================

This guide covers deployment scenarios for the AI Hydra agent, from local development to production environments.

Overview
--------

The AI Hydra agent supports multiple deployment modes:

* **Direct Execution**: Traditional mode for development and testing
* **Headless Server**: Production mode with ZeroMQ communication
* **Containerized Deployment**: Docker-based deployment for scalability
* **Distributed Setup**: Multi-machine deployment for large-scale analysis

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Python**: 3.8 or higher
* **Memory**: Minimum 4GB RAM (8GB+ recommended for larger budgets)
* **CPU**: Multi-core processor recommended for parallel exploration
* **Storage**: 1GB free space for logs and model checkpoints

Dependencies
~~~~~~~~~~~~

.. code-block:: bash

    # Core dependencies
    pip install torch torchvision
    pip install hydra-core hydra-zen
    pip install pyzmq
    pip install numpy scipy
    pip install pytest hypothesis  # For testing

    # Optional dependencies
    pip install tensorboard        # For training visualization
    pip install psutil            # For resource monitoring
    pip install docker            # For containerized deployment

Local Development Setup
-----------------------

Basic Setup
~~~~~~~~~~~

1. **Clone and Install**:

.. code-block:: bash

    git clone <repository-url>
    cd ai_hydra
    pip install -e .

2. **Verify Installation**:

.. code-block:: python

    from ai_hydra import HydraMgr
    from ai_hydra.config import SimulationConfig
    
    config = SimulationConfig(grid_size=(6, 6), move_budget=20)
    hydra_mgr = HydraMgr(config)
    print("Installation successful!")

3. **Run Basic Test**:

.. code-block:: bash

    python -m pytest tests/test_basic_functionality.py -v

Development Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a development configuration file:

.. code-block:: python

    # dev_config.py
    from ai_hydra.config import SimulationConfig, NetworkConfig
    
    # Fast development configuration
    DEV_CONFIG = SimulationConfig(
        grid_size=(6, 6),           # Smaller grid for faster testing
        move_budget=20,             # Reduced budget for quick cycles
        nn_enabled=True,
        random_seed=42,
        log_level="DEBUG"
    )
    
    DEV_NETWORK_CONFIG = NetworkConfig(
        input_features=19,
        hidden_layers=(50, 50),     # Smaller network for faster training
        output_actions=3,
        learning_rate=0.01          # Faster learning for development
    )

Headless Server Deployment
---------------------------

Basic Headless Setup
~~~~~~~~~~~~~~~~~~~~

1. **Start Headless Server**:

.. code-block:: bash

    python -m ai_hydra.headless_server \
        --command-port 5555 \
        --broadcast-port 5556 \
        --log-level INFO

2. **Test Connection**:

.. code-block:: python

    from ai_hydra.zmq_client_example import ZMQClient
    
    client = ZMQClient("tcp://localhost:5555")
    status = client.send_command("GET_STATUS")
    print(f"Server status: {status}")

3. **Start Simulation**:

.. code-block:: python

    config = {
        "grid_size": [8, 8],
        "move_budget": 50,
        "nn_enabled": True,
        "random_seed": 123
    }
    
    response = client.send_command("START_SIMULATION", config)
    print(f"Simulation started: {response}")

Production Headless Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create Production Configuration**:

.. code-block:: python

    # production_config.py
    PRODUCTION_CONFIG = SimulationConfig(
        grid_size=(10, 10),         # Standard grid size
        move_budget=100,            # Full budget for thorough exploration
        nn_enabled=True,
        collision_penalty=-20,      # Higher penalty for better learning
        food_reward=15,             # Higher reward for motivation
        random_seed=None,           # Random seed for varied games
        log_level="INFO",
        enable_profiling=True       # Performance monitoring
    )
    
    PRODUCTION_NETWORK_CONFIG = NetworkConfig(
        input_features=19,
        hidden_layers=(200, 200),   # Full network architecture
        output_actions=3,
        learning_rate=0.001,        # Conservative learning rate
        batch_size=64               # Larger batches for stability
    )

2. **Production Server Script**:

.. code-block:: python

    # production_server.py
    import logging
    import signal
    import sys
    from ai_hydra.headless_server import HeadlessServer
    
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/snake_ai.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    def main():
        setup_logging()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        server = HeadlessServer(
            command_port=5555,
            broadcast_port=5556,
            bind_address="0.0.0.0",
            log_level="INFO",
            log_file="/var/log/snake_ai.log"
        )
        
        logging.info("Starting production AI server...")
        server.start()
    
    if __name__ == "__main__":
        main()

3. **Run Production Server**:

.. code-block:: bash

    python production_server.py

Containerized Deployment
-------------------------

Docker Setup
~~~~~~~~~~~~

1. **Create Dockerfile**:

.. code-block:: dockerfile

    FROM python:3.9-slim

    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

    # Set working directory
    WORKDIR /app

    # Copy requirements and install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY . .

    # Install the package
    RUN pip install -e .

    # Create log directory
    RUN mkdir -p /var/log

    # Expose ports
    EXPOSE 5555 5556

    # Set environment variables
    ENV PYTHONPATH=/app
    ENV LOG_LEVEL=INFO

    # Run the headless server
    CMD ["python", "-m", "ai_hydra.headless_server", \
         "--bind", "0.0.0.0", \
         "--command-port", "5555", \
         "--broadcast-port", "5556", \
         "--log-level", "INFO"]

2. **Build and Run Container**:

.. code-block:: bash

    # Build image
    docker build -t snake-ai-agent .

    # Run container
    docker run -d \
        --name snake-ai \
        -p 5555:5555 \
        -p 5556:5556 \
        -v /host/logs:/var/log \
        snake-ai-agent

3. **Docker Compose Setup**:

.. code-block:: yaml

    # docker-compose.yml
    version: '3.8'
    
    services:
      snake-ai:
        build: .
        ports:
          - "5555:5555"
          - "5556:5556"
        volumes:
          - ./logs:/var/log
          - ./models:/app/models
        environment:
          - LOG_LEVEL=INFO
          - PYTHONPATH=/app
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "python", "-c", "import zmq; c=zmq.Context(); s=c.socket(zmq.REQ); s.connect('tcp://localhost:5555'); s.close(); c.term()"]
          interval: 30s
          timeout: 10s
          retries: 3

Monitoring and Logging
----------------------

Log Configuration
~~~~~~~~~~~~~~~~~

1. **Structured Logging Setup**:

.. code-block:: python

    import logging
    import json
    from datetime import datetime
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if hasattr(record, 'simulation_id'):
                log_entry['simulation_id'] = record.simulation_id
            if hasattr(record, 'decision_cycle'):
                log_entry['decision_cycle'] = record.decision_cycle
                
            return json.dumps(log_entry)
    
    # Configure structured logging
    handler = logging.FileHandler('/var/log/snake_ai_structured.log')
    handler.setFormatter(StructuredFormatter())
    logging.getLogger('ai_hydra').addHandler(handler)

2. **Log Rotation Setup**:

.. code-block:: python

    import logging.handlers
    
    # Rotating file handler
    handler = logging.handlers.RotatingFileHandler(
        '/var/log/snake_ai.log',
        maxBytes=100*1024*1024,  # 100MB
        backupCount=5
    )
    
    # Time-based rotation
    time_handler = logging.handlers.TimedRotatingFileHandler(
        '/var/log/snake_ai_daily.log',
        when='midnight',
        interval=1,
        backupCount=30
    )

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

1. **Resource Monitoring Script**:

.. code-block:: python

    # monitor.py
    import psutil
    import time
    import json
    import zmq
    
    class ResourceMonitor:
        def __init__(self, zmq_address="tcp://localhost:5556"):
            self.context = zmq.Context()
            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect(zmq_address)
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        
        def monitor_resources(self):
            while True:
                # System resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # AI agent metrics (from ZMQ)
                try:
                    message = self.subscriber.recv_string(zmq.NOBLOCK)
                    ai_metrics = json.loads(message)
                except zmq.Again:
                    ai_metrics = {}
                
                metrics = {
                    'timestamp': time.time(),
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'disk_percent': disk.percent
                    },
                    'ai_agent': ai_metrics
                }
                
                print(json.dumps(metrics, indent=2))
                time.sleep(5)

2. **Health Check Endpoint**:

.. code-block:: python

    # health_check.py
    import zmq
    import json
    import sys
    
    def health_check(server_address="tcp://localhost:5555"):
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            socket.connect(server_address)
            
            message = {
                "message_type": "GET_STATUS",
                "timestamp": time.time(),
                "client_id": "health_check",
                "request_id": "health_001"
            }
            
            socket.send_string(json.dumps(message))
            response = socket.recv_string()
            data = json.loads(response)
            
            if data.get('status') == 'OK':
                print("Health check: PASS")
                return 0
            else:
                print(f"Health check: FAIL - {data}")
                return 1
                
        except Exception as e:
            print(f"Health check: ERROR - {e}")
            return 1
        finally:
            socket.close()
            context.term()
    
    if __name__ == "__main__":
        sys.exit(health_check())

Scaling and Performance
-----------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

For large-scale analysis, deploy multiple AI agents:

.. code-block:: yaml

    # docker-compose-scale.yml
    version: '3.8'
    
    services:
      snake-ai-1:
        build: .
        ports:
          - "5555:5555"
          - "5556:5556"
        environment:
          - AGENT_ID=agent_1
          - LOG_LEVEL=INFO
      
      snake-ai-2:
        build: .
        ports:
          - "5557:5555"
          - "5558:5556"
        environment:
          - AGENT_ID=agent_2
          - LOG_LEVEL=INFO
      
      load-balancer:
        image: nginx:alpine
        ports:
          - "80:80"
        volumes:
          - ./nginx.conf:/etc/nginx/nginx.conf
        depends_on:
          - snake-ai-1
          - snake-ai-2

Performance Tuning
~~~~~~~~~~~~~~~~~~

1. **Memory Optimization**:

.. code-block:: python

    # Optimized configuration for memory-constrained environments
    MEMORY_OPTIMIZED_CONFIG = SimulationConfig(
        grid_size=(8, 8),           # Moderate grid size
        move_budget=50,             # Reduced budget
        nn_enabled=True,
        max_tree_depth=5,           # Limit tree depth
        cleanup_frequency=10        # More frequent cleanup
    )
    
    MEMORY_OPTIMIZED_NETWORK = NetworkConfig(
        input_features=19,
        hidden_layers=(100, 100),   # Smaller network
        output_actions=3,
        batch_size=16               # Smaller batches
    )

2. **CPU Optimization**:

.. code-block:: python

    # CPU-optimized configuration
    CPU_OPTIMIZED_CONFIG = SimulationConfig(
        grid_size=(6, 6),           # Smaller grid for faster processing
        move_budget=30,             # Reduced budget
        nn_enabled=True,
        decision_timeout=5.0,       # Timeout for long decisions
        parallel_clones=2           # Reduce parallelism
    )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Port Already in Use**:

.. code-block:: bash

    # Find process using port
    lsof -i :5555
    
    # Kill process
    kill -9 <PID>
    
    # Or use different port
    python -m ai_hydra.headless_server --command-port 5557

**Memory Issues**:

.. code-block:: bash

    # Monitor memory usage
    top -p $(pgrep -f headless_server)
    
    # Reduce memory usage
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

**Connection Timeouts**:

.. code-block:: python

    # Increase timeout in client
    socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds

**Performance Issues**:

.. code-block:: bash

    # Profile the application
    python -m cProfile -o profile.stats -m ai_hydra.headless_server
    
    # Analyze profile
    python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

Backup and Recovery
-------------------

Model Checkpoints
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Automatic model saving
    CHECKPOINT_CONFIG = NetworkConfig(
        checkpoint_frequency=100,   # Save every 100 decision cycles
        checkpoint_dir="/app/models",
        max_checkpoints=10          # Keep last 10 checkpoints
    )

Log Backup
~~~~~~~~~~

.. code-block:: bash

    # Daily log backup script
    #!/bin/bash
    DATE=$(date +%Y%m%d)
    tar -czf /backup/logs_$DATE.tar.gz /var/log/snake_ai*.log
    find /backup -name "logs_*.tar.gz" -mtime +30 -delete

This deployment guide provides comprehensive coverage for running the AI Hydra agent in various environments, from development to production scale.