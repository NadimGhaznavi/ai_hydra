Troubleshooting Guide
=====================

This guide covers common issues and solutions when working with the AI Hydra agent.

Installation Issues
-------------------

PyTorch Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: PyTorch installation fails or CUDA compatibility issues

**Solutions**:

.. code-block:: bash

    # For CPU-only installation
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # For CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # For CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

**Verification**:

.. code-block:: python

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

ZeroMQ Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ZeroMQ compilation errors on some systems

**Solutions**:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt-get install libzmq3-dev
    pip install pyzmq
    
    # CentOS/RHEL
    sudo yum install zeromq-devel
    pip install pyzmq
    
    # macOS
    brew install zmq
    pip install pyzmq
    
    # Windows (use conda)
    conda install pyzmq

**Alternative Installation**:

.. code-block:: bash

    # Use pre-compiled wheels
    pip install --only-binary=pyzmq pyzmq

Configuration Issues
--------------------

Invalid Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Configuration validation errors

**Common Problems**:

1. **Grid size too small**:

.. code-block:: python

    # Error: Grid size must be at least (5, 5)
    config = SimulationConfig(grid_size=(3, 3))  # Too small
    
    # Solution:
    config = SimulationConfig(grid_size=(8, 8))  # Minimum recommended

2. **Snake length exceeds grid**:

.. code-block:: python

    # Error: Snake length too large for grid
    config = SimulationConfig(
        grid_size=(6, 6),
        initial_snake_length=10  # Too long for 6x6 grid
    )
    
    # Solution:
    config = SimulationConfig(
        grid_size=(10, 10),
        initial_snake_length=3   # Reasonable length
    )

3. **Invalid budget values**:

.. code-block:: python

    # Error: Budget must be positive
    config = SimulationConfig(move_budget=0)  # Invalid
    
    # Solution:
    config = SimulationConfig(move_budget=50)  # Positive value

**Debugging Configuration**:

.. code-block:: python

    from ai_hydra.config import SimulationConfig
    
    try:
        config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=50,
            nn_enabled=True
        )
        print("Configuration valid!")
    except ValueError as e:
        print(f"Configuration error: {e}")

Hydra Zen Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Hydra Zen configuration errors

**Common Solutions**:

.. code-block:: python

    from hydra_zen import make_config, instantiate
    from ai_hydra.config import SimulationConfig
    
    # Create Hydra Zen config
    Config = make_config(
        grid_size=(8, 8),
        move_budget=100,
        nn_enabled=True,
        hydra_defaults=["_self_"]
    )
    
    # Instantiate configuration
    try:
        config = instantiate(Config)
        sim_config = SimulationConfig(**config)
    except Exception as e:
        print(f"Hydra Zen error: {e}")

Runtime Issues
--------------

Memory Issues
~~~~~~~~~~~~~

**Issue**: Out of memory errors during simulation

**Symptoms**:
- Process killed by OS
- PyTorch CUDA out of memory errors
- Slow performance and swapping

**Solutions**:

1. **Reduce memory usage**:

.. code-block:: python

    # Use smaller configuration
    config = SimulationConfig(
        grid_size=(6, 6),        # Smaller grid
        move_budget=30,          # Reduced budget
        nn_enabled=True
    )
    
    network_config = NetworkConfig(
        hidden_layers=(50, 50),  # Smaller network
        batch_size=16            # Smaller batches
    )

2. **Enable memory monitoring**:

.. code-block:: python

    import psutil
    import torch
    
    def monitor_memory():
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Process memory: {memory_mb:.1f} MB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"GPU memory: {gpu_memory:.1f} MB")

3. **Force garbage collection**:

.. code-block:: python

    import gc
    import torch
    
    # After each decision cycle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

Performance Issues
~~~~~~~~~~~~~~~~~~

**Issue**: Simulation runs too slowly

**Diagnosis**:

.. code-block:: python

    import time
    import cProfile
    
    # Profile the simulation
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run simulation
    start_time = time.time()
    result = hydra_mgr.run_simulation()
    end_time = time.time()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    
    print(f"Simulation time: {end_time - start_time:.2f} seconds")

**Solutions**:

1. **Optimize configuration**:

.. code-block:: python

    # Fast configuration for development
    FAST_CONFIG = SimulationConfig(
        grid_size=(6, 6),        # Smaller grid
        move_budget=20,          # Reduced budget
        nn_enabled=False,        # Disable NN for pure tree search
        max_tree_depth=3         # Limit tree depth
    )

2. **Use performance mode**:

.. code-block:: python

    # Performance-optimized settings
    PERF_CONFIG = SimulationConfig(
        grid_size=(8, 8),
        move_budget=50,
        nn_enabled=True,
        decision_timeout=5.0,    # Timeout long decisions
        enable_profiling=False,  # Disable profiling overhead
        log_level="WARNING"      # Reduce logging overhead
    )

Test Failures
~~~~~~~~~~~~~

**Issue**: Tests hanging or timing out

**Common Causes**:
- Infinite loops in tree search
- Deadlocks in concurrent operations
- Tests without timeout protection

**Solutions**:

1. **Add timeout decorators**:

.. code-block:: python

    import pytest
    
    @pytest.mark.timeout(60)  # 60 second timeout
    def test_simulation_cycle():
        config = SimulationConfig(move_budget=10)  # Small budget for testing
        hydra_mgr = HydraMgr(config)
        result = hydra_mgr.execute_decision_cycle()
        assert result is not None

2. **Use test-specific configurations**:

.. code-block:: python

    # Test configuration with reduced complexity
    TEST_CONFIG = SimulationConfig(
        grid_size=(5, 5),        # Minimal grid
        move_budget=5,           # Minimal budget
        nn_enabled=False,        # Disable NN for faster tests
        log_level="ERROR"        # Minimal logging
    )

3. **Debug hanging tests**:

.. code-block:: bash

    # Run with verbose output
    pytest -v -s tests/test_problematic.py
    
    # Run single test with debugging
    pytest -v -s tests/test_problematic.py::test_specific_function --pdb

ZeroMQ Communication Issues
---------------------------

Connection Problems
~~~~~~~~~~~~~~~~~~~

**Issue**: Cannot connect to ZeroMQ server

**Diagnosis**:

.. code-block:: python

    import zmq
    import time
    
    def test_connection(address="tcp://localhost:5555"):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        
        try:
            socket.connect(address)
            
            # Send test message
            test_message = '{"message_type": "GET_STATUS", "timestamp": ' + str(time.time()) + '}'
            socket.send_string(test_message)
            
            # Wait for response
            response = socket.recv_string()
            print(f"Connection successful: {response}")
            return True
            
        except zmq.Again:
            print("Connection timeout - server not responding")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
        finally:
            socket.close()
            context.term()

**Solutions**:

1. **Check server status**:

.. code-block:: bash

    # Check if server is running
    ps aux | grep headless_server
    
    # Check port availability
    netstat -an | grep 5555
    lsof -i :5555

2. **Verify firewall settings**:

.. code-block:: bash

    # Ubuntu/Debian
    sudo ufw status
    sudo ufw allow 5555
    
    # CentOS/RHEL
    sudo firewall-cmd --list-ports
    sudo firewall-cmd --add-port=5555/tcp --permanent

3. **Test with telnet**:

.. code-block:: bash

    # Test basic connectivity
    telnet localhost 5555

Message Format Issues
~~~~~~~~~~~~~~~~~~~~~

**Issue**: Invalid message format errors

**Common Problems**:

1. **Invalid JSON**:

.. code-block:: python

    # Wrong: Invalid JSON
    message = "{'message_type': 'GET_STATUS'}"  # Single quotes
    
    # Correct: Valid JSON
    message = '{"message_type": "GET_STATUS"}'  # Double quotes

2. **Missing required fields**:

.. code-block:: python

    # Wrong: Missing timestamp
    message = {
        "message_type": "START_SIMULATION"
    }
    
    # Correct: All required fields
    message = {
        "message_type": "START_SIMULATION",
        "timestamp": time.time(),
        "client_id": "test_client",
        "request_id": "req_001",
        "data": {}
    }

3. **Invalid data types**:

.. code-block:: python

    # Wrong: String instead of list
    data = {"grid_size": "8,8"}
    
    # Correct: List of integers
    data = {"grid_size": [8, 8]}

**Message Validation**:

.. code-block:: python

    import json
    from ai_hydra.zmq_protocol import validate_message
    
    def validate_and_send(socket, message_dict):
        try:
            # Validate message structure
            validate_message(message_dict)
            
            # Convert to JSON
            json_message = json.dumps(message_dict)
            
            # Send message
            socket.send_string(json_message)
            
        except ValueError as e:
            print(f"Message validation error: {e}")
        except json.JSONEncodeError as e:
            print(f"JSON encoding error: {e}")

Neural Network Issues
---------------------

Training Problems
~~~~~~~~~~~~~~~~~

**Issue**: Neural network not learning or poor performance

**Diagnosis**:

.. code-block:: python

    from ai_hydra.oracle_trainer import OracleTrainer
    
    # Check training statistics
    trainer = OracleTrainer(model)
    stats = trainer.get_training_stats()
    
    print(f"Training samples: {stats['total_samples']}")
    print(f"Accuracy: {stats['accuracy']:.3f}")
    print(f"Loss: {stats['average_loss']:.3f}")

**Solutions**:

1. **Adjust learning rate**:

.. code-block:: python

    # Too high learning rate (unstable training)
    config = NetworkConfig(learning_rate=0.1)  # Too high
    
    # Better learning rate
    config = NetworkConfig(learning_rate=0.001)  # More stable

2. **Check feature extraction**:

.. code-block:: python

    from ai_hydra.feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(game_board)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"NaN values: {torch.isnan(features).sum()}")

3. **Verify training data quality**:

.. code-block:: python

    # Check for data imbalance
    move_counts = {"left": 0, "straight": 0, "right": 0}
    for sample in training_samples:
        move = sample.target_move
        move_counts[move] += 1
    
    print(f"Move distribution: {move_counts}")

Model Loading Issues
~~~~~~~~~~~~~~~~~~~~

**Issue**: Cannot load saved neural network models

**Solutions**:

.. code-block:: python

    import torch
    import os
    
    def safe_load_model(model_path, model_class):
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return None
            
            # Load model state
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Create model instance
            model = model_class()
            model.load_state_dict(state_dict)
            
            print(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

Logging Issues
--------------

Log File Problems
~~~~~~~~~~~~~~~~~

**Issue**: Log files not created or permission errors

**Solutions**:

.. code-block:: bash

    # Create log directory with proper permissions
    sudo mkdir -p /var/log/snake_ai
    sudo chown $USER:$USER /var/log/snake_ai
    sudo chmod 755 /var/log/snake_ai

**Alternative log location**:

.. code-block:: python

    import os
    from pathlib import Path
    
    # Use user home directory for logs
    log_dir = Path.home() / "snake_ai_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "simulation.log"

Log Level Issues
~~~~~~~~~~~~~~~~

**Issue**: Too much or too little logging output

**Solutions**:

.. code-block:: python

    import logging
    
    # Adjust log levels for different components
    logging.getLogger('ai_hydra.hydra_mgr').setLevel(logging.INFO)
    logging.getLogger('ai_hydra.neural_network').setLevel(logging.WARNING)
    logging.getLogger('ai_hydra.zmq_server').setLevel(logging.DEBUG)

**Custom log filtering**:

.. code-block:: python

    class CustomFilter(logging.Filter):
        def filter(self, record):
            # Filter out noisy messages
            if "budget consumed" in record.getMessage():
                return False
            return True
    
    # Apply filter
    logger = logging.getLogger('ai_hydra')
    logger.addFilter(CustomFilter())

Getting Help
------------

Debug Information Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When reporting issues, collect this information:

.. code-block:: python

    import sys
    import torch
    import zmq
    import platform
    from ai_hydra import __version__
    
    def collect_debug_info():
        info = {
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture()
            },
            "packages": {
                "ai_hydra": __version__,
                "torch": torch.__version__,
                "zmq": zmq.zmq_version(),
                "pyzmq": zmq.pyzmq_version()
            },
            "hardware": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return info
    
    # Print debug information
    import json
    print(json.dumps(collect_debug_info(), indent=2))

Debug and Diagnostic Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project includes specialized debug scripts for testing specific functionality:

**Token Tracker File Patterns Debug Script**

For issues related to token tracking metadata collection, use the file patterns debug script:

.. code-block:: bash

    # Test file patterns preservation in token tracking
    python debug_file_patterns.py

This script validates:

* MetadataCollector hook context processing
* Full metadata collection including file patterns
* End-to-end transaction recording with file patterns

**Expected successful output:**

.. code-block:: text

    Testing file patterns preservation...
    
    1. Testing MetadataCollector.get_hook_context()...
    ✓ file_patterns found: ['*.py', '*.md', '*.txt']
    ✓ file_patterns match input
    
    2. Testing MetadataCollector.collect_execution_metadata()...
    ✓ file_patterns found in metadata: ['*.py', '*.md', '*.txt']
    ✓ file_patterns match input in full metadata
    
    3. Testing TokenTracker integration...
    ✓ file_patterns preserved in transaction
    
    ✓ All tests passed!

**Troubleshooting with debug script:**

.. code-block:: bash

    # Run and capture output for analysis
    python debug_file_patterns.py > token_debug.log 2>&1
    
    # Check exit code
    if python debug_file_patterns.py; then
        echo "Token tracking file patterns working correctly"
    else
        echo "Token tracking has issues - check token_debug.log"
        cat token_debug.log
    fi

**Common debug script failure patterns:**

* **"file_patterns not found in hook context"**: MetadataCollector.get_hook_context() not preserving file_patterns
* **"file_patterns not found in full metadata"**: collect_execution_metadata() not including hook context properly
* **"file_patterns not preserved in transaction"**: TokenTracker not storing metadata correctly
* **"Transaction recording failed"**: CSV writing or validation issues

Log Analysis
~~~~~~~~~~~~

.. code-block:: bash

    # Search for errors in logs
    grep -i error /var/log/snake_ai.log
    
    # Find timeout issues
    grep -i timeout /var/log/snake_ai.log
    
    # Check memory usage patterns
    grep -i "memory\|oom" /var/log/snake_ai.log

Common Error Patterns
~~~~~~~~~~~~~~~~~~~~~

**Pattern**: "Budget exhausted but no paths found"
**Cause**: All exploration clones terminated immediately
**Solution**: Check collision detection logic and initial game state

**Pattern**: "ZMQ socket operation on closed socket"
**Cause**: Client disconnected while server was sending
**Solution**: Add proper connection state checking

**Pattern**: "CUDA out of memory"
**Cause**: Neural network or batch size too large
**Solution**: Reduce network size or use CPU-only mode

**Pattern**: "Assertion failed in tree search"
**Cause**: Invalid game state or logic error
**Solution**: Enable debug logging and check game state validation

This troubleshooting guide should help resolve most common issues. For additional support, check the project documentation and issue tracker.