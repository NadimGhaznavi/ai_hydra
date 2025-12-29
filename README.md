# AI Hydra

A sophisticated **parallel neural network exploration system** for Snake Game AI that achieves mastery over basic collision avoidance through exhaustive concurrent NN exploration.

## Overview

The AI Hydra addresses the fundamental problem where legacy AI regularly scores below 10 by implementing a "deep thinking" architecture that explores multiple NN learning trajectories simultaneously before making each move.

**Core Architecture**: Every decision spawns multiple concurrent NN instances that explore different learning paths through a budget-constrained tree search. The system exhausts its move budget analyzing alternatives before making a single move in the master game, prioritizing decision quality over speed.

## Key Features

- **🧠 Parallel Neural Network Exploration**: Spawn multiple NN instances for each decision
- **🌳 Budget-Constrained Tree Search**: Efficient exploration within computational limits  
- **🎯 Collision Avoidance Mastery**: Achieve consistent scores > 10 through deep thinking
- **🔄 Deterministic Reproducibility**: Seed-controlled randomness for reliable experiments
- **📡 ZeroMQ Headless Operation**: Complete message-based control without GUI dependencies
- **⚙️ Hydra Zen Configuration**: Flexible parameter management for concurrent systems
- **🧪 Comprehensive Testing**: Property-based tests with timeout protection

## Quick Start

### Installation

```bash
git clone <repository-url>
cd ai-hydra
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from ai_hydra import HydraMgr
from ai_hydra.config import SimulationConfig, NetworkConfig

# Create configuration for parallel NN exploration
sim_config = SimulationConfig(
    grid_size=(10, 10),
    move_budget=100,  # Budget for parallel exploration
    nn_enabled=True,
    random_seed=42
)

net_config = NetworkConfig(
    input_features=19,
    hidden_layers=(200, 200),
    output_actions=3,
    learning_rate=0.001
)

# Initialize the parallel NN system
hydra_mgr = HydraMgr(sim_config, net_config)

# Run simulation with concurrent NN exploration
result = hydra_mgr.run_simulation()
print(f"Final score: {result.final_score}")
print(f"Collision avoidance achieved: {result.final_score > 10}")
```

### Headless Operation

For production deployments, run the system headlessly:

```python
from ai_hydra.headless_server import HeadlessServer

# Start headless AI agent
server = HeadlessServer(port=5555)
server.start()

print("Headless AI agent running on port 5555")
print("Control via ZeroMQ messages")
```

Control via client:

```python
from ai_hydra.zmq_client_example import ZMQClient

# Connect to headless server
client = ZMQClient("tcp://localhost:5555")

# Start simulation
response = client.send_command("START_SIMULATION", {
    "grid_size": [8, 8],
    "move_budget": 30,
    "nn_enabled": True
})

# Monitor progress
status = client.send_command("GET_STATUS")
print(f"Current score: {status['current_score']}")
```

## Architecture

The system consists of several key components:

- **HydraMgr**: Main orchestration system managing parallel NN instances and exploration
- **GameBoard**: Immutable game state representation with perfect cloning
- **GameLogic**: Pure functions for game mechanics and move execution
- **Neural Network**: PyTorch-based move prediction with concurrent spawning
- **Tree Search**: Budget-constrained exploration with parallel NN evaluation
- **Oracle Trainer**: Learning system that improves NN from tree search results
- **ZeroMQ Server**: Headless communication layer for remote control

### Decision Flow

1. **Neural Network Prediction**: NN predicts optimal move from current state
2. **Tree Search Exploration**: Budget-constrained exploration starting from NN prediction
3. **Path Evaluation**: Select path with highest cumulative reward
4. **Oracle Training**: Compare NN prediction with tree search result
5. **Master Game Update**: Apply selected move and reset for next cycle

## Performance Expectations

- **Decision Quality over Speed**: System prioritizes thorough analysis over fast moves
- **Collision Avoidance**: Consistent scores > 10 through exhaustive exploration
- **Resource Intensive**: Expects slow execution due to parallel NN exploration
- **Scalable Budget**: Computational cost scales with move budget allocation

## Configuration

### Parallel NN Exploration

```python
config = SimulationConfig(
    grid_size=(8, 8),            # Smaller grid for faster exploration
    move_budget=100,             # Budget for parallel NN spawning
    nn_enabled=True,             # Enable concurrent NN instances
    collision_penalty=-20,       # Higher penalty for better learning
    random_seed=123
)
```

### Neural Network Architecture

```python
net_config = NetworkConfig(
    input_features=19,           # Standard feature vector size
    hidden_layers=(200, 200),    # Deep network for complex decisions
    output_actions=3,            # Left, Straight, Right
    learning_rate=0.001,         # Conservative learning rate
    batch_size=32                # Batch size for training
)
```

### Performance Tuning

```python
# For faster development testing
config = SimulationConfig(
    grid_size=(6, 6),            # Smaller grid
    move_budget=20,              # Reduced budget
    nn_enabled=True
)

net_config = NetworkConfig(
    hidden_layers=(50, 50),      # Smaller network
    learning_rate=0.01           # Faster learning
)
```

## ZeroMQ Communication Protocol

The system provides a comprehensive ZeroMQ-based protocol for headless operation:

### Command Messages
- `START_SIMULATION`: Initialize and start a new simulation
- `STOP_SIMULATION`: Stop current simulation
- `PAUSE_SIMULATION`: Pause execution while maintaining state
- `RESUME_SIMULATION`: Resume paused simulation
- `GET_STATUS`: Query current status and metrics

### Broadcast Messages
- `STATUS_UPDATE`: Real-time game state and performance metrics
- `DECISION_CYCLE_COMPLETE`: Results of each decision cycle
- `GAME_OVER`: Final simulation results
- `ERROR_OCCURRED`: Error notifications with recovery information

### Example Client

```python
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
```

## Testing

The system includes comprehensive testing with both unit tests and property-based tests:

```bash
# Run all tests
pytest

# Run with timeout protection
pytest --timeout=600

# Run specific test categories
pytest -m "unit"          # Unit tests only
pytest -m "property"      # Property-based tests only
pytest -m "integration"   # Integration tests only

# Run with coverage
pytest --cov=ai_hydra --cov-report=html
```

### Property-Based Testing

The system uses Hypothesis for property-based testing to validate universal properties:

```python
@given(
    config=simulation_configs(),
    budget=st.integers(min_value=10, max_value=1000)
)
@settings(max_examples=10, deadline=5000)
@pytest.mark.timeout(120)
def test_budget_lifecycle_management(config, budget):
    """Property 5: Budget Lifecycle Management"""
    # Test implementation validates budget invariants
```

## Documentation

Comprehensive documentation is available:

- **Getting Started**: Installation and basic usage
- **Architecture**: System design and component interaction
- **ZeroMQ Protocol**: Complete communication protocol reference
- **Deployment**: Production deployment scenarios
- **Troubleshooting**: Common issues and solutions
- **API Reference**: Complete API documentation

Build documentation locally:

```bash
cd docs
make html
# Open _build/html/index.html in browser
```

## Deployment

### Local Development

```bash
# Start headless server
python -m ai_hydra.headless_server

# Connect with example client
python -m ai_hydra.zmq_client_example --mode interactive
```

### Production Deployment

```bash
# Run as daemon with logging
python -m ai_hydra.headless_server \
    --bind "0.0.0.0" \
    --command-port 5555 \
    --broadcast-port 5556 \
    --log-file /var/log/snake_ai.log \
    --log-level INFO \
    --daemon
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 5555 5556

CMD ["python", "-m", "ai_hydra.headless_server", \
     "--bind", "0.0.0.0", \
     "--command-port", "5555", \
     "--broadcast-port", "5556"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive tests
4. Ensure all tests pass with timeout protection
5. Update documentation as needed
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
pytest --cov=ai_hydra

# Run property-based tests
pytest -m property --timeout=600

# Build documentation
cd docs && make html
```

## License

[License information to be added]

## Acknowledgments

This project implements advanced concepts in:
- Budget-constrained tree search algorithms
- Parallel neural network exploration
- Hybrid AI decision-making systems
- ZeroMQ-based distributed communication
- Property-based testing methodologies

The system prioritizes decision quality over speed, making it suitable for research applications where thorough analysis is more important than real-time performance.