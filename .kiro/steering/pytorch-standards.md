# PyTorch Standards and Best Practices

## Overview

This document outlines PyTorch-specific standards for the AI Hydra project, focusing on deterministic behavior, memory management, and integration with the tree search system.

## Deterministic Behavior

### 1. Seed Management
```python
def setup_deterministic_pytorch(seed: int):
    """Setup PyTorch for deterministic behavior."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    # Disable benchmarking for determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

### 2. Random State Isolation
```python
# Good: Isolated random state per simulation
class SimulationRunner:
    def __init__(self, seed: int):
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
    
    def generate_random_tensor(self):
        return torch.rand(10, generator=self.rng)

# Bad: Global random state
def generate_random_tensor():
    return torch.rand(10)  # Uses global state
```

### 3. Reproducible Neural Network Initialization
```python
class SnakeNet(nn.Module):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        self.input_layer = nn.Linear(19, 200)
        self.hidden_layer = nn.Linear(200, 200)
        self.output_layer = nn.Linear(200, 3)
        
        # Initialize weights deterministically
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
```

## Memory Management

### 1. Tensor Lifecycle
```python
# Good: Explicit memory management
def process_game_state(game_board: GameBoard) -> torch.Tensor:
    features = extract_features(game_board)
    tensor = torch.tensor(features, dtype=torch.float32)
    
    # Process tensor
    result = model(tensor)
    
    # Clean up if needed
    del tensor
    return result.detach()  # Detach from computation graph

# Bad: Memory leaks
def process_game_state_bad(game_board: GameBoard) -> torch.Tensor:
    features = extract_features(game_board)
    tensor = torch.tensor(features, dtype=torch.float32)
    result = model(tensor)
    return result  # Keeps computation graph in memory
```

### 2. Batch Processing
```python
def process_multiple_states(game_boards: List[GameBoard]) -> torch.Tensor:
    """Process multiple game states efficiently."""
    # Extract features for all boards
    features_list = [extract_features(board) for board in game_boards]
    
    # Stack into batch tensor
    batch_tensor = torch.stack([
        torch.tensor(features, dtype=torch.float32) 
        for features in features_list
    ])
    
    # Process batch
    with torch.no_grad():  # Disable gradients for inference
        results = model(batch_tensor)
    
    return results
```

### 3. GPU Memory Management
```python
def manage_gpu_memory():
    """Best practices for GPU memory management."""
    if torch.cuda.is_available():
        # Clear cache periodically
        torch.cuda.empty_cache()
        
        # Monitor memory usage
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        
        if allocated > MEMORY_THRESHOLD:
            # Implement memory cleanup strategy
            cleanup_old_tensors()
```

## Neural Network Integration

### 1. Feature Extraction Standards
```python
class FeatureExtractor:
    """Extract standardized features from game state."""
    
    FEATURE_SIZE = 19
    
    def extract_features(self, board: GameBoard) -> torch.Tensor:
        """
        Extract 19-dimensional feature vector.
        
        Features:
        - Snake collision directions (3): straight, left, right
        - Wall collision directions (3): straight, left, right  
        - Current direction flags (4): up, down, left, right
        - Food relative position (2): normalized dx, dy
        - Snake length binary (7): up to 127 segments
        """
        features = []
        
        # Collision features (6 total)
        features.extend(self._get_collision_features(board))
        
        # Direction features (4 total)
        features.extend(self._get_direction_features(board))
        
        # Food features (2 total)
        features.extend(self._get_food_features(board))
        
        # Snake length features (7 total)
        features.extend(self._get_snake_length_features(board))
        
        assert len(features) == self.FEATURE_SIZE
        return torch.tensor(features, dtype=torch.float32)
```

### 2. Model Architecture Standards
```python
class SnakeNet(nn.Module):
    """Standard neural network architecture for Snake AI."""
    
    def __init__(self, input_size: int = 19, hidden_size: int = 200, output_size: int = 3):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper tensor handling."""
        # Ensure input is 2D (batch_size, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        
        return self.softmax(x)
    
    def predict_move(self, features: torch.Tensor) -> int:
        """Predict single move from features."""
        with torch.no_grad():
            probabilities = self.forward(features)
            return torch.argmax(probabilities, dim=1).item()
```

### 3. Training Integration
```python
class OracleTrainer:
    """Train neural network from tree search results."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training metrics
        self.training_samples = []
        self.accuracy_history = []
    
    def update_from_oracle(self, features: torch.Tensor, optimal_move: int):
        """Update model when NN prediction differs from tree search."""
        # Get current prediction
        with torch.no_grad():
            prediction = self.model.predict_move(features)
        
        # Only train if prediction is wrong
        if prediction != optimal_move:
            self._train_step(features, optimal_move)
            self.training_samples.append((features.clone(), optimal_move))
    
    def _train_step(self, features: torch.Tensor, target_move: int):
        """Perform single training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(features)
        target = torch.tensor([target_move], dtype=torch.long)
        
        # Compute loss and backpropagate
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

## Performance Optimization

### 1. Inference Optimization
```python
def optimize_for_inference(model: nn.Module):
    """Optimize model for inference performance."""
    # Set to evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Consider TorchScript compilation for production
    if PRODUCTION_MODE:
        model = torch.jit.script(model)
    
    return model
```

### 2. Batch Processing
```python
def process_exploration_clones(clones: List[ExplorationClone], 
                             model: nn.Module) -> List[int]:
    """Process multiple clones efficiently with batching."""
    # Extract features from all clones
    features_batch = torch.stack([
        extract_features(clone.get_current_board()) 
        for clone in clones
    ])
    
    # Single forward pass for all clones
    with torch.no_grad():
        predictions = model(features_batch)
        moves = torch.argmax(predictions, dim=1)
    
    return moves.tolist()
```

### 3. Memory Profiling
```python
def profile_memory_usage():
    """Profile PyTorch memory usage during simulation."""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Use torch.profiler for detailed analysis
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Run simulation step
        run_simulation_step()
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

## Error Handling

### 1. Tensor Shape Validation
```python
def validate_tensor_shapes(tensor: torch.Tensor, expected_shape: Tuple[int, ...]):
    """Validate tensor has expected shape."""
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")

def safe_forward_pass(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Perform forward pass with shape validation."""
    validate_tensor_shapes(input_tensor, (1, 19))  # Batch size 1, 19 features
    
    try:
        output = model(input_tensor)
        validate_tensor_shapes(output, (1, 3))  # Batch size 1, 3 actions
        return output
    except RuntimeError as e:
        raise RuntimeError(f"Forward pass failed: {e}")
```

### 2. CUDA Error Handling
```python
def safe_cuda_operation(operation_func, *args, **kwargs):
    """Safely execute CUDA operations with fallback."""
    try:
        return operation_func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        # Clear cache and retry on CPU
        torch.cuda.empty_cache()
        # Move tensors to CPU and retry
        cpu_args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
        return operation_func(*cpu_args, **kwargs)
    except RuntimeError as e:
        if "CUDA" in str(e):
            # Fallback to CPU
            cpu_args = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
            return operation_func(*cpu_args, **kwargs)
        raise
```

## Testing Standards

### 1. Deterministic Testing
```python
def test_neural_network_determinism():
    """Test that neural network produces deterministic results."""
    seed = 12345
    
    # Create two identical networks
    torch.manual_seed(seed)
    net1 = SnakeNet()
    
    torch.manual_seed(seed)
    net2 = SnakeNet()
    
    # Test with same input
    input_tensor = torch.randn(1, 19)
    
    output1 = net1(input_tensor)
    output2 = net2(input_tensor)
    
    assert torch.allclose(output1, output2), "Networks should produce identical outputs"
```

### 2. Performance Testing
```python
def test_inference_performance():
    """Test neural network inference performance."""
    model = SnakeNet()
    model.eval()
    
    # Warm up
    dummy_input = torch.randn(1, 19)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(1000):
        with torch.no_grad():
            _ = model(dummy_input)
    
    avg_time = (time.time() - start_time) / 1000
    assert avg_time < 0.001, f"Inference too slow: {avg_time:.4f}s per call"
```

## Integration Guidelines

### 1. With Tree Search System
```python
def integrate_nn_with_tree_search(game_board: GameBoard, 
                                model: nn.Module,
                                tree_search_func) -> int:
    """Integrate neural network prediction with tree search validation."""
    # Get NN prediction
    features = extract_features(game_board)
    nn_prediction = model.predict_move(features)
    
    # Run tree search starting with NN prediction
    optimal_move = tree_search_func(game_board, initial_move=nn_prediction)
    
    # Use tree search result if different from NN
    if optimal_move != nn_prediction:
        # Generate training sample for later
        training_sample = (features, optimal_move)
        return optimal_move, training_sample
    
    return nn_prediction, None
```

### 2. With Configuration System
```python
def create_model_from_config(config: NetworkConfig) -> nn.Module:
    """Create neural network from configuration."""
    model = SnakeNet(
        input_size=config.input_features,
        hidden_size=config.hidden_layers[0],  # Use first hidden layer size
        output_size=config.output_actions
    )
    
    if config.training_enabled:
        model.train()
    else:
        model.eval()
    
    return model
```

## Documentation Standards

### 1. Docstring Format
```python
def extract_features(self, board: GameBoard) -> torch.Tensor:
    """
    Extract standardized feature vector from game board.
    
    Args:
        board: Current game board state
        
    Returns:
        torch.Tensor: Feature vector of shape (19,) with:
            - Collision features (6): snake/wall in straight/left/right directions
            - Direction features (4): current direction one-hot encoding
            - Food features (2): normalized relative position to food
            - Snake length features (7): binary representation of length
            
    Raises:
        ValueError: If board state is invalid
        
    Example:
        >>> board = GameBoard(...)
        >>> features = extractor.extract_features(board)
        >>> assert features.shape == (19,)
    """
```

### 2. Type Hints
```python
from typing import Optional, Tuple, List, Union
import torch
from torch import nn

def process_batch(inputs: torch.Tensor, 
                 model: nn.Module,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    """Process batch of inputs through model."""
    pass
```

This document ensures consistent PyTorch usage across the AI Hydra project, focusing on deterministic behavior, memory efficiency, and proper integration with the tree search system.