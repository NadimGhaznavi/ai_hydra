# Testing Standards

## Overview

This document defines testing standards for the AI Hydra project, covering unit tests, property-based tests, integration tests, and performance testing.

## Testing Philosophy

### Dual Testing Approach
The project uses both unit tests and property-based tests as complementary approaches:

- **Unit Tests**: Validate specific examples, edge cases, and error conditions
- **Property-Based Tests**: Validate universal properties across all inputs
- **Integration Tests**: Validate component interactions and end-to-end flows
- **Performance Tests**: Validate system performance and resource usage

### Test Execution Time Management
All tests must complete within reasonable time limits to support efficient development:

- **Timeout Requirements**: All long-running tests must include timeout decorators
- **Development Speed**: Full test suite should complete within 10 minutes on standard hardware
- **Optimized Parameters**: Use smaller networks, reduced budgets, and fewer iterations for testing
- **Timeout Limits**: 
  - Unit tests: 30 seconds maximum
  - Property-based tests: 60 seconds maximum
  - Integration tests: 2-5 minutes maximum
  - End-to-end tests: 5-10 minutes maximum
- **Failure Handling**: Timeout failures should provide diagnostic information

## Test Organization

### Directory Structure
```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_game_board.py
│   ├── test_game_logic.py
│   ├── test_neural_network.py
│   └── test_config.py
├── property/                # Property-based tests
│   ├── test_deterministic_reproducibility.py
│   ├── test_budget_management.py
│   └── test_clone_management.py
├── integration/             # Integration tests
│   ├── test_simulation_pipeline.py
│   ├── test_hybrid_execution.py
│   └── test_end_to_end.py
└── performance/             # Performance tests
    ├── test_tree_search_performance.py
    └── test_memory_usage.py
```

### File Naming Conventions
- Unit tests: `test_{component_name}.py`
- Property tests: `test_{property_name}.py`
- Integration tests: `test_{integration_scenario}.py`
- Performance tests: `test_{performance_aspect}.py`

## Unit Testing Standards

### 1. Test Structure
```python
class TestGameBoard:
    """Unit tests for GameBoard class."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.grid_size = (10, 10)
        self.initial_snake_length = 3
        self.seed = 42
    
    def test_board_initialization(self):
        """Test that GameBoard initializes correctly."""
        board = GameBoard.create_initial(
            self.grid_size, 
            self.initial_snake_length, 
            self.seed
        )
        
        assert board.grid_size == self.grid_size
        assert len(board.snake_body) == self.initial_snake_length
        assert board.score == 0
    
    def test_board_cloning(self):
        """Test that GameBoard cloning creates independent copies."""
        original = GameBoard.create_initial(self.grid_size, 3, self.seed)
        clone = original.clone()
        
        # Clones should be equal but independent
        assert original == clone
        assert original is not clone
        assert original.snake_body is not clone.snake_body
```

### 2. Edge Case Testing
```python
def test_edge_cases(self):
    """Test edge cases and boundary conditions."""
    # Minimum grid size
    min_board = GameBoard.create_initial((5, 5), 1, 42)
    assert min_board.grid_size == (5, 5)
    
    # Maximum snake length for grid
    max_snake = GameBoard.create_initial((10, 10), 8, 42)
    assert len(max_snake.snake_body) == 8
    
    # Edge positions
    corner_positions = [(0, 0), (9, 9), (0, 9), (9, 0)]
    for pos in corner_positions:
        # Test collision detection at corners
        collision = GameLogic.check_wall_collision(pos, (10, 10))
        assert collision  # Should detect wall collision
```

### 3. Error Condition Testing
```python
def test_invalid_inputs(self):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="Grid size must be at least"):
        GameBoard.create_initial((2, 2), 3, 42)
    
    with pytest.raises(ValueError, match="Snake length too large"):
        GameBoard.create_initial((5, 5), 10, 42)
    
    with pytest.raises(TypeError, match="Expected tuple"):
        GameBoard.create_initial("invalid", 3, 42)
```

### 4. Mock Usage Guidelines
```python
def test_with_mocks(self, mocker):
    """Use mocks sparingly and only for external dependencies."""
    # Good: Mock external file system
    mock_save = mocker.patch('builtins.open', mocker.mock_open())
    config_manager.save_config(config, "test.json")
    mock_save.assert_called_once()
    
    # Bad: Mock internal logic (test the real implementation)
    # mock_game_logic = mocker.patch('game_logic.execute_move')
```

## Property-Based Testing Standards

### 1. Property Test Structure
```python
class TestBudgetManagement:
    """Property-based tests for budget management."""
    
    @given(
        initial_budget=st.integers(min_value=1, max_value=1000),
        moves=st.lists(st.just("move"), min_size=0, max_size=100)
    )
    @settings(max_examples=50, deadline=2000)
    def test_budget_lifecycle_property(self, initial_budget, moves):
        """
        **Feature: ai-hydra, Property 5: Budget Lifecycle Management**
        **Validates: Requirements 4.1, 4.2, 6.2**
        
        For any initial budget and sequence of moves, the budget should
        decrease by exactly 1 per move and never go below 0.
        """
        controller = BudgetController(initial_budget)
        
        moves_executed = 0
        for _ in moves:
            if controller.get_remaining_budget() > 0:
                controller.consume_move()
                moves_executed += 1
            else:
                break
        
        expected_remaining = initial_budget - moves_executed
        assert controller.get_remaining_budget() == expected_remaining
        assert controller.get_remaining_budget() >= 0
```

### 2. Smart Generators
```python
@st.composite
def valid_game_boards(draw):
    """Generate valid game board states."""
    # Generate grid size
    width = draw(st.integers(min_value=8, max_value=20))
    height = draw(st.integers(min_value=8, max_value=20))
    grid_size = (width, height)
    
    # Generate snake length that fits in grid
    max_length = min(width, height) - 2
    snake_length = draw(st.integers(min_value=3, max_value=max_length))
    
    # Generate seed
    seed = draw(st.integers(min_value=0, max_value=2**31-1))
    
    return GameBoard.create_initial(grid_size, snake_length, seed)

@st.composite
def move_sequences(draw, max_length=20):
    """Generate realistic move sequences."""
    length = draw(st.integers(min_value=1, max_value=max_length))
    moves = draw(st.lists(
        st.sampled_from([Move.LEFT_TURN, Move.STRAIGHT, Move.RIGHT_TURN]),
        min_size=length,
        max_size=length
    ))
    return moves
```

### 3. Performance Configuration
```python
# For expensive operations (neural network, tree search)
@settings(max_examples=10, deadline=5000)
def test_expensive_property(self, data):
    """Test with reduced examples for expensive operations."""
    pass

# For fast operations (data structures)
@settings(max_examples=100, deadline=1000)
def test_fast_property(self, data):
    """Test with more examples for fast operations."""
    pass

# For debugging
@settings(max_examples=10, deadline=None, verbosity=Verbosity.verbose)
def test_debug_property(self, data):
    """Test with verbose output for debugging."""
    pass
```

## Integration Testing Standards

### 1. Component Integration
```python
class TestSimulationPipeline:
    """Integration tests for complete simulation pipeline."""
    
    def test_full_decision_cycle(self):
        """Test complete decision cycle from NN prediction to master move."""
        # Setup
        config = SimulationConfig(grid_size=(10, 10), move_budget=50)
        hydra_mgr = HydraMgr(config)
        
        # Execute one complete decision cycle
        initial_score = hydra_mgr.master_game.get_score()
        move_result = hydra_mgr.execute_decision_cycle()
        
        # Verify results
        assert move_result.move in [Move.LEFT_TURN, Move.STRAIGHT, Move.RIGHT_TURN]
        assert move_result.budget_used > 0
        assert move_result.paths_evaluated > 0
        
        # Verify master game state updated
        final_score = hydra_mgr.master_game.get_score()
        assert final_score >= initial_score  # Score should not decrease
    
    def test_neural_network_tree_search_integration(self):
        """Test integration between neural network and tree search."""
        board = GameBoard.create_initial((10, 10), 3, 42)
        nn_model = SnakeNet()
        
        # Get NN prediction
        features = FeatureExtractor().extract_features(board)
        nn_prediction = nn_model.predict_move(features)
        
        # Run tree search
        tree_result = TreeSearch().find_optimal_move(board, budget=20)
        
        # Integration should handle both results
        oracle_trainer = OracleTrainer(nn_model)
        final_move = oracle_trainer.compare_and_decide(
            nn_prediction, tree_result.optimal_move
        )
        
        assert final_move in [0, 1, 2]  # Valid move index
```

### 2. End-to-End Testing
```python
def test_complete_game_simulation(self):
    """Test complete game from start to finish."""
    config = SimulationConfig(
        grid_size=(8, 8),
        move_budget=30,
        random_seed=12345
    )
    
    simulation = SimulationPipeline(config)
    result = simulation.run_complete_game()
    
    # Verify game completed properly
    assert result.game_over is True
    assert result.final_score >= 0
    assert result.moves_executed > 0
    assert result.decision_cycles > 0
    
    # Verify logging captured all events
    assert len(result.log_entries) > 0
    assert any("GAME_OVER" in entry for entry in result.log_entries)
```

## Performance Testing Standards

### 1. Timing Tests
```python
def test_decision_cycle_performance(self):
    """Test that decision cycles complete within time limits."""
    config = SimulationConfig(move_budget=100)
    hydra_mgr = HydraMgr(config)
    
    start_time = time.time()
    result = hydra_mgr.execute_decision_cycle()
    execution_time = time.time() - start_time
    
    # Should complete within reasonable time
    assert execution_time < 5.0, f"Decision cycle too slow: {execution_time:.2f}s"
    
    # Should use budget efficiently
    efficiency = result.paths_evaluated / result.budget_used
    assert efficiency > 0.1, f"Poor budget efficiency: {efficiency:.3f}"
```

### 2. Memory Tests
```python
def test_memory_usage(self):
    """Test memory usage stays within bounds."""
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run multiple decision cycles
    config = SimulationConfig(move_budget=50)
    hydra_mgr = HydraMgr(config)
    
    for _ in range(10):
        hydra_mgr.execute_decision_cycle()
        gc.collect()  # Force garbage collection
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    assert memory_increase < 100, f"Memory leak detected: {memory_increase:.1f}MB increase"
```

### 3. Scalability Tests
```python
@pytest.mark.parametrize("budget", [10, 50, 100, 200])
def test_budget_scalability(self, budget):
    """Test performance scales reasonably with budget size."""
    config = SimulationConfig(move_budget=budget)
    hydra_mgr = HydraMgr(config)
    
    start_time = time.time()
    result = hydra_mgr.execute_decision_cycle()
    execution_time = time.time() - start_time
    
    # Time should scale sub-linearly with budget
    time_per_budget = execution_time / budget
    assert time_per_budget < 0.1, f"Poor scalability: {time_per_budget:.3f}s per budget unit"
```

## Test Configuration

### 1. pytest Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "--cov=ai_hydra",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--timeout=600",  # 10 minute global timeout
]
markers = [
    "unit: Unit tests",
    "property: Property-based tests", 
    "integration: Integration tests",
    "performance: Performance tests",
    "slow: Slow tests (skip in CI)",
    "timeout: Tests with custom timeout limits",
]
```

### 2. Timeout Configuration
```python
import pytest

# Timeout decorator for different test types
def timeout_test(seconds):
    """Decorator to add timeout to tests."""
    return pytest.mark.timeout(seconds)

class TestExample:
    @timeout_test(30)  # 30 second timeout for unit tests
    def test_unit_functionality(self):
        """Unit test with timeout protection."""
        pass
    
    @timeout_test(120)  # 2 minute timeout for integration tests
    def test_integration_scenario(self):
        """Integration test with timeout protection."""
        pass
    
    @timeout_test(300)  # 5 minute timeout for end-to-end tests
    def test_end_to_end_simulation(self):
        """End-to-end test with timeout protection."""
        pass
```

### 3. Optimized Test Parameters
```python
# Use smaller parameters for faster testing
TEST_CONFIG = {
    "grid_size": (6, 6),        # Smaller than production (8, 8)
    "move_budget": 15,          # Smaller than production (50-100)
    "nn_hidden_size": (15, 15), # Smaller than production (200, 200)
    "max_examples": 10,         # Reduced for property tests
    "test_iterations": 2        # Fewer simulation runs
}
```

### 4. Test Fixtures
```python
@pytest.fixture
def sample_game_board():
    """Provide a standard game board for testing."""
    return GameBoard.create_initial((10, 10), 3, 42)

@pytest.fixture
def simulation_config():
    """Provide a standard simulation configuration."""
    return SimulationConfig(
        grid_size=(8, 8),
        move_budget=50,
        random_seed=12345
    )

@pytest.fixture
def neural_network():
    """Provide a trained neural network for testing."""
    net = SnakeNet()
    # Load pre-trained weights if available
    return net
```

### 3. Test Markers
```python
@pytest.mark.unit
def test_game_board_creation():
    """Unit test for game board creation."""
    pass

@pytest.mark.property
@pytest.mark.slow
def test_deterministic_reproducibility():
    """Property-based test (marked as slow)."""
    pass

@pytest.mark.integration
def test_simulation_pipeline():
    """Integration test."""
    pass

@pytest.mark.performance
@pytest.mark.slow
def test_memory_usage():
    """Performance test (marked as slow)."""
    pass
```

## Continuous Integration

### 1. Test Execution Strategy
```yaml
# CI configuration example
test_matrix:
  fast_tests:
    - pytest tests/unit/ -m "not slow"
    - pytest tests/property/ -m "not slow" --max-examples=20
  
  comprehensive_tests:
    - pytest tests/ --max-examples=50
    - pytest tests/performance/ -m "not slow"
  
  nightly_tests:
    - pytest tests/ --max-examples=100
    - pytest tests/performance/
```

### 2. Coverage Requirements
- Unit tests: 90% line coverage minimum
- Integration tests: Cover all major component interactions
- Property tests: Cover all testable acceptance criteria
- Performance tests: Cover critical performance paths

### 3. Test Data Management
```python
# Use deterministic test data
TEST_SEEDS = [42, 12345, 999, 54321, 777]

@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_with_multiple_seeds(seed):
    """Test with multiple deterministic seeds."""
    pass

# Avoid random test data in CI
def test_deterministic_only():
    """Use only deterministic inputs in CI."""
    # Good: Deterministic
    test_data = [1, 2, 3, 4, 5]
    
    # Bad: Random (non-deterministic in CI)
    # test_data = [random.randint(1, 100) for _ in range(5)]
```

## Documentation and Reporting

### 1. Test Documentation
```python
def test_complex_scenario(self):
    """
    Test complex scenario with multiple components.
    
    This test validates the interaction between:
    - Neural network prediction
    - Tree search validation  
    - Oracle training
    - Master game state updates
    
    Expected behavior:
    1. NN makes initial prediction
    2. Tree search validates/corrects prediction
    3. Oracle trainer updates NN if needed
    4. Master game applies final move
    
    Validates: Requirements 11.3, 11.4, 11.5
    """
```

### 2. Test Reports
- Generate HTML coverage reports
- Include property test statistics
- Track performance regression
- Document test failures with context

### 3. Debugging Support
```python
def test_with_debug_info(self):
    """Include debug information for test failures."""
    board = create_test_board()
    
    try:
        result = complex_operation(board)
        assert result.is_valid()
    except AssertionError:
        # Provide debug context
        print(f"Board state: {board}")
        print(f"Result: {result}")
        print(f"Debug info: {result.get_debug_info()}")
        raise
```

This testing standard ensures comprehensive validation of the AI Hydra system while maintaining good performance and clear documentation.