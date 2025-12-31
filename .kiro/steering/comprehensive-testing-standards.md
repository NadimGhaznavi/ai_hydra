---
inclusion: always
---

# Comprehensive Testing Standards

This document defines the testing standards and practices for the AI Hydra project to ensure comprehensive test coverage, reliable test execution, and maintainable test suites.

## Testing Philosophy

### Core Principles

- **Comprehensive Coverage**: Test all critical functionality with multiple approaches
- **Reliability**: Tests should be deterministic and pass consistently
- **Maintainability**: Tests should be easy to understand, modify, and extend
- **Performance**: Test suite should complete within reasonable time limits
- **Documentation**: Tests serve as living documentation of system behavior

### Testing Pyramid

```
    /\
   /  \     E2E Tests (Few)
  /____\    - Full system integration
 /      \   - User workflow validation
/________\  Integration Tests (Some)
           - Component interaction
           - API contract testing
           
           Unit Tests (Many)
           - Individual function/class testing
           - Fast feedback loop
           
           Property-Based Tests (Critical)
           - Universal behavior validation
           - Edge case discovery
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual functions, classes, and methods in isolation

**Characteristics:**
- Fast execution (< 30 seconds per test)
- No external dependencies
- Focused on single responsibility
- High code coverage

**Example Structure:**
```python
class TestBudgetController:
    """Unit tests for BudgetController class."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.initial_budget = 100
        self.controller = BudgetController(self.initial_budget)
    
    def test_budget_initialization(self):
        """Test that BudgetController initializes correctly."""
        assert self.controller.get_remaining_budget() == self.initial_budget
        assert not self.controller.is_budget_exhausted()
    
    def test_budget_consumption(self):
        """Test budget decrements correctly."""
        self.controller.consume_move()
        assert self.controller.get_remaining_budget() == self.initial_budget - 1
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion detection."""
        for _ in range(self.initial_budget):
            self.controller.consume_move()
        assert self.controller.is_budget_exhausted()
        assert self.controller.get_remaining_budget() == 0
```

### 2. Property-Based Tests

**Purpose**: Test universal properties that should hold across all valid inputs

**Characteristics:**
- Generate many test cases automatically
- Discover edge cases humans might miss
- Validate correctness properties from design document
- Minimum 100 iterations per property

**Example Structure:**
```python
from hypothesis import given, strategies as st, settings

class TestBudgetManagement:
    """Property-based tests for budget management."""
    
    @given(
        initial_budget=st.integers(min_value=1, max_value=1000),
        moves=st.lists(st.just("move"), min_size=0, max_size=100)
    )
    @settings(max_examples=100, deadline=2000)
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

### 3. Integration Tests

**Purpose**: Test component interactions and system integration points

**Characteristics:**
- Test multiple components working together
- Validate API contracts and interfaces
- Test data flow between components
- Moderate execution time (< 2 minutes per test)

**Example Structure:**
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
```

### 4. End-to-End Tests

**Purpose**: Test complete user workflows and system behavior

**Characteristics:**
- Test from user perspective
- Include external dependencies
- Validate complete feature functionality
- Longer execution time (< 10 minutes per test)

**Example Structure:**
```python
class TestCompleteSimulation:
    """End-to-end tests for complete simulation workflows."""
    
    @pytest.mark.timeout(300)  # 5 minute timeout
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

### 5. Performance Tests

**Purpose**: Validate system performance and resource usage

**Characteristics:**
- Measure execution time and memory usage
- Validate performance requirements
- Detect performance regressions
- Run with realistic data sizes

**Example Structure:**
```python
class TestPerformance:
    """Performance tests for critical system components."""
    
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

## Test Organization

### Directory Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_game_board.py
│   ├── test_game_logic.py
│   ├── test_neural_network.py
│   ├── test_budget_controller.py
│   └── test_state_manager.py
├── property/                # Property-based tests
│   ├── test_deterministic_reproducibility.py
│   ├── test_budget_management.py
│   ├── test_clone_management.py
│   └── test_path_evaluation.py
├── integration/             # Integration tests
│   ├── test_simulation_pipeline.py
│   ├── test_hybrid_execution.py
│   ├── test_zmq_communication.py
│   └── test_headless_operation.py
├── e2e/                     # End-to-end tests
│   ├── test_complete_simulation.py
│   ├── test_user_workflows.py
│   └── test_deployment_scenarios.py
├── performance/             # Performance tests
│   ├── test_decision_cycle_performance.py
│   ├── test_memory_usage.py
│   └── test_scalability.py
├── fixtures/                # Test data and fixtures
│   ├── sample_configs.py
│   ├── test_game_states.py
│   └── mock_data.py
└── conftest.py             # Shared test configuration
```

### File Naming Conventions

- **Unit tests**: `test_{component_name}.py`
- **Property tests**: `test_{property_name}.py`
- **Integration tests**: `test_{integration_scenario}.py`
- **E2E tests**: `test_{user_workflow}.py`
- **Performance tests**: `test_{performance_aspect}.py`

## Test Configuration

### pytest Configuration

```toml
# pyproject.toml
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
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "slow: Slow tests (skip in CI)",
    "timeout: Tests with custom timeout limits",
]
```

### Timeout Management

**Timeout Requirements:**
- Unit tests: 30 seconds maximum
- Property-based tests: 60 seconds maximum
- Integration tests: 2-5 minutes maximum
- End-to-end tests: 5-10 minutes maximum
- Performance tests: Custom timeouts based on requirements

**Implementation:**
```python
import pytest

@pytest.mark.timeout(30)  # 30 second timeout for unit tests
def test_unit_functionality(self):
    """Unit test with timeout protection."""
    pass

@pytest.mark.timeout(120)  # 2 minute timeout for integration tests
def test_integration_scenario(self):
    """Integration test with timeout protection."""
    pass
```

### Test Data Management

**Deterministic Test Data:**
```python
# Use fixed seeds for reproducible tests
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

## Coverage Requirements

### Code Coverage Targets

- **Overall Coverage**: 90% minimum
- **Unit Test Coverage**: 95% minimum for core components
- **Integration Coverage**: 80% minimum for component interactions
- **Property Test Coverage**: 100% of testable acceptance criteria

### Coverage Analysis

```python
# Generate coverage reports
pytest --cov=ai_hydra --cov-report=html --cov-report=term-missing

# Check coverage thresholds
pytest --cov=ai_hydra --cov-fail-under=90
```

### Coverage Exclusions

```python
# .coveragerc
[run]
source = ai_hydra
omit = 
    */tests/*
    */test_*.py
    */conftest.py
    */setup.py
    */__main__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

## Test Execution Strategy

### Local Development

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/property/      # Property tests only
pytest tests/integration/   # Integration tests only

# Run with coverage
pytest --cov=ai_hydra --cov-report=html

# Run fast tests only (skip slow tests)
pytest -m "not slow"
```

### Continuous Integration

```yaml
# CI test matrix
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

### Test Execution Time Management

**Optimized Parameters for Testing:**
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

## Quality Assurance

### Test Review Checklist

**For All Tests:**
- [ ] Clear, descriptive test names
- [ ] Proper test isolation (no shared state)
- [ ] Appropriate assertions with descriptive messages
- [ ] Proper error handling and edge cases
- [ ] Deterministic behavior (no random failures)

**For Property Tests:**
- [ ] Universal quantification ("for all" statements)
- [ ] References to design document properties
- [ ] Appropriate generators for input space
- [ ] Sufficient iterations (minimum 100)
- [ ] Proper timeout configuration

**For Integration Tests:**
- [ ] Tests realistic component interactions
- [ ] Validates API contracts
- [ ] Includes error scenarios
- [ ] Reasonable execution time

### Test Maintenance

**Regular Maintenance Tasks:**
- Review and update test data quarterly
- Validate test execution times and optimize slow tests
- Update property tests when requirements change
- Refactor tests to maintain readability
- Remove obsolete tests for deprecated functionality

**Test Debt Management:**
- Track and prioritize missing test coverage
- Refactor complex or hard-to-maintain tests
- Update tests when refactoring production code
- Document known test limitations

## Specialized Testing Patterns

### Testing Neural Networks

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

### Testing ZeroMQ Communication

```python
@pytest.mark.asyncio
async def test_zmq_message_protocol():
    """Test ZeroMQ message protocol integrity."""
    server = ZMQServer()
    await server.start()
    
    try:
        # Test message serialization/deserialization
        original_message = ZMQMessage(
            message_type=MessageType.START_SIMULATION,
            timestamp=time.time(),
            data={"config": "test"}
        )
        
        serialized = server.serialize_message(original_message)
        deserialized = server.deserialize_message(serialized)
        
        assert deserialized == original_message
    finally:
        await server.stop()
```

### Testing Property-Based Test Failures

```python
# When a property test fails, use the failing example to create a regression test
def test_budget_edge_case_regression():
    """Regression test for specific budget management edge case."""
    # This test case was discovered by property-based testing
    initial_budget = 1
    controller = BudgetController(initial_budget)
    
    # The specific sequence that caused the failure
    controller.consume_move()
    
    # Verify the fix
    assert controller.get_remaining_budget() == 0
    assert controller.is_budget_exhausted()
```

## Test Automation and CI/CD Integration

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: Run unit tests
        entry: pytest tests/unit/ -x
        language: system
        pass_filenames: false
        
      - id: pytest-property-quick
        name: Run quick property tests
        entry: pytest tests/property/ --max-examples=10
        language: system
        pass_filenames: false
```

### GitHub Actions Integration

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11]
        test-category: [unit, property, integration]
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        run: |
          pytest tests/${{ matrix.test-category }}/ --cov=ai_hydra --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

This comprehensive testing standard ensures that the AI Hydra system is thoroughly tested, reliable, and maintainable while providing fast feedback to developers and comprehensive validation of system correctness.