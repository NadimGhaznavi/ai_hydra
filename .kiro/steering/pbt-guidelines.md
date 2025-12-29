# Property-Based Testing Guidelines

## Overview

This project uses property-based testing (PBT) with Hypothesis to validate correctness properties across the AI Hydra system. PBT generates many test cases automatically to find edge cases that manual testing might miss.

## Core Principles

### 1. Universal Quantification
Every property must contain an explicit "for all" or "for any" statement:
```python
# Good: Universal quantification
def test_budget_lifecycle(self, budget):
    """For any valid budget value, the system should..."""
    
# Bad: Specific example
def test_budget_with_100():
    """Test budget with value 100"""
```

### 2. Requirements Traceability
Each property test must reference the requirements it validates:
```python
def test_deterministic_reproducibility(self, seed):
    """
    **Feature: ai-hydra, Property 11: Deterministic Reproducibility**
    **Validates: Requirements 7.1, 7.2, 7.5**
    """
```

### 3. Smart Generators
Write generators that constrain to valid input spaces:
```python
@st.composite
def game_states(draw):
    """Generate valid game states for testing."""
    grid_size = draw(st.tuples(
        st.integers(min_value=8, max_value=20),
        st.integers(min_value=8, max_value=20)
    ))
    # Ensure snake fits in grid
    max_snake_length = min(grid_size) - 3
    snake_length = draw(st.integers(min_value=3, max_value=max_snake_length))
    # ... more constraints
```

## Test Configuration

### Standard Settings
```python
@settings(max_examples=10, deadline=5000)  # For complex tests
@settings(max_examples=100, deadline=1000) # For simple tests
```

### Performance Considerations
- Use `max_examples=10` for expensive operations (tree search, neural network)
- Use `max_examples=100` for fast operations (data structure manipulation)
- Set reasonable `deadline` values to prevent timeouts

## Common Patterns

### 1. Round-Trip Properties
```python
def test_serialization_round_trip(self, game_board):
    """For any game board, serialize then deserialize should be identity."""
    serialized = serialize(game_board)
    deserialized = deserialize(serialized)
    assert game_board == deserialized
```

### 2. Invariant Properties
```python
def test_budget_invariant(self, initial_budget, moves):
    """For any budget and move sequence, remaining budget should be non-negative."""
    controller = BudgetController(initial_budget)
    for _ in moves:
        if controller.get_remaining_budget() > 0:
            controller.consume_move()
    assert controller.get_remaining_budget() >= 0
```

### 3. Metamorphic Properties
```python
def test_clone_independence(self, game_board):
    """For any game board, clones should be independent."""
    clone1 = game_board.clone()
    clone2 = game_board.clone()
    # Modify clone1
    result1 = GameLogic.execute_move(clone1, Move.LEFT)
    # clone2 should be unchanged
    assert clone2 == game_board
```

## Error Handling in Tests

### Triaging Counter-Examples
When a property test fails:

1. **Analyze the counter-example**: Is it a valid input that reveals a bug?
2. **Check test correctness**: Is the property correctly specified?
3. **Verify requirements**: Does the failure indicate missing requirements?

```python
# Use assume() to filter invalid inputs
@given(st.integers())
def test_positive_values(self, x):
    assume(x > 0)  # Skip negative values
    assert process_positive(x) > 0
```

### Debugging Failed Properties
```python
# Add detailed assertions for debugging
def test_complex_property(self, data):
    result = complex_operation(data)
    assert result.is_valid(), f"Invalid result for input {data}: {result}"
    assert result.meets_constraint(), f"Constraint violation: {result.debug_info()}"
```

## Integration with Spec Requirements

### Mapping Properties to Requirements
Each property test should validate specific acceptance criteria:

```python
# From requirements.md: "7.1 THE Seed_Manager SHALL ensure all random events use deterministic seeds"
def test_deterministic_seeds(self, seed):
    """
    Property 11: Deterministic Reproducibility
    Validates: Requirements 7.1, 7.2, 7.5
    
    For any seed value, identical configurations should produce identical results.
    """
```

### Coverage Analysis
Ensure all testable acceptance criteria have corresponding property tests:
- Review requirements.md for EARS patterns
- Identify which criteria are testable as properties vs examples
- Create property tests for universal behaviors
- Use unit tests for specific examples and edge cases

## Best Practices

### 1. Keep Properties Simple
```python
# Good: Single concern
def test_budget_decreases(self, budget):
    """Budget should decrease after consuming moves."""
    
# Bad: Multiple concerns
def test_budget_and_clones_and_logging(self, budget, clones, log_level):
    """Test budget, clone creation, and logging together."""
```

### 2. Use Descriptive Names
```python
# Good
def test_exploration_clones_maintain_independence(self, game_state):
    
# Bad  
def test_clones(self, state):
```

### 3. Document Expected Behavior
```python
def test_tree_expansion_logic(self, surviving_clone):
    """
    When a clone survives (reward >= 0) and budget remains,
    exactly 3 sub-clones should be created.
    
    This validates the tree expansion invariant.
    """
```

## Performance Guidelines

### 1. Optimize Expensive Operations
```python
# Cache expensive setups
@st.composite
def neural_networks(draw):
    # Cache network creation
    if not hasattr(neural_networks, '_cache'):
        neural_networks._cache = create_test_network()
    return neural_networks._cache
```

### 2. Use Appropriate Example Counts
- Complex tree search: `max_examples=5-10`
- Neural network operations: `max_examples=10-20`
- Simple data operations: `max_examples=100`

### 3. Set Reasonable Timeouts
```python
@settings(deadline=10000)  # 10 seconds for complex operations
def test_full_simulation_cycle(self, config):
    """Test complete simulation with generous timeout."""
```

## Common Pitfalls

### 1. Over-Constraining Generators
```python
# Bad: Too restrictive
@given(st.integers(min_value=42, max_value=42))  # Only tests one value

# Good: Reasonable range
@given(st.integers(min_value=1, max_value=1000))
```

### 2. Non-Deterministic Tests
```python
# Bad: Uses system time
def test_with_timestamp():
    timestamp = time.time()
    # Test depends on current time
    
# Good: Use deterministic inputs
@given(st.integers(min_value=0, max_value=2**31-1))
def test_with_seed(self, seed):
    # Test uses provided seed
```

### 3. Ignoring Shrinking
```python
# Let Hypothesis shrink to minimal failing examples
# Don't use overly complex data structures that prevent shrinking
```

## Integration with CI/CD

Property-based tests should run in CI with:
- Consistent seed for reproducible failures
- Appropriate timeouts for CI environment
- Clear failure reporting with counter-examples

```python
# In CI configuration
@settings(
    max_examples=50,  # Reduced for CI speed
    deadline=5000,    # 5 second timeout
    derandomize=True  # Consistent results
)
```