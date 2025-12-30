Testing Guide
=============

This document provides comprehensive information about the AI Hydra testing framework, including test organization, execution instructions, and coverage details.

Overview
--------

The AI Hydra project uses a **dual testing approach** combining unit tests and property-based tests to ensure comprehensive validation:

- **Unit Tests**: Validate specific examples, edge cases, and error conditions
- **Property-Based Tests**: Validate universal properties across all inputs using Hypothesis
- **Integration Tests**: Validate component interactions and end-to-end flows
- **Performance Tests**: Validate system performance and resource usage

Testing Framework
-----------------

The project uses the following testing tools:

- **pytest**: Primary testing framework with extensive plugin support
- **Hypothesis**: Property-based testing library for generating test cases
- **pytest-cov**: Code coverage reporting
- **pytest-timeout**: Timeout protection for long-running tests
- **pytest-asyncio**: Support for async/await testing patterns

Test Organization
-----------------

Tests are organized in the ``tests/`` directory with the following structure:

.. code-block::

    tests/
    ├── __init__.py
    ├── property/                           # Property-based tests
    │   └── test_efficiency_based_selection.py
    ├── test_budget_controller.py           # Budget management tests
    ├── test_collision_avoidance_improvement.py  # AI improvement tests
    ├── test_deterministic_reproducibility.py    # Reproducibility tests
    ├── test_end_to_end_simulation.py       # Complete simulation tests
    ├── test_error_handling.py              # Error handling and recovery
    ├── test_game_board.py                  # Game state management
    ├── test_game_logic.py                  # Core game mechanics
    ├── test_headless_operation.py          # ZeroMQ headless server
    ├── test_hybrid_execution.py            # Neural network + tree search
    ├── test_hydra_zen_integration.py       # Configuration management
    ├── test_logging_system.py              # Logging and monitoring
    ├── test_neural_network.py              # Neural network components
    ├── test_oracle_trainer.py              # NN training from tree search
    ├── test_parallel_nn_lifecycle.py       # Parallel NN management
    ├── test_realtime_status_broadcasting.py # ZeroMQ status updates
    ├── test_simple_collision.py            # Basic collision detection
    ├── test_simulation_pipeline.py         # Complete pipeline tests
    └── test_zmq_communication.py           # ZeroMQ protocol tests

Test Categories
---------------

Unit Tests
~~~~~~~~~~

Unit tests focus on individual components and specific scenarios:

- **Game Logic Tests** (``test_game_logic.py``): Core game mechanics, move execution, collision detection
- **Game Board Tests** (``test_game_board.py``): State management, cloning, immutability
- **Neural Network Tests** (``test_neural_network.py``): Feature extraction, network architecture, training
- **Budget Controller Tests** (``test_budget_controller.py``): Resource management, budget tracking
- **Error Handling Tests** (``test_error_handling.py``): Exception handling, recovery mechanisms

Property-Based Tests
~~~~~~~~~~~~~~~~~~~~

Property-based tests validate universal properties using generated test data:

- **Deterministic Reproducibility** (``test_deterministic_reproducibility.py``): Seed-based consistency
- **GameBoard Cloning** (``test_game_board.py``): Perfect state copying and independence
- **Efficiency-Based Selection** (``tests/property/test_efficiency_based_selection.py``): Optimal path selection

Integration Tests
~~~~~~~~~~~~~~~~~

Integration tests validate component interactions:

- **End-to-End Simulation** (``test_end_to_end_simulation.py``): Complete game simulations
- **Simulation Pipeline** (``test_simulation_pipeline.py``): Component coordination
- **Hybrid Execution** (``test_hybrid_execution.py``): Neural network + tree search integration
- **Headless Operation** (``test_headless_operation.py``): ZeroMQ server functionality

Performance Tests
~~~~~~~~~~~~~~~~~

Performance tests validate system efficiency and resource usage:

- **Collision Avoidance Improvement** (``test_collision_avoidance_improvement.py``): AI learning metrics
- **Parallel NN Lifecycle** (``test_parallel_nn_lifecycle.py``): Resource management efficiency

Running Tests
-------------

Prerequisites
~~~~~~~~~~~~~

Ensure you have the development dependencies installed:

.. code-block:: bash

    # Install with development dependencies
    pip install -e ".[dev]"
    
    # Or install specific testing dependencies
    pip install pytest pytest-cov pytest-timeout pytest-asyncio hypothesis

Basic Test Execution
~~~~~~~~~~~~~~~~~~~~~

Run All Tests
^^^^^^^^^^^^^

.. code-block:: bash

    # Run all tests with coverage
    pytest
    
    # Run all tests with verbose output
    pytest -v
    
    # Run all tests with coverage report
    pytest --cov=ai_hydra --cov-report=html

Run Specific Test Files
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run a specific test file
    pytest tests/test_game_logic.py
    
    # Run multiple specific files
    pytest tests/test_game_logic.py tests/test_game_board.py
    
    # Run tests with pattern matching
    pytest tests/test_*neural*.py

Run Specific Test Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run a specific test function
    pytest tests/test_game_logic.py::test_execute_move_basic
    
    # Run a specific test class
    pytest tests/test_game_logic.py::TestGameLogic
    
    # Run a specific method in a class
    pytest tests/test_game_logic.py::TestGameLogic::test_collision_detection

Test Categories and Markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project uses pytest markers to categorize tests:

Run by Test Type
^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run only unit tests
    pytest -m unit
    
    # Run only property-based tests
    pytest -m property
    
    # Run only integration tests
    pytest -m integration
    
    # Run only performance tests
    pytest -m performance

Run by Speed
^^^^^^^^^^^^

.. code-block:: bash

    # Run fast tests only (exclude slow tests)
    pytest -m "not slow"
    
    # Run only slow tests
    pytest -m slow
    
    # Run with custom timeout
    pytest --timeout=300  # 5 minute timeout

Run Async Tests
^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run async tests (ZeroMQ, headless server)
    pytest -m asyncio
    
    # Run async tests with specific event loop
    pytest -m asyncio --asyncio-mode=auto

Advanced Test Execution
~~~~~~~~~~~~~~~~~~~~~~~~

Parallel Test Execution
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Install pytest-xdist for parallel execution
    pip install pytest-xdist
    
    # Run tests in parallel (4 workers)
    pytest -n 4
    
    # Run tests in parallel with auto worker detection
    pytest -n auto

Test Selection and Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run tests matching a keyword
    pytest -k "neural_network"
    
    # Run tests NOT matching a keyword
    pytest -k "not slow"
    
    # Run tests matching multiple keywords
    pytest -k "neural_network and not slow"
    
    # Run tests by directory
    pytest tests/property/

Debugging and Verbose Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run with maximum verbosity
    pytest -vvv
    
    # Show local variables on failure
    pytest -l
    
    # Drop into debugger on failure
    pytest --pdb
    
    # Stop on first failure
    pytest -x
    
    # Show test output (print statements)
    pytest -s

Coverage Reporting
~~~~~~~~~~~~~~~~~~

Generate Coverage Reports
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Generate HTML coverage report
    pytest --cov=ai_hydra --cov-report=html
    
    # Generate terminal coverage report
    pytest --cov=ai_hydra --cov-report=term-missing
    
    # Generate XML coverage report (for CI)
    pytest --cov=ai_hydra --cov-report=xml
    
    # Generate multiple report formats
    pytest --cov=ai_hydra --cov-report=html --cov-report=term-missing

View Coverage Results
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # View HTML coverage report
    open htmlcov/index.html  # macOS
    xdg-open htmlcov/index.html  # Linux
    
    # Coverage reports are generated in:
    # - htmlcov/          (HTML reports)
    # - coverage.xml      (XML report)

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~

Property-based tests use Hypothesis to generate test cases. You can control their behavior:

.. code-block:: bash

    # Run property tests with more examples
    pytest tests/test_deterministic_reproducibility.py --hypothesis-show-statistics
    
    # Run property tests with custom settings
    pytest -s tests/test_game_board.py::test_perfect_gameboard_cloning

Test Configuration
------------------

The test configuration is defined in ``pyproject.toml``:

.. code-block:: toml

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
        "asyncio: Async tests",
    ]

Test Implementation Details
---------------------------

Property-Based Test Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Property-based tests validate universal properties:

.. code-block:: python

    @given(board=game_boards())
    def test_perfect_gameboard_cloning(board):
        """
        Feature: ai-hydra, Property 3: Perfect GameBoard Cloning
        
        For any GameBoard cloning operation, the cloned board should be 
        identical to the source board in all aspects and completely independent.
        
        **Validates: Requirements 3.4, 6.4, 7.3, 10.4**
        """
        cloned_board = board.clone()
        
        # Test identity
        assert cloned_board.snake_head == board.snake_head
        assert cloned_board.snake_body == board.snake_body
        
        # Test independence
        assert cloned_board is not board
        assert cloned_board.snake_body is not board.snake_body

Unit Test Examples
~~~~~~~~~~~~~~~~~~

Unit tests validate specific behaviors:

.. code-block:: python

    def test_execute_move_basic(self):
        """Test basic move execution without collisions."""
        board = GameLogic.create_initial_board((10, 10), 3, 42)
        result = GameLogic.execute_move(board, Move.STRAIGHT)
        
        assert result.outcome == MoveOutcome.EMPTY
        assert result.reward == 0
        assert not result.is_terminal

Integration Test Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests validate component interactions:

.. code-block:: python

    @timeout_test(120)  # 2 minute timeout
    def test_complete_game_simulation_start_to_finish(self):
        """Test complete game simulations from start to finish."""
        sim_config = SimulationConfig(
            grid_size=(6, 6),
            move_budget=10,
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        result = pipeline.run_complete_simulation()
        
        assert result.success
        assert result.game_result.total_moves > 0

Continuous Integration
----------------------

The test suite is designed to run efficiently in CI environments:

Fast Test Subset
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run fast tests only (for CI)
    pytest -m "not slow" --timeout=300
    
    # Run with reduced property test examples
    pytest --hypothesis-max-examples=10

Full Test Suite
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run complete test suite (for nightly builds)
    pytest --cov=ai_hydra --cov-report=xml --timeout=1800

Test Data and Fixtures
----------------------

The tests use various fixtures and test data:

- **Deterministic Seeds**: Tests use fixed seeds (42, 12345, etc.) for reproducibility
- **Reduced Parameters**: Tests use smaller grids, budgets, and networks for speed
- **Mock Objects**: ZeroMQ communication uses mock servers for isolated testing
- **Timeout Protection**: All long-running tests have timeout decorators

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Tests Timeout**

.. code-block:: bash

    # Increase timeout for specific tests
    pytest --timeout=1200  # 20 minutes
    
    # Run without timeout (for debugging)
    pytest --timeout=0

**Property Tests Fail Intermittently**

.. code-block:: bash

    # Run with fixed seed for reproducibility
    pytest --hypothesis-seed=42
    
    # Reduce examples for faster execution
    pytest --hypothesis-max-examples=5

**Memory Issues**

.. code-block:: bash

    # Run tests sequentially (no parallel execution)
    pytest -n 0
    
    # Run specific test subsets
    pytest tests/test_game_logic.py tests/test_game_board.py

**ZeroMQ Tests Fail**

.. code-block:: bash

    # Run async tests with specific configuration
    pytest -m asyncio --asyncio-mode=auto
    
    # Skip ZeroMQ tests if needed
    pytest -m "not asyncio"

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Property-based tests** use reduced examples (5-10) for expensive operations
- **Neural network tests** use smaller architectures (15x15 instead of 200x200)
- **Integration tests** use smaller grids (6x6) and reduced budgets (10-20 moves)
- **Timeout limits** prevent tests from running indefinitely

Test Coverage Goals
-------------------

The project maintains high test coverage standards:

- **Unit Tests**: 90% line coverage minimum
- **Integration Tests**: Cover all major component interactions  
- **Property Tests**: Cover all testable acceptance criteria
- **Performance Tests**: Cover critical performance paths

Current coverage can be viewed by running:

.. code-block:: bash

    pytest --cov=ai_hydra --cov-report=term-missing

Contributing Tests
------------------

When adding new tests:

1. **Follow naming conventions**: ``test_*.py`` files, ``test_*`` functions
2. **Use appropriate markers**: ``@pytest.mark.unit``, ``@pytest.mark.property``, etc.
3. **Add timeouts**: Use ``@timeout_test(seconds)`` for long-running tests
4. **Document properties**: Include requirement validation comments
5. **Use fixtures**: Leverage existing fixtures for common test data

Example test structure:

.. code-block:: python

    """
    Tests for new component.
    
    **Validates: Requirements X.Y, X.Z**
    """
    
    import pytest
    from hypothesis import given, strategies as st
    
    @pytest.mark.unit
    def test_specific_behavior(self):
        """Test specific behavior with concrete examples."""
        # Test implementation
        pass
    
    @pytest.mark.property
    @given(data=st.data())
    def test_universal_property(self, data):
        """
        **Feature: ai-hydra, Property N: Property Name**
        **Validates: Requirements X.Y**
        
        For any valid input, the system should...
        """
        # Property test implementation
        pass

This comprehensive testing framework ensures the AI Hydra system maintains high quality and reliability across all components.