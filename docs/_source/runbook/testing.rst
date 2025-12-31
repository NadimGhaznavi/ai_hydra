Testing Guide
=============

This document provides comprehensive information about the AI Hydra testing framework, including test organization, execution instructions, and coverage details.

For complete development standards including directory layout and file organization, see :doc:`development_standards`.

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
- **pytest-timeout**: Timeout protection for long-running tests (critical for decision cycles)
- **pytest-asyncio**: Support for async/await testing patterns (ZeroMQ components)

**Timeout Requirements**: All tests must include timeout decorators to prevent hanging during decision cycle testing. The enhanced decision flow with 9 distinct states requires careful timeout management:

- Unit tests: 30 seconds maximum
- Property-based tests: 60 seconds maximum  
- Integration tests: 2-5 minutes maximum
- End-to-end tests: 5-10 minutes maximum

Test Organization
-----------------

Tests are organized in the ``tests/`` directory with the following structure:

.. code-block::

    tests/
    ├── __init__.py
    ├── property/                           # Property-based tests
    │   ├── test_efficiency_based_selection.py
    │   ├── test_token_tracker_data_validation.py
    │   ├── test_token_tracker_error_recovery.py
    │   ├── test_token_tracker_hook_integration.py
    │   ├── test_token_tracker_configuration_management.py
    │   └── test_token_tracker_special_characters.py
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

**Core System Properties:**

- **Deterministic Reproducibility** (``test_deterministic_reproducibility.py``): Seed-based consistency
- **GameBoard Cloning** (``test_game_board.py``): Perfect state copying and independence
- **Efficiency-Based Selection** (``tests/property/test_efficiency_based_selection.py``): Optimal path selection

**Token Tracking Properties:**

- **Property 1: CSV Transaction Persistence** (``test_token_tracker_data_validation.py``): All valid transactions can be stored and retrieved correctly
- **Property 2: Data Append Safety** (``test_token_tracker_data_validation.py``): Concurrent append operations preserve data integrity
- **Property 4: Hook-Tracker Integration** (``test_token_tracker_hook_integration.py``): Agent hook automatically triggers token tracking and records complete metadata
- **Property 5: Error Recovery Resilience** (``test_token_tracker_error_recovery.py``): System recovers gracefully from various error conditions
- **Property 6: Configuration State Management** (``test_token_tracker_configuration_management.py``): Hook maintains consistent configuration state and supports runtime updates
- **Property 7: Special Character Handling** (``test_token_tracker_special_characters.py``): Unicode and CSV special characters are handled correctly
- **Property 8: Data Validation Integrity** (``test_token_tracker_data_validation.py``): All data validation rules are enforced consistently

**Token Tracking Property Test Examples:**

.. code-block:: python

    @given(
        special_text=st.text(
            alphabet=st.characters(
                categories=("Lu", "Ll", "Nd", "Po", "Ps", "Pe"),
                include_characters=["\n", "\r", "\t", '"', "'", ",", ";", "\\"]
            ),
            min_size=1,
            max_size=500
        ),
        tokens_used=st.integers(min_value=1, max_value=10000),
        elapsed_time=st.floats(min_value=0.001, max_value=60.0)
    )
    @settings(max_examples=100, deadline=3000)
    def test_special_character_handling_property(self, special_text, tokens_used, elapsed_time):
        """
        **Feature: kiro-token-tracker, Property 7: Special Character Handling**
        **Validates: Requirements 6.3, 6.4**

        For any text containing special characters, newlines, or Unicode content,
        the CSV encoding should preserve the text exactly and remain readable by
        standard spreadsheet tools.
        """
        # Test implementation validates Unicode handling, CSV safety, and round-trip integrity

Integration Tests
~~~~~~~~~~~~~~~~~

Integration tests validate component interactions:

- **End-to-End Simulation** (``test_end_to_end_simulation.py``): Complete game simulations
- **Simulation Pipeline** (``test_simulation_pipeline.py``): Component coordination
- **Hybrid Execution** (``test_hybrid_execution.py``): Neural network + tree search integration
- **Headless Operation** (``test_headless_operation.py``): ZeroMQ server functionality

Decision Flow Testing
~~~~~~~~~~~~~~~~~~~~~

The enhanced decision flow architecture with 9 distinct states requires comprehensive testing:

**State Transition Testing**

.. code-block:: python

    @pytest.mark.integration
    @timeout_test(180)  # 3 minute timeout
    def test_decision_cycle_state_transitions(self):
        """Test all 9 decision cycle states execute in correct order."""
        config = SimulationConfig(grid_size=(6, 6), move_budget=15, random_seed=42)
        hydra_mgr = HydraMgr(config)
        
        # Mock state tracker to verify state transitions
        state_tracker = DecisionCycleStateTracker()
        hydra_mgr.add_state_observer(state_tracker)
        
        # Execute decision cycle
        result = hydra_mgr.execute_decision_cycle()
        
        # Verify all states were executed in correct order
        expected_states = [
            "INITIALIZATION", "NN_PREDICTION", "TREE_SETUP", 
            "EXPLORATION", "EVALUATION", "ORACLE_TRAINING",
            "MASTER_UPDATE", "CLEANUP", "TERMINATION_CHECK"
        ]
        
        assert state_tracker.states_executed == expected_states
        assert state_tracker.state_durations["EXPLORATION"] > 0
        assert state_tracker.state_durations["NN_PREDICTION"] > 0

**Budget Management Testing**

.. code-block:: python

    @pytest.mark.property
    @given(budget=st.integers(min_value=5, max_value=50))
    @settings(max_examples=10, deadline=5000)
    @timeout_test(120)
    def test_budget_lifecycle_with_round_completion(self, budget):
        """
        **Feature: ai-hydra, Property 5: Budget Lifecycle Management**
        **Validates: Requirements 4.1, 4.2, 6.2**
        
        For any budget value, the system should complete current round
        even if budget is exhausted, then properly reset for next cycle.
        """
        config = SimulationConfig(grid_size=(6, 6), move_budget=budget, random_seed=42)
        hydra_mgr = HydraMgr(config)
        
        # Execute decision cycle
        result = hydra_mgr.execute_decision_cycle()
        
        # Verify budget was properly managed
        assert result.budget_used <= budget + 3  # Allow round completion
        assert result.budget_used > 0
        
        # Verify budget reset for next cycle
        next_budget = hydra_mgr.budget_controller.get_remaining_budget()
        assert next_budget == budget

**Neural Network Integration Testing**

.. code-block:: python

    @pytest.mark.integration
    @timeout_test(90)
    def test_nn_oracle_training_integration(self):
        """Test neural network and oracle trainer integration."""
        config = SimulationConfig(
            grid_size=(6, 6), 
            move_budget=20, 
            nn_enabled=True, 
            random_seed=42
        )
        hydra_mgr = HydraMgr(config)
        
        # Execute multiple decision cycles to test learning
        initial_accuracy = hydra_mgr.oracle_trainer.get_prediction_accuracy()
        
        for _ in range(5):  # Multiple cycles for learning
            result = hydra_mgr.execute_decision_cycle()
            
            # Verify NN prediction was made
            assert result.nn_prediction is not None
            assert result.oracle_comparison is not None
            
            # Verify training occurred if predictions differed
            if result.nn_prediction != result.optimal_move:
                assert result.training_sample_generated
        
        # Verify accuracy tracking
        final_accuracy = hydra_mgr.oracle_trainer.get_prediction_accuracy()
        assert final_accuracy is not None

Performance Tests
~~~~~~~~~~~~~~~~~

Performance tests validate system efficiency and resource usage:

- **Collision Avoidance Improvement** (``test_collision_avoidance_improvement.py``): AI learning metrics

Token Tracking Tests
~~~~~~~~~~~~~~~~~~~~

The token tracking system includes comprehensive testing for data integrity, error handling, and special character support:

**Data Validation Tests:**

.. code-block:: python

    def test_transaction_validation():
        """Test comprehensive transaction data validation."""
        config = TrackerConfig.create_for_testing()
        tracker = TokenTracker(config)
        
        # Test valid transaction
        success = tracker.record_transaction(
            prompt_text="Valid prompt text",
            tokens_used=150,
            elapsed_time=1.5,
            context={"workspace_folder": "test_project"}
        )
        assert success
        
        # Test invalid data handling
        success = tracker.record_transaction(
            prompt_text="",  # Empty prompt should be rejected
            tokens_used=-10,  # Negative tokens should be rejected
            elapsed_time=-1.0,  # Negative time should be rejected
            context={}
        )
        assert not success

**CSV Integrity Tests:**

.. code-block:: python

    def test_csv_file_integrity():
        """Test CSV file creation and validation."""
        config = TrackerConfig.create_for_testing()
        tracker = TokenTracker(config)
        
        # Record multiple transactions
        for i in range(10):
            tracker.record_transaction(
                prompt_text=f"Test prompt {i}",
                tokens_used=100 + i,
                elapsed_time=1.0 + i * 0.1,
                context={"workspace_folder": "test"}
            )
        
        # Validate CSV integrity
        integrity_results = tracker.validate_csv_integrity()
        assert integrity_results["file_exists"]
        assert integrity_results["header_valid"]
        assert integrity_results["valid_rows"] == 10
        assert integrity_results["invalid_rows"] == 0

**Unicode and Special Character Tests:**

.. code-block:: python

    def test_unicode_compatibility():
        """Test Unicode and special character handling."""
        config = TrackerConfig.create_for_testing()
        tracker = TokenTracker(config)
        
        # Test various Unicode and special characters
        test_strings = [
            "Hello, 世界! 🌍",  # Chinese and emoji
            "Café résumé naïve",  # Accented characters
            'Text with "quotes" and commas, semicolons;',  # CSV special chars
            "Text with\nnewlines\rand\ttabs",  # Control characters
            "Math symbols: ∑∏∫∆∇∂∞±≤≥≠≈",  # Mathematical symbols
        ]
        
        unicode_results = tracker.test_unicode_compatibility()
        assert unicode_results["unicode_support_verified"]
        assert unicode_results["special_chars_handled"]
        assert unicode_results["round_trip_successful"]

**Error Recovery Tests:**

.. code-block:: python

    def test_error_recovery():
        """Test system resilience to various error conditions."""
        config = TrackerConfig.create_for_testing()
        tracker = TokenTracker(config)
        
        # Test file permission errors
        with mock.patch('builtins.open', side_effect=PermissionError("Access denied")):
            success = tracker.record_transaction(
                prompt_text="Test during permission error",
                tokens_used=100,
                elapsed_time=1.0,
                context={}
            )
            # Should handle gracefully and continue operation
            assert isinstance(success, bool)
        
        # Test disk space errors
        with mock.patch('pathlib.Path.stat', side_effect=OSError("No space left")):
            success = tracker.record_transaction(
                prompt_text="Test during disk space error",
                tokens_used=100,
                elapsed_time=1.0,
                context={}
            )
            # Should handle gracefully
            assert isinstance(success, bool)

**Configuration Management Tests:**

.. code-block:: python

    @given(
        enabled=st.booleans(),
        max_prompt_length=st.integers(min_value=10, max_value=5000),
        backup_enabled=st.booleans(),
        log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    )
    @settings(max_examples=50, deadline=3000)
    def test_configuration_state_consistency_property(
        self, enabled, max_prompt_length, backup_enabled, log_level
    ):
        """
        **Property 6: Configuration State Management**
        **Validates: Requirements 2.5**
        
        For any valid configuration parameters, the hook should:
        1. Accept the configuration update
        2. Maintain consistent internal state
        3. Reflect changes in subsequent operations
        4. Preserve configuration across enable/disable cycles
        """
        hook = TokenTrackingHook()
        
        try:
            # Create new configuration with test parameters
            new_config = TrackerConfig(
                enabled=enabled,
                max_prompt_length=max_prompt_length,
                backup_enabled=backup_enabled,
                log_level=log_level,
            )
            
            # Configuration update should succeed for valid parameters
            update_success = hook.update_configuration(new_config)
            assert update_success, f"Configuration update failed for valid parameters"
            
            # Hook state should reflect the new configuration
            current_config = hook.get_configuration()
            assert current_config["enabled"] == enabled
            assert current_config["max_prompt_length"] == max_prompt_length
            assert current_config["backup_enabled"] == backup_enabled
            assert current_config["log_level"] == log_level
            
            # Hook enabled state should match configuration enabled state
            assert hook.is_enabled == enabled
            
            # Configuration should be consistent across multiple reads
            config_read_1 = hook.get_configuration()
            config_read_2 = hook.get_configuration()
            assert config_read_1 == config_read_2
            
            # Enable/disable operations should preserve other configuration
            if enabled:
                hook.disable()
                assert not hook.is_enabled
                disabled_config = hook.get_configuration()
                # All other settings should remain the same
                assert disabled_config["max_prompt_length"] == max_prompt_length
                assert disabled_config["backup_enabled"] == backup_enabled
                assert disabled_config["log_level"] == log_level
                
                hook.enable()
                assert hook.is_enabled
                enabled_config = hook.get_configuration()
                # All other settings should still be preserved
                assert enabled_config["max_prompt_length"] == max_prompt_length
                assert enabled_config["backup_enabled"] == backup_enabled
                assert enabled_config["log_level"] == log_level
            
            # Configuration validation should work correctly
            validation_result = hook.validate_configuration()
            assert isinstance(validation_result, dict)
            assert "valid" in validation_result
            assert "issues" in validation_result
            assert isinstance(validation_result["issues"], list)
            
        finally:
            hook.cleanup()

The configuration management tests validate that the TokenTrackingHook can:

* **Accept valid configuration updates** at runtime without requiring system restart
* **Maintain consistent internal state** across all configuration operations
* **Preserve configuration values** during enable/disable cycles
* **Handle partial configuration changes** while preserving unchanged values
* **Support file persistence** for configuration save/load operations
* **Validate configuration** and provide meaningful error messages
* **Handle invalid configurations** gracefully without corrupting system state
- **Parallel NN Lifecycle** (``test_parallel_nn_lifecycle.py``): Resource management efficiency

Documentation Tests
~~~~~~~~~~~~~~~~~~~

Documentation tests validate the integrity and accuracy of project documentation:

- **Documentation Test Suite** (``test_documentation.py``): Comprehensive documentation validation

Debug and Diagnostic Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project includes specialized debug scripts for testing and validation:

- **Token Tracker Debug Script** (``debug_file_patterns.py``): Tests file patterns preservation in metadata collection

**Token Tracker Debug Script**

The ``debug_file_patterns.py`` script provides comprehensive testing of the token tracking system's metadata collection chain:

.. code-block:: bash

    # Run the debug script
    python debug_file_patterns.py

**What it tests:**

1. **MetadataCollector.get_hook_context()**: Verifies that file patterns are correctly extracted from hook context
2. **MetadataCollector.collect_execution_metadata()**: Tests full metadata collection including file patterns
3. **TokenTracker integration**: Validates end-to-end file patterns preservation through transaction recording

**Example output:**

.. code-block:: text

    Testing file patterns preservation...
    Input context: {'trigger_type': 'agentExecutionCompleted', 'hook_name': 'test-hook', 'file_patterns': ['*.py', '*.md', '*.txt']}

    1. Testing MetadataCollector.get_hook_context()...
    Hook context result: {'trigger_type': 'agentExecutionCompleted', 'hook_name': 'test-hook', 'file_patterns': ['*.py', '*.md', '*.txt']}
    ✓ file_patterns found: ['*.py', '*.md', '*.txt']
    ✓ file_patterns match input

    2. Testing MetadataCollector.collect_execution_metadata()...
    Full metadata keys: ['timestamp', 'session_id', 'agent_execution_id', 'workspace_folder', 'hook_trigger_type', 'hook_name', 'file_patterns', ...]
    ✓ file_patterns found in metadata: ['*.py', '*.md', '*.txt']
    ✓ file_patterns match input in full metadata

    3. Testing TokenTracker integration...
    Transaction recording result: True
    Transaction file_patterns: ['*.py', '*.md', '*.txt']
    ✓ file_patterns preserved in transaction

    ✓ All tests passed!

**Usage in development:**

.. code-block:: bash

    # Test file patterns preservation during development
    python debug_file_patterns.py
    
    # Use in CI/CD pipeline for validation
    python debug_file_patterns.py && echo "File patterns test passed"

The documentation test suite provides automated validation of:

**RST Syntax Validation**
  Validates that all RST files have correct syntax using docutils parsing

**Cross-Reference Validation**
  Ensures all ``:ref:`` references point to valid labels within the documentation

**Content Accuracy Testing**
  Verifies that documentation content matches the actual system architecture and requirements

**Build Integrity Testing**
  Tests that Sphinx can successfully build HTML documentation without critical errors

**Internal Link Validation**
  Checks that all internal links within generated HTML documentation are valid

**Code Block Validation**
  Validates that Python code blocks in documentation have correct syntax

**Table of Contents Structure**
  Ensures the main index.rst has a valid toctree structure with existing files

**Decision Flow Content Validation**
  Specifically validates that decision flow documentation contains required sections and technical details

Example usage:

.. code-block:: bash

    # Run documentation tests
    python test_documentation.py
    
    # Run with custom docs directory
    python test_documentation.py /path/to/docs
    
    # Integration with pytest
    pytest test_documentation.py -v

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

Run Documentation Tests
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Run documentation validation tests
    python test_documentation.py
    
    # Run with verbose output
    python test_documentation.py docs
    
    # Integration with pytest (if test is in tests/ directory)
    pytest test_documentation.py -v

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

Integration tests validate component interactions and the enhanced decision flow:

.. code-block:: python

    @timeout_test(120)  # 2 minute timeout for decision cycle testing
    def test_complete_decision_cycle_with_nn_integration(self):
        """Test complete decision cycle with neural network integration."""
        sim_config = SimulationConfig(
            grid_size=(6, 6),
            move_budget=10,  # Reduced for testing
            nn_enabled=True,
            random_seed=42
        )
        
        hydra_mgr = HydraMgr(sim_config)
        
        # Test all 9 decision cycle states
        initial_score = hydra_mgr.master_game.get_score()
        
        # Execute one complete decision cycle
        decision_result = hydra_mgr.execute_decision_cycle()
        
        # Verify decision cycle completed all states
        assert decision_result.move in [Move.LEFT_TURN, Move.STRAIGHT, Move.RIGHT_TURN]
        assert decision_result.budget_used > 0
        assert decision_result.paths_evaluated > 0
        assert decision_result.nn_prediction is not None
        assert decision_result.oracle_comparison is not None
        
        # Verify master game state updated
        final_score = hydra_mgr.master_game.get_score()
        assert final_score >= initial_score  # Score should not decrease

    @timeout_test(300)  # 5 minute timeout for complete simulation
    def test_complete_game_simulation_start_to_finish(self):
        """Test complete game simulations from start to finish."""
        sim_config = SimulationConfig(
            grid_size=(6, 6),
            move_budget=10,  # Reduced for testing
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        result = pipeline.run_complete_simulation()
        
        # Verify simulation completed properly
        assert result.success
        assert result.game_result.total_moves > 0
        assert result.game_result.decision_cycles > 0
        
        # Verify all decision cycle states were executed
        assert result.game_result.nn_predictions > 0
        assert result.game_result.oracle_comparisons > 0
        assert result.game_result.budget_resets > 0

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
    
    # Run documentation tests in CI
    python test_documentation.py

Full Test Suite
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run complete test suite (for nightly builds)
    pytest --cov=ai_hydra --cov-report=xml --timeout=1800
    
    # Include documentation validation
    python test_documentation.py

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