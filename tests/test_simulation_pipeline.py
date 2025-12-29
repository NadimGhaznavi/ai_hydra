"""
Tests for the complete simulation pipeline integration.

This module tests the end-to-end integration of all components including
neural network, tree search, oracle training, and comprehensive logging.
"""

import pytest
import torch
from ai_hydra.simulation_pipeline import SimulationPipeline, PipelineResult
from ai_hydra.config import SimulationConfig, NetworkConfig, LoggingConfig


class TestSimulationPipeline:
    """Test suite for the complete simulation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test that the pipeline initializes all components correctly."""
        # Create configuration
        sim_config = SimulationConfig(
            grid_size=(10, 10),
            move_budget=50,
            nn_enabled=True,
            random_seed=42
        )
        
        # Create pipeline
        pipeline = SimulationPipeline(sim_config)
        
        # Initialize pipeline
        success = pipeline.initialize_pipeline()
        
        assert success, "Pipeline initialization should succeed"
        assert pipeline.is_initialized, "Pipeline should be marked as initialized"
        
        # Verify components are initialized
        status = pipeline.get_pipeline_status()
        assert status["initialized"], "Pipeline status should show initialized"
        assert status["ready_for_execution"], "Pipeline should be ready for execution"
        
        expected_components = ["HydraMgr", "MasterGame", "BudgetController", "StateManager"]
        if sim_config.nn_enabled:
            expected_components.extend(["SnakeNet", "FeatureExtractor", "OracleTrainer"])
        
        for component in expected_components:
            assert component in status["components"], f"Component {component} should be initialized"
    
    def test_pipeline_initialization_without_nn(self):
        """Test pipeline initialization with neural network disabled."""
        sim_config = SimulationConfig(
            grid_size=(10, 10),
            move_budget=30,
            nn_enabled=False,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        success = pipeline.initialize_pipeline()
        
        assert success, "Pipeline initialization should succeed without NN"
        
        status = pipeline.get_pipeline_status()
        expected_components = ["HydraMgr", "MasterGame", "BudgetController", "StateManager"]
        
        for component in expected_components:
            assert component in status["components"], f"Component {component} should be initialized"
        
        # NN components should not be present
        nn_components = ["SnakeNet", "FeatureExtractor", "OracleTrainer"]
        for component in nn_components:
            assert component not in status["components"], f"NN component {component} should not be initialized"
    
    def test_complete_simulation_execution(self):
        """Test running a complete simulation from start to finish."""
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=20,  # Small budget for quick test
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        pipeline.initialize_pipeline()
        
        # Run complete simulation
        result = pipeline.run_complete_simulation()
        
        assert isinstance(result, PipelineResult), "Should return PipelineResult"
        assert result.success, f"Simulation should succeed, but got error: {result.error_message}"
        assert result.game_result is not None, "Game result should not be None"
        assert result.execution_time > 0, "Execution time should be positive"
        
        # Verify game result structure
        game_result = result.game_result
        assert game_result.final_score >= 0, "Final score should be non-negative"
        assert game_result.total_moves > 0, "Should have made at least one move"
        assert game_result.game_length_seconds > 0, "Game should have taken some time"
        
        # Verify pipeline metrics
        assert "execution_time" in result.pipeline_metrics
        assert "moves_per_second" in result.pipeline_metrics
        assert "decision_cycles" in result.pipeline_metrics
        assert "pipeline_efficiency" in result.pipeline_metrics
        
        # Verify component statistics
        assert "master_game" in result.component_statistics
        assert "budget_controller" in result.component_statistics
        assert "state_manager" in result.component_statistics
        
        if sim_config.nn_enabled:
            assert "neural_network" in result.component_statistics
            assert "nn_accuracy" in result.pipeline_metrics
            assert "nn_training_updates" in result.pipeline_metrics
    
    def test_simulation_without_neural_network(self):
        """Test simulation execution with neural network disabled."""
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=15,
            nn_enabled=False,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        pipeline.initialize_pipeline()
        
        result = pipeline.run_complete_simulation()
        
        assert result.success, "Simulation without NN should succeed"
        assert result.game_result is not None, "Game result should not be None"
        
        # NN-specific metrics should not be present
        assert "nn_accuracy" not in result.pipeline_metrics
        assert "nn_training_updates" not in result.pipeline_metrics
        assert "neural_network" not in result.component_statistics
    
    def test_multiple_simulations(self):
        """Test running multiple simulations in sequence."""
        sim_config = SimulationConfig(
            grid_size=(6, 6),
            move_budget=10,  # Very small for quick tests
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        pipeline.initialize_pipeline()
        
        # Run multiple simulations
        results = pipeline.run_multiple_simulations(3)
        
        assert len(results) == 3, "Should return results for all simulations"
        
        for i, result in enumerate(results):
            assert result.success, f"Simulation {i+1} should succeed"
            assert result.game_result is not None, f"Game result {i+1} should not be None"
            assert result.execution_time > 0, f"Execution time {i+1} should be positive"
        
        # Verify that simulations are independent (different results)
        scores = [r.game_result.final_score for r in results]
        moves = [r.game_result.total_moves for r in results]
        
        # At least some variation expected (though not guaranteed with small budgets)
        assert len(set(scores + moves)) > 1, "Should have some variation across simulations"
    
    def test_pipeline_component_integration(self):
        """Test that all components are properly integrated and communicate."""
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=25,
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        pipeline.initialize_pipeline()
        
        # Verify component integration before execution
        assert pipeline._verify_component_integration(), "Components should be properly integrated"
        
        # Run simulation and verify component interactions
        result = pipeline.run_complete_simulation()
        
        assert result.success, "Integrated simulation should succeed"
        
        # Verify that all components contributed to the result
        stats = result.component_statistics
        
        # Master game should have tracked moves and score
        master_stats = stats["master_game"]
        assert master_stats["total_moves"] > 0, "Master game should have tracked moves"
        assert master_stats["current_score"] >= 0, "Master game should have valid score"
        
        # Budget controller should have utilization data
        budget_stats = stats["budget_controller"]
        assert "current_utilization_rate" in budget_stats, "Budget controller should track utilization"
        
        # State manager should have tree statistics
        state_stats = stats["state_manager"]
        assert "tree_stats" in state_stats, "State manager should have tree statistics"
        assert state_stats["tree_stats"]["total_clones_created"] > 0, "Should have created clones"
        
        # Neural network should have training statistics (if enabled)
        if sim_config.nn_enabled:
            nn_stats = stats["neural_network"]
            assert "training_stats" in nn_stats, "Should have training statistics"
            assert "network_info" in nn_stats, "Should have network information"
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling for invalid configurations."""
        # Test with invalid grid size
        invalid_config = SimulationConfig(
            grid_size=(2, 2),  # Too small
            move_budget=10,
            nn_enabled=False,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(invalid_config)
        
        # Initialization might fail or succeed depending on validation
        # If it succeeds, simulation should handle the constraint gracefully
        if pipeline.initialize_pipeline():
            result = pipeline.run_complete_simulation()
            # Should either succeed with constraints or fail gracefully
            assert isinstance(result, PipelineResult), "Should return PipelineResult even on failure"
    
    def test_pipeline_shutdown(self):
        """Test graceful pipeline shutdown."""
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=15,
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        pipeline.initialize_pipeline()
        
        # Verify pipeline is ready
        assert pipeline.is_initialized, "Pipeline should be initialized"
        
        # Shutdown pipeline
        pipeline.shutdown_pipeline()
        
        # Verify shutdown state
        assert not pipeline.is_initialized, "Pipeline should not be initialized after shutdown"
        assert pipeline.execution_count == 0, "Execution count should be reset"
        
        status = pipeline.get_pipeline_status()
        assert not status["initialized"], "Status should show not initialized"
        assert not status["ready_for_execution"], "Should not be ready for execution"
    
    def test_pipeline_status_reporting(self):
        """Test comprehensive pipeline status reporting."""
        sim_config = SimulationConfig(
            grid_size=(10, 10),
            move_budget=30,
            nn_enabled=True,
            random_seed=42
        )
        
        pipeline = SimulationPipeline(sim_config)
        
        # Check status before initialization
        status_before = pipeline.get_pipeline_status()
        assert not status_before["initialized"], "Should not be initialized initially"
        assert status_before["execution_count"] == 0, "Execution count should be 0"
        assert not status_before["ready_for_execution"], "Should not be ready initially"
        
        # Initialize and check status
        pipeline.initialize_pipeline()
        status_after = pipeline.get_pipeline_status()
        
        assert status_after["initialized"], "Should be initialized after init"
        assert status_after["ready_for_execution"], "Should be ready after init"
        assert len(status_after["components"]) > 0, "Should have initialized components"
        
        # Verify configuration is reported correctly
        config = status_after["configuration"]
        assert config["grid_size"] == sim_config.grid_size
        assert config["move_budget"] == sim_config.move_budget
        assert config["nn_enabled"] == sim_config.nn_enabled
        assert config["random_seed"] == sim_config.random_seed
        
        # Run simulation and check execution count
        pipeline.run_complete_simulation()
        status_after_run = pipeline.get_pipeline_status()
        assert status_after_run["execution_count"] == 1, "Execution count should increment"


class TestPipelineConfiguration:
    """Test suite for pipeline configuration handling."""
    
    def test_default_configuration(self):
        """Test pipeline with default configuration."""
        sim_config = SimulationConfig()  # Use defaults
        pipeline = SimulationPipeline(sim_config)
        
        success = pipeline.initialize_pipeline()
        assert success, "Should initialize with default configuration"
        
        status = pipeline.get_pipeline_status()
        config = status["configuration"]
        
        # Verify default values
        assert config["grid_size"] == (20, 20), "Default grid size should be 20x20"
        assert config["move_budget"] == 100, "Default move budget should be 100"
        assert config["nn_enabled"] == True, "Neural network should be enabled by default"
    
    def test_custom_network_configuration(self):
        """Test pipeline with custom network configuration."""
        sim_config = SimulationConfig(nn_enabled=True)
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(100, 100),
            output_actions=3,
            learning_rate=0.01,
            batch_size=16
        )
        
        pipeline = SimulationPipeline(sim_config, net_config)
        pipeline.initialize_pipeline()
        
        # Run a quick simulation to verify network configuration is used
        result = pipeline.run_complete_simulation()
        assert result.success, "Should succeed with custom network config"
        
        # Verify network configuration is applied
        nn_stats = result.component_statistics.get("neural_network", {})
        if nn_stats:
            network_info = nn_stats.get("network_info", {})
            assert network_info.get("input_features") == 19, "Should use custom input features"
    
    def test_custom_logging_configuration(self):
        """Test pipeline with custom logging configuration."""
        sim_config = SimulationConfig(move_budget=20)
        log_config = LoggingConfig(
            level="DEBUG",
            log_clone_steps=True,
            log_decision_cycles=True,
            log_neural_network=True
        )
        
        pipeline = SimulationPipeline(sim_config, logging_config=log_config)
        success = pipeline.initialize_pipeline()
        
        assert success, "Should initialize with custom logging config"
        
        # Run simulation to verify logging configuration is used
        result = pipeline.run_complete_simulation()
        assert result.success, "Should succeed with custom logging config"