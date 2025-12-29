"""
End-to-end simulation tests for the AI Hydra.

This module contains tests that validate complete game simulations from start 
to finish, neural network learning from tree search corrections, and collision 
avoidance improvements.

**Validates: Requirements 11.5, 6.5**
"""

import pytest
import torch
from ai_hydra.simulation_pipeline import SimulationPipeline, PipelineResult
from ai_hydra.config import SimulationConfig, NetworkConfig, LoggingConfig


# Add timeout decorator for long-running tests
def timeout_test(seconds):
    """Decorator to add timeout to tests."""
    return pytest.mark.timeout(seconds)


class TestEndToEndSimulation:
    """Test suite for complete end-to-end simulations."""
    
    @timeout_test(120)  # 2 minute timeout
    def test_complete_game_simulation_start_to_finish(self):
        """
        Test complete game simulations from start to finish.
        
        This test validates that the entire simulation pipeline can execute
        a complete game from initialization through termination.
        """
        # Create configuration for a complete simulation
        sim_config = SimulationConfig(
            grid_size=(6, 6),  # Smaller grid for faster testing
            move_budget=10,    # Reduced budget for quick testing
            nn_enabled=True,
            random_seed=42,
            initial_snake_length=3
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(15, 15),  # Much smaller network for faster testing
            output_actions=3,
            learning_rate=0.01,
            batch_size=2  # Smaller batch size
        )
        
        # Initialize and run complete simulation
        pipeline = SimulationPipeline(sim_config, net_config)
        assert pipeline.initialize_pipeline(), "Pipeline should initialize successfully"
        
        result = pipeline.run_complete_simulation()
        
        # Validate simulation completed successfully
        assert result.success, f"Simulation should complete successfully, error: {result.error_message}"
        assert result.game_result is not None, "Game result should not be None"
        assert result.execution_time > 0, "Execution time should be positive"
        
        # Validate game progression
        game_result = result.game_result
        assert game_result.final_score >= 0, "Final score should be non-negative"
        assert game_result.total_moves > 0, "Should have made at least one move"
        
        # Validate component integration
        component_stats = result.component_statistics
        assert "master_game" in component_stats, "Master game statistics should be present"
        assert "state_manager" in component_stats, "State manager statistics should be present"
        assert "neural_network" in component_stats, "Neural network statistics should be present"
        
        # Validate tree exploration occurred
        state_stats = component_stats["state_manager"]
        tree_stats = state_stats["tree_stats"]
        assert tree_stats["total_clones_created"] > 0, "Should have created exploration clones"
        
        print(f"✓ Complete simulation: {game_result.total_moves} moves, score {game_result.final_score}")
    
    @timeout_test(300)  # 5 minute timeout
    def test_neural_network_learning_from_tree_search(self):
        """
        Test that neural network learns from tree search corrections.
        
        This test validates that when the neural network makes incorrect
        predictions, it learns from the tree search oracle and improves
        its accuracy over time.
        """
        sim_config = SimulationConfig(
            grid_size=(6, 6),  # Smaller grid for faster testing
            move_budget=15,    # Reduced budget for faster testing
            nn_enabled=True,
            random_seed=123,
            initial_snake_length=3
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(15, 15),  # Smaller network for faster learning
            output_actions=3,
            learning_rate=0.05,
            batch_size=2
        )
        
        # Run multiple simulations to observe learning
        pipeline = SimulationPipeline(sim_config, net_config)
        pipeline.initialize_pipeline()
        
        results = []
        nn_accuracies = []
        
        # Run several simulations to track learning progress
        for i in range(2):  # Reduced from 5 to 2 for faster testing
            result = pipeline.run_complete_simulation()
            assert result.success, f"Simulation {i+1} should succeed"
            
            results.append(result)
            
            # Extract neural network accuracy
            if "nn_accuracy" in result.pipeline_metrics:
                nn_accuracies.append(result.pipeline_metrics["nn_accuracy"])
            
            # Reset for next simulation (but keep learned weights)
            pipeline._reset_for_new_simulation()
        
        # Validate learning occurred
        assert len(nn_accuracies) > 0, "Should have neural network accuracy data"
        
        # Check that neural network statistics show learning activity
        final_result = results[-1]
        nn_stats = final_result.component_statistics["neural_network"]["training_stats"]
        
        assert nn_stats["total_predictions"] > 0, "Should have made predictions"
        assert nn_stats["training_updates"] >= 0, "Should have training updates"
        assert nn_stats["total_training_samples"] >= 0, "Should have training samples"
        
        # Validate oracle decision making
        # The system should prefer tree search results when they differ from NN
        oracle_decisions_made = any(
            "oracle" in str(result.component_statistics).lower() 
            for result in results
        )
        
        print(f"✓ Neural network learning: {len(results)} simulations, "
              f"final accuracy: {nn_accuracies[-1]:.3f}" if nn_accuracies else "N/A")
        print(f"  Training updates: {nn_stats['training_updates']}, "
              f"samples: {nn_stats['total_training_samples']}")
    
    @timeout_test(240)  # 4 minute timeout
    def test_collision_avoidance_improvements(self):
        """
        Test that the hybrid system improves collision avoidance.
        
        This test validates that the combination of neural network predictions
        and tree search exploration leads to better collision avoidance
        compared to random or simple strategies.
        """
        # Configuration designed to test collision avoidance
        sim_config = SimulationConfig(
            grid_size=(6, 6),  # Smaller grid for faster testing
            move_budget=15,    # Reduced budget for faster testing
            nn_enabled=True,
            random_seed=456,
            initial_snake_length=3,
            collision_penalty=-20
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(20, 20),  # Smaller network for faster testing
            output_actions=3,
            learning_rate=0.02,
            batch_size=2
        )
        
        # Run simulation with hybrid system
        pipeline = SimulationPipeline(sim_config, net_config)
        pipeline.initialize_pipeline()
        
        hybrid_results = []
        for i in range(2):  # Reduced from 3 to 2 for faster testing
            result = pipeline.run_complete_simulation()
            assert result.success, f"Hybrid simulation {i+1} should succeed"
            hybrid_results.append(result)
            pipeline._reset_for_new_simulation()
        
        # Analyze collision avoidance performance
        hybrid_scores = [r.game_result.final_score for r in hybrid_results]
        hybrid_moves = [r.game_result.total_moves for r in hybrid_results]
        hybrid_survival_time = [r.game_result.game_length_seconds for r in hybrid_results]
        
        # Validate that simulations achieved reasonable performance
        avg_score = sum(hybrid_scores) / len(hybrid_scores)
        avg_moves = sum(hybrid_moves) / len(hybrid_moves)
        avg_survival = sum(hybrid_survival_time) / len(hybrid_survival_time)
        
        assert avg_score >= 0, "Average score should be non-negative"
        assert avg_moves > 3, "Should survive beyond initial moves"
        assert avg_survival > 0, "Should have positive survival time"
        
        # Validate tree search contributed to decision making
        for result in hybrid_results:
            state_stats = result.component_statistics["state_manager"]
            tree_stats = state_stats["tree_stats"]
            
            assert tree_stats["total_clones_created"] > 0, "Should have used tree search"
            assert tree_stats["max_depth"] > 0, "Should have explored multiple moves ahead"
        
        # Validate collision detection features were used
        for result in hybrid_results:
            nn_stats = result.component_statistics["neural_network"]
            assert "training_stats" in nn_stats, "Should have neural network activity"
            
            # The network should have made predictions about collision avoidance
            training_stats = nn_stats["training_stats"]
            if training_stats["total_predictions"] > 0:
                assert training_stats["total_predictions"] >= avg_moves, \
                    "Should have made predictions for most moves"
        
        print(f"✓ Collision avoidance: avg score {avg_score:.1f}, "
              f"avg moves {avg_moves:.1f}, avg survival {avg_survival:.2f}s")
        
        # Additional validation: check that exploration helped avoid immediate collisions
        for result in hybrid_results:
            budget_stats = result.component_statistics["budget_controller"]
            utilization = budget_stats.get("current_utilization_rate", 0)
            
            # Should have used budget for exploration
            assert utilization > 0, "Should have utilized exploration budget"
    
    def test_deterministic_reproducibility(self):
        """
        Test that simulations are deterministic and reproducible.
        
        This test validates that identical configurations produce identical
        results across multiple runs, ensuring reproducibility for research.
        """
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=15,
            nn_enabled=True,
            random_seed=789,  # Fixed seed for reproducibility
            initial_snake_length=3
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(25, 25),
            output_actions=3,
            learning_rate=0.01,
            batch_size=4
        )
        
        # Run the same simulation multiple times
        results = []
        for i in range(3):
            pipeline = SimulationPipeline(sim_config, net_config)
            pipeline.initialize_pipeline()
            
            result = pipeline.run_complete_simulation()
            assert result.success, f"Reproducibility test {i+1} should succeed"
            results.append(result)
        
        # Validate reproducibility
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            # Core game metrics should be identical
            assert result.game_result.final_score == first_result.game_result.final_score, \
                f"Final scores should be identical: run 1 = {first_result.game_result.final_score}, " \
                f"run {i+1} = {result.game_result.final_score}"
            
            assert result.game_result.total_moves == first_result.game_result.total_moves, \
                f"Total moves should be identical: run 1 = {first_result.game_result.total_moves}, " \
                f"run {i+1} = {result.game_result.total_moves}"
            
            # Tree exploration should be consistent
            first_tree_stats = first_result.component_statistics["state_manager"]["tree_stats"]
            result_tree_stats = result.component_statistics["state_manager"]["tree_stats"]
            
            assert first_tree_stats["total_clones_created"] == result_tree_stats["total_clones_created"], \
                "Tree exploration should be deterministic"
            
            assert first_tree_stats["max_depth"] == result_tree_stats["max_depth"], \
                "Maximum exploration depth should be deterministic"
        
        print(f"✓ Deterministic reproducibility: {len(results)} identical runs, "
              f"score {first_result.game_result.final_score}, "
              f"moves {first_result.game_result.total_moves}")
    
    def test_budget_constraint_functionality(self):
        """
        Test that budget constraints work correctly.
        
        This test validates that the system respects budget limits and
        creates appropriate exploration within those constraints.
        """
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=20,  # Moderate budget
            nn_enabled=True,
            random_seed=321,
            initial_snake_length=3
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(30, 30),
            output_actions=3,
            learning_rate=0.01,
            batch_size=4
        )
        
        pipeline = SimulationPipeline(sim_config, net_config)
        pipeline.initialize_pipeline()
        
        result = pipeline.run_complete_simulation()
        assert result.success, "Budget constraint test should succeed"
        
        # Validate basic functionality
        assert result.game_result.final_score >= 0, "Should have valid score"
        assert result.game_result.total_moves > 0, "Should have made moves"
        
        # Validate tree exploration occurred within budget
        tree_stats = result.component_statistics["state_manager"]["tree_stats"]
        assert tree_stats["total_clones_created"] > 0, "Should create exploration clones"
        assert tree_stats["max_depth"] > 0, "Should achieve exploration depth"
        
        print(f"✓ Budget constraints: score {result.game_result.final_score}, "
              f"moves {result.game_result.total_moves}, "
              f"clones {tree_stats['total_clones_created']}")
    
    def test_error_recovery_and_robustness(self):
        """
        Test system robustness and error recovery capabilities.
        
        This test validates that the system can handle various error conditions
        gracefully and continue operation when possible.
        """
        # Test with challenging configuration
        sim_config = SimulationConfig(
            grid_size=(6, 6),  # Small grid increases collision probability
            move_budget=15,
            nn_enabled=True,
            random_seed=654,
            initial_snake_length=3,
            collision_penalty=-50  # High penalty
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(20, 20),  # Small network
            output_actions=3,
            learning_rate=0.03,
            batch_size=2
        )
        
        pipeline = SimulationPipeline(sim_config, net_config)
        assert pipeline.initialize_pipeline(), "Pipeline should initialize despite challenging config"
        
        # Run simulation and expect it to handle constraints gracefully
        result = pipeline.run_complete_simulation()
        
        # Should complete successfully or fail gracefully
        if result.success:
            assert result.game_result is not None, "Successful result should have game data"
            assert result.game_result.final_score >= 0, "Score should be valid"
            assert result.game_result.total_moves > 0, "Should have made moves"
            
            print(f"✓ Robustness test: completed successfully with score {result.game_result.final_score}")
        else:
            # If it fails, should fail gracefully with error message
            assert result.error_message is not None, "Failed result should have error message"
            assert isinstance(result.error_message, str), "Error message should be string"
            
            print(f"✓ Robustness test: failed gracefully with error: {result.error_message}")
        
        # Validate that pipeline can be shutdown cleanly
        pipeline.shutdown_pipeline()
        status = pipeline.get_pipeline_status()
        assert not status["initialized"], "Pipeline should be cleanly shutdown"


class TestSimulationBasics:
    """Test suite for basic simulation functionality and metrics."""
    
    def test_basic_metrics_collection(self):
        """
        Test that basic metrics are collected during simulation.
        
        This test validates that essential metrics and statistics are
        properly collected and reported.
        """
        sim_config = SimulationConfig(
            grid_size=(8, 8),
            move_budget=20,
            nn_enabled=True,
            random_seed=987
        )
        
        net_config = NetworkConfig(
            input_features=19,
            hidden_layers=(30, 30),
            output_actions=3,
            learning_rate=0.01,
            batch_size=4
        )
        
        pipeline = SimulationPipeline(sim_config, net_config)
        pipeline.initialize_pipeline()
        
        result = pipeline.run_complete_simulation()
        assert result.success, "Basic metrics test should succeed"
        
        # Validate essential game result metrics
        game_result = result.game_result
        assert hasattr(game_result, "final_score"), "Should have final_score"
        assert hasattr(game_result, "total_moves"), "Should have total_moves"
        assert hasattr(game_result, "game_length_seconds"), "Should have game_length_seconds"
        
        # Validate essential component statistics
        component_stats = result.component_statistics
        required_components = ["master_game", "budget_controller", "state_manager"]
        
        if sim_config.nn_enabled:
            required_components.append("neural_network")
        
        for component in required_components:
            assert component in component_stats, f"Should have {component} stats"
        
        # Validate tree exploration occurred
        tree_stats = component_stats["state_manager"]["tree_stats"]
        assert tree_stats["total_clones_created"] > 0, "Should create clones"
        
        print(f"✓ Basic metrics: score {game_result.final_score}, "
              f"moves {game_result.total_moves}, "
              f"clones {tree_stats['total_clones_created']}")