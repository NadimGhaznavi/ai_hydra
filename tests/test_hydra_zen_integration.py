"""
Tests for Hydra Zen configuration integration.

This module tests the enhanced configuration system with Hydra Zen support,
including structured configs, validation, inheritance, and reproducibility.
"""

import pytest
import tempfile
import json
from pathlib import Path
from dataclasses import asdict

from ai_hydra.config import (
    SimulationConfig, NetworkConfig, LoggingConfig, 
    ExperimentConfig, ReproducibilityConfig,
    ConfigValidator, ConfigurationManager
)


class TestConfigurationValidation:
    """Test suite for configuration validation."""
    
    def test_valid_simulation_config(self):
        """Test validation of valid simulation configuration."""
        config = SimulationConfig(
            grid_size=(15, 15),
            move_budget=50,
            nn_enabled=True,
            random_seed=123
        )
        
        # Should not raise any exception
        ConfigValidator.validate_simulation_config(config)
    
    def test_invalid_grid_size(self):
        """Test validation fails for invalid grid size."""
        # Too small
        config = SimulationConfig(grid_size=(3, 3))
        with pytest.raises(ValueError, match="Grid size must be at least 5x5"):
            ConfigValidator.validate_simulation_config(config)
        
        # Too large
        config = SimulationConfig(grid_size=(150, 150))
        with pytest.raises(ValueError, match="Grid size cannot exceed 100x100"):
            ConfigValidator.validate_simulation_config(config)
        
        # Wrong type
        config = SimulationConfig(grid_size=(10,))  # Only one dimension
        with pytest.raises(ValueError, match="grid_size must be a tuple of two integers"):
            ConfigValidator.validate_simulation_config(config)
    
    def test_invalid_move_budget(self):
        """Test validation fails for invalid move budget."""
        # Too small
        config = SimulationConfig(move_budget=0)
        with pytest.raises(ValueError, match="Move budget must be at least 1"):
            ConfigValidator.validate_simulation_config(config)
        
        # Too large
        config = SimulationConfig(move_budget=20000)
        with pytest.raises(ValueError, match="Move budget cannot exceed 10000"):
            ConfigValidator.validate_simulation_config(config)
    
    def test_invalid_snake_length(self):
        """Test validation fails for invalid snake length."""
        # Too small
        config = SimulationConfig(initial_snake_length=0)
        with pytest.raises(ValueError, match="Initial snake length must be at least 1"):
            ConfigValidator.validate_simulation_config(config)
        
        # Too large for grid
        config = SimulationConfig(grid_size=(8, 8), initial_snake_length=7)
        with pytest.raises(ValueError, match="Initial snake length too large for grid size"):
            ConfigValidator.validate_simulation_config(config)
    
    def test_invalid_reward_values(self):
        """Test validation fails for invalid reward values."""
        # Non-positive food reward
        config = SimulationConfig(food_reward=0)
        with pytest.raises(ValueError, match="Food reward must be positive"):
            ConfigValidator.validate_simulation_config(config)
        
        # Non-negative collision penalty
        config = SimulationConfig(collision_penalty=5)
        with pytest.raises(ValueError, match="Collision penalty must be negative"):
            ConfigValidator.validate_simulation_config(config)
    
    def test_valid_network_config(self):
        """Test validation of valid network configuration."""
        config = NetworkConfig(
            input_features=19,
            hidden_layers=(100, 150, 100),
            output_actions=3,
            learning_rate=0.01,
            batch_size=16
        )
        
        # Should not raise any exception
        ConfigValidator.validate_network_config(config)
    
    def test_invalid_network_config(self):
        """Test validation fails for invalid network configuration."""
        # Invalid input features
        config = NetworkConfig(input_features=0)
        with pytest.raises(ValueError, match="Input features must be at least 1"):
            ConfigValidator.validate_network_config(config)
        
        # Invalid hidden layers
        config = NetworkConfig(hidden_layers=())
        with pytest.raises(ValueError, match="At least one hidden layer is required"):
            ConfigValidator.validate_network_config(config)
        
        # Invalid learning rate
        config = NetworkConfig(learning_rate=1.5)
        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1"):
            ConfigValidator.validate_network_config(config)
    
    def test_valid_experiment_config(self):
        """Test validation of valid experiment configuration."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            num_simulations=5,
            parallel_execution=True,
            max_workers=2
        )
        
        # Should not raise any exception
        ConfigValidator.validate_experiment_config(config)
    
    def test_invalid_experiment_config(self):
        """Test validation fails for invalid experiment configuration."""
        # Invalid number of simulations
        config = ExperimentConfig(num_simulations=0)
        with pytest.raises(ValueError, match="Number of simulations must be at least 1"):
            ConfigValidator.validate_experiment_config(config)
        
        # Too many simulations
        config = ExperimentConfig(num_simulations=15000)
        with pytest.raises(ValueError, match="Number of simulations cannot exceed 10000"):
            ConfigValidator.validate_experiment_config(config)
        
        # Invalid max workers
        config = ExperimentConfig(parallel_execution=True, max_workers=0)
        with pytest.raises(ValueError, match="Max workers must be at least 1"):
            ConfigValidator.validate_experiment_config(config)
    
    def test_valid_reproducibility_config(self):
        """Test validation of valid reproducibility configuration."""
        config = ReproducibilityConfig(
            master_seed=12345,
            use_deterministic_algorithms=True,
            seed_sequence=[1, 2, 3, 4, 5]
        )
        
        # Should not raise any exception
        ConfigValidator.validate_reproducibility_config(config)
    
    def test_invalid_reproducibility_config(self):
        """Test validation fails for invalid reproducibility configuration."""
        # Invalid master seed
        config = ReproducibilityConfig(master_seed=-1)
        with pytest.raises(ValueError, match="Master seed must be between 0 and 2\\^31-1"):
            ConfigValidator.validate_reproducibility_config(config)
        
        # Invalid seed in sequence
        config = ReproducibilityConfig(seed_sequence=[1, 2, -1, 4])
        with pytest.raises(ValueError, match="All seeds in sequence must be between 0 and 2\\^31-1"):
            ConfigValidator.validate_reproducibility_config(config)


class TestConfigurationManager:
    """Test suite for configuration management functionality."""
    
    def test_create_game_config_from_simulation(self):
        """Test conversion from SimulationConfig to GameConfig."""
        sim_config = SimulationConfig(
            grid_size=(15, 15),
            move_budget=75,
            random_seed=999,
            food_reward=15,
            collision_penalty=-15
        )
        
        game_config = ConfigurationManager.create_game_config_from_simulation(sim_config)
        
        assert game_config.grid_size == sim_config.grid_size
        assert game_config.move_budget == sim_config.move_budget
        assert game_config.random_seed == sim_config.random_seed
        assert game_config.food_reward == sim_config.food_reward
        assert game_config.collision_penalty == sim_config.collision_penalty
    
    def test_validate_all_configs(self):
        """Test validation of all configuration types together."""
        sim_config = SimulationConfig()
        net_config = NetworkConfig()
        log_config = LoggingConfig()
        exp_config = ExperimentConfig()
        repro_config = ReproducibilityConfig()
        
        # Should not raise any exception
        ConfigurationManager.validate_all_configs(
            sim_config, net_config, log_config, exp_config, repro_config
        )
    
    def test_validate_all_configs_with_invalid(self):
        """Test validation fails when any config is invalid."""
        sim_config = SimulationConfig(grid_size=(2, 2))  # Invalid
        net_config = NetworkConfig()
        log_config = LoggingConfig()
        
        with pytest.raises(ValueError):
            ConfigurationManager.validate_all_configs(sim_config, net_config, log_config)
    
    def test_create_experiment_variants(self):
        """Test creation of configuration variants for parameter sweeps."""
        base_config = SimulationConfig(move_budget=100)
        
        variants = {
            "move_budget": [50, 100, 150],
            "grid_size": [(10, 10), (15, 15)],
            "nn_enabled": [True, False]
        }
        
        configs = ConfigurationManager.create_experiment_variants(base_config, variants)
        
        # Should create 3 * 2 * 2 = 12 variants
        assert len(configs) == 12
        
        # Check that all combinations are present
        budgets = {config.move_budget for config in configs}
        grid_sizes = {config.grid_size for config in configs}
        nn_enabled_values = {config.nn_enabled for config in configs}
        
        assert budgets == {50, 100, 150}
        assert grid_sizes == {(10, 10), (15, 15)}
        assert nn_enabled_values == {True, False}
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration to/from file."""
        config = SimulationConfig(
            grid_size=(12, 12),
            move_budget=80,
            random_seed=456
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            ConfigurationManager.save_config_to_file(config, temp_path)
            
            # Load config
            loaded_config = ConfigurationManager.load_config_from_file(SimulationConfig, temp_path)
            
            # Verify they match
            assert loaded_config.grid_size == config.grid_size
            assert loaded_config.move_budget == config.move_budget
            assert loaded_config.random_seed == config.random_seed
            
        finally:
            Path(temp_path).unlink()


class TestReproducibilityConfig:
    """Test suite for reproducibility configuration."""
    
    def test_seed_sequence_generation(self):
        """Test automatic seed sequence generation."""
        config = ReproducibilityConfig(master_seed=42)
        
        # Should have generated a seed sequence
        assert config.seed_sequence is not None
        assert len(config.seed_sequence) == 100
        
        # All seeds should be valid
        for seed in config.seed_sequence:
            assert 0 <= seed < 2**31
        
        # Should be deterministic - same master seed produces same sequence
        config2 = ReproducibilityConfig(master_seed=42)
        assert config.seed_sequence == config2.seed_sequence
        
        # Different master seed produces different sequence
        config3 = ReproducibilityConfig(master_seed=123)
        assert config.seed_sequence != config3.seed_sequence
    
    def test_custom_seed_sequence(self):
        """Test using custom seed sequence."""
        custom_sequence = [1, 2, 3, 4, 5]
        config = ReproducibilityConfig(
            master_seed=42,
            seed_sequence=custom_sequence
        )
        
        # Should use the provided sequence
        assert config.seed_sequence == custom_sequence
    
    def test_setup_reproducibility(self):
        """Test reproducibility setup functionality."""
        import torch
        import numpy as np
        import random
        
        config = ReproducibilityConfig(
            master_seed=12345,
            use_deterministic_algorithms=True,
            benchmark_mode=False
        )
        
        # Setup reproducibility
        ConfigurationManager.setup_reproducibility(config)
        
        # Test that random states are set
        # Generate some random numbers
        py_random = random.random()
        np_random = np.random.random()
        torch_random = torch.rand(1).item()
        
        # Reset and generate again - should be the same
        ConfigurationManager.setup_reproducibility(config)
        
        py_random2 = random.random()
        np_random2 = np.random.random()
        torch_random2 = torch.rand(1).item()
        
        assert py_random == py_random2
        assert np_random == np_random2
        assert torch_random == torch_random2


class TestConfigurationInheritance:
    """Test suite for configuration inheritance and composition."""
    
    def test_default_configurations(self):
        """Test that default configurations are valid."""
        sim_config = SimulationConfig()
        net_config = NetworkConfig()
        log_config = LoggingConfig()
        exp_config = ExperimentConfig()
        repro_config = ReproducibilityConfig()
        
        # All default configs should be valid
        ConfigValidator.validate_simulation_config(sim_config)
        ConfigValidator.validate_network_config(net_config)
        ConfigValidator.validate_logging_config(log_config)
        ConfigValidator.validate_experiment_config(exp_config)
        ConfigValidator.validate_reproducibility_config(repro_config)
    
    def test_configuration_composition(self):
        """Test that configurations can be composed together."""
        # Create configurations with different settings
        sim_config = SimulationConfig(
            grid_size=(12, 12),
            move_budget=60,
            nn_enabled=True
        )
        
        net_config = NetworkConfig(
            hidden_layers=(150, 150),
            learning_rate=0.005,
            batch_size=24
        )
        
        log_config = LoggingConfig(
            level="DEBUG",
            log_clone_steps=True,
            log_neural_network=True
        )
        
        exp_config = ExperimentConfig(
            experiment_name="composition_test",
            num_simulations=3,
            save_results=True
        )
        
        repro_config = ReproducibilityConfig(
            master_seed=789,
            use_deterministic_algorithms=True
        )
        
        # Should validate successfully when composed
        ConfigurationManager.validate_all_configs(
            sim_config, net_config, log_config, exp_config, repro_config
        )
        
        # Test that they maintain their individual properties
        assert sim_config.grid_size == (12, 12)
        assert net_config.hidden_layers == (150, 150)
        assert log_config.level == "DEBUG"
        assert exp_config.experiment_name == "composition_test"
        assert repro_config.master_seed == 789
    
    def test_configuration_serialization(self):
        """Test that configurations can be serialized and deserialized."""
        configs = {
            "simulation": SimulationConfig(grid_size=(8, 8), move_budget=40),
            "network": NetworkConfig(hidden_layers=(100, 100), learning_rate=0.01),
            "logging": LoggingConfig(level="WARNING", log_file="test.log"),
            "experiment": ExperimentConfig(num_simulations=5, save_results=False),
            "reproducibility": ReproducibilityConfig(master_seed=999)
        }
        
        # Convert to dictionaries
        config_dicts = {key: asdict(config) for key, config in configs.items()}
        
        # Should be JSON serializable
        json_str = json.dumps(config_dicts, default=str)
        loaded_dicts = json.loads(json_str)
        
        # Convert lists back to tuples for tuple fields
        if 'grid_size' in loaded_dicts["simulation"]:
            loaded_dicts["simulation"]["grid_size"] = tuple(loaded_dicts["simulation"]["grid_size"])
        if 'hidden_layers' in loaded_dicts["network"]:
            loaded_dicts["network"]["hidden_layers"] = tuple(loaded_dicts["network"]["hidden_layers"])
        
        # Should be able to reconstruct configurations
        reconstructed_sim = SimulationConfig(**loaded_dicts["simulation"])
        reconstructed_net = NetworkConfig(**loaded_dicts["network"])
        
        assert reconstructed_sim.grid_size == configs["simulation"].grid_size
        assert reconstructed_net.hidden_layers == configs["network"].hidden_layers