"""
Hydra Zen configuration system for the AI Hydra.

This module provides structured configuration management using Hydra Zen,
enabling easy parameter management and experimental reproducibility.
"""

from hydra_zen import make_config, store, zen
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Dict, List
from .models import GameConfig
import random


# Base configuration for Snake game simulation
@dataclass
class SimulationConfig:
    """Main configuration for Snake game simulation."""
    # Game parameters
    grid_size: Tuple[int, int] = (20, 20)
    initial_snake_length: int = 3
    random_seed: int = 42
    
    # Tree search parameters
    move_budget: int = 100
    max_tree_depth: Optional[int] = None
    max_moves_multiplier: int = 100  # Game terminates when moves > max_moves_multiplier * snake_length
    
    # Reward system
    food_reward: int = 10
    collision_penalty: int = -10
    empty_move_reward: int = 0
    
    # Neural network parameters
    nn_enabled: bool = True
    nn_learning_rate: float = 0.001
    nn_hidden_size: int = 200
    
    # Logging parameters
    log_level: str = "INFO"
    log_clone_steps: bool = True
    log_decision_cycles: bool = True
    
    # Performance parameters
    enable_profiling: bool = False
    memory_limit_mb: Optional[int] = None


@dataclass
class NetworkConfig:
    """Configuration for neural network components."""
    input_features: int = 19
    hidden_layers: Tuple[int, ...] = (200, 200)
    output_actions: int = 3
    learning_rate: float = 0.001
    batch_size: int = 32
    training_enabled: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_clone_steps: bool = True
    log_decision_cycles: bool = True
    log_neural_network: bool = True
    log_file: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for experimental setups and batch runs."""
    experiment_name: str = "snake_ai_experiment"
    num_simulations: int = 1
    output_directory: str = "experiments"
    save_results: bool = True
    save_models: bool = False
    parallel_execution: bool = False
    max_workers: Optional[int] = None


@dataclass
class ReproducibilityConfig:
    """Configuration for ensuring reproducible experiments."""
    master_seed: int = 42
    use_deterministic_algorithms: bool = True
    benchmark_mode: bool = False
    seed_sequence: Optional[List[int]] = None
    
    def __post_init__(self):
        """Generate seed sequence if not provided."""
        if self.seed_sequence is None:
            # Generate deterministic seed sequence based on master seed
            rng = random.Random(self.master_seed)
            self.seed_sequence = [rng.randint(0, 2**31 - 1) for _ in range(100)]


# Create Hydra Zen configs with proper inheritance and composition
SimulationConfigZen = make_config(
    hydra_defaults=["_self_"],
    simulation=SimulationConfig(),
    network=NetworkConfig(),
    logging=LoggingConfig(),
    experiment=ExperimentConfig(),
    reproducibility=ReproducibilityConfig(),
)

# Store base configurations
cs = store(group="simulation")
cs(SimulationConfig, name="default")
cs(SimulationConfig(
    move_budget=50, 
    grid_size=(15, 15),
    nn_enabled=True
), name="small")
cs(SimulationConfig(
    move_budget=200, 
    grid_size=(25, 25),
    nn_enabled=True,
    max_tree_depth=10
), name="large")
cs(SimulationConfig(
    move_budget=30,
    grid_size=(10, 10),
    nn_enabled=False
), name="fast")

# Network configurations
cs_net = store(group="network")
cs_net(NetworkConfig, name="default")
cs_net(NetworkConfig(
    hidden_layers=(100, 100),
    learning_rate=0.01,
    batch_size=16
), name="small")
cs_net(NetworkConfig(
    hidden_layers=(300, 300, 200),
    learning_rate=0.0005,
    batch_size=64
), name="large")
cs_net(NetworkConfig(
    hidden_layers=(150, 150),
    learning_rate=0.002,
    batch_size=8
), name="experimental")

# Logging configurations
cs_log = store(group="logging")
cs_log(LoggingConfig, name="default")
cs_log(LoggingConfig(
    level="DEBUG", 
    log_file="simulation_debug.log",
    log_clone_steps=True,
    log_decision_cycles=True,
    log_neural_network=True
), name="debug")
cs_log(LoggingConfig(
    level="WARNING", 
    log_clone_steps=False,
    log_decision_cycles=False,
    log_neural_network=False
), name="minimal")
cs_log(LoggingConfig(
    level="INFO",
    log_file="experiment.log",
    log_clone_steps=True,
    log_decision_cycles=True,
    log_neural_network=True
), name="experiment")

# Experiment configurations
cs_exp = store(group="experiment")
cs_exp(ExperimentConfig, name="default")
cs_exp(ExperimentConfig(
    experiment_name="batch_experiment",
    num_simulations=10,
    save_results=True,
    save_models=True
), name="batch")
cs_exp(ExperimentConfig(
    experiment_name="performance_test",
    num_simulations=100,
    parallel_execution=True,
    max_workers=4
), name="performance")
cs_exp(ExperimentConfig(
    experiment_name="single_run",
    num_simulations=1,
    save_results=False,
    save_models=False
), name="single")

# Reproducibility configurations
cs_repro = store(group="reproducibility")
cs_repro(ReproducibilityConfig, name="default")
cs_repro(ReproducibilityConfig(
    master_seed=12345,
    use_deterministic_algorithms=True,
    benchmark_mode=True
), name="strict")
cs_repro(ReproducibilityConfig(
    master_seed=999,
    use_deterministic_algorithms=False,
    benchmark_mode=False
), name="flexible")


class ConfigValidator:
    """Validates configuration objects for correctness and completeness."""
    
    @staticmethod
    def validate_simulation_config(config: SimulationConfig) -> None:
        """
        Validate a simulation configuration object.
        
        Args:
            config: The simulation configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Validate grid size
        if not isinstance(config.grid_size, tuple) or len(config.grid_size) != 2:
            raise ValueError("grid_size must be a tuple of two integers")
        
        width, height = config.grid_size
        if width < 5 or height < 5:
            raise ValueError("Grid size must be at least 5x5")
        
        if width > 100 or height > 100:
            raise ValueError("Grid size cannot exceed 100x100")
        
        # Validate snake length
        if config.initial_snake_length < 1:
            raise ValueError("Initial snake length must be at least 1")
        
        if config.initial_snake_length >= min(config.grid_size) - 2:
            raise ValueError("Initial snake length too large for grid size")
        
        # Validate move budget
        if config.move_budget < 1:
            raise ValueError("Move budget must be at least 1")
        
        if config.move_budget > 10000:
            raise ValueError("Move budget cannot exceed 10000")
        
        # Validate max tree depth
        if config.max_tree_depth is not None and config.max_tree_depth < 1:
            raise ValueError("Max tree depth must be at least 1 if specified")
        
        # Validate reward values
        if config.food_reward <= 0:
            raise ValueError("Food reward must be positive")
        
        if config.collision_penalty >= 0:
            raise ValueError("Collision penalty must be negative")
        
        # Validate neural network parameters
        if config.nn_learning_rate <= 0 or config.nn_learning_rate > 1:
            raise ValueError("Neural network learning rate must be between 0 and 1")
    
    @staticmethod
    def validate_network_config(config: NetworkConfig) -> None:
        """
        Validate a network configuration object.
        
        Args:
            config: The network configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        if config.input_features < 1:
            raise ValueError("Input features must be at least 1")
        
        if config.output_actions < 1:
            raise ValueError("Output actions must be at least 1")
        
        if not config.hidden_layers:
            raise ValueError("At least one hidden layer is required")
        
        for layer_size in config.hidden_layers:
            if layer_size < 1:
                raise ValueError("Hidden layer sizes must be at least 1")
        
        if config.learning_rate <= 0 or config.learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        
        if config.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
    
    @staticmethod
    def validate_logging_config(config: LoggingConfig) -> None:
        """
        Validate a logging configuration object.
        
        Args:
            config: The logging configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.level not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
    
    @staticmethod
    def validate_experiment_config(config: ExperimentConfig) -> None:
        """
        Validate an experiment configuration object.
        
        Args:
            config: The experiment configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        if config.num_simulations < 1:
            raise ValueError("Number of simulations must be at least 1")
        
        if config.num_simulations > 10000:
            raise ValueError("Number of simulations cannot exceed 10000")
        
        if config.parallel_execution and config.max_workers is not None:
            if config.max_workers < 1:
                raise ValueError("Max workers must be at least 1")
            if config.max_workers > 32:
                raise ValueError("Max workers cannot exceed 32")
    
    @staticmethod
    def validate_reproducibility_config(config: ReproducibilityConfig) -> None:
        """
        Validate a reproducibility configuration object.
        
        Args:
            config: The reproducibility configuration to validate
            
        Raises:
            ValueError: If the configuration is invalid
        """
        if config.master_seed < 0 or config.master_seed >= 2**31:
            raise ValueError("Master seed must be between 0 and 2^31-1")
        
        if config.seed_sequence is not None:
            if len(config.seed_sequence) < 1:
                raise ValueError("Seed sequence must contain at least one seed")
            
            for seed in config.seed_sequence:
                if seed < 0 or seed >= 2**31:
                    raise ValueError("All seeds in sequence must be between 0 and 2^31-1")


class ConfigurationManager:
    """Manages configuration inheritance, composition, and validation."""
    
    @staticmethod
    def create_game_config_from_simulation(sim_config: SimulationConfig) -> GameConfig:
        """
        Convert a SimulationConfig to a GameConfig for internal use.
        
        Args:
            sim_config: The simulation configuration
            
        Returns:
            GameConfig: The converted game configuration
        """
        return GameConfig(
            grid_size=sim_config.grid_size,
            initial_snake_length=sim_config.initial_snake_length,
            move_budget=sim_config.move_budget,
            random_seed=sim_config.random_seed,
            max_tree_depth=sim_config.max_tree_depth,
            food_reward=sim_config.food_reward,
            collision_penalty=sim_config.collision_penalty,
            empty_move_reward=sim_config.empty_move_reward,
        )
    
    @staticmethod
    def get_default_config() -> SimulationConfig:
        """Get the default simulation configuration."""
        return SimulationConfig()
    
    @staticmethod
    def validate_all_configs(sim_config: SimulationConfig, 
                            net_config: NetworkConfig,
                            log_config: LoggingConfig,
                            exp_config: Optional[ExperimentConfig] = None,
                            repro_config: Optional[ReproducibilityConfig] = None) -> None:
        """
        Validate all configuration objects together.
        
        Args:
            sim_config: Simulation configuration
            net_config: Network configuration  
            log_config: Logging configuration
            exp_config: Experiment configuration (optional)
            repro_config: Reproducibility configuration (optional)
            
        Raises:
            ValueError: If any configuration is invalid
        """
        ConfigValidator.validate_simulation_config(sim_config)
        ConfigValidator.validate_network_config(net_config)
        ConfigValidator.validate_logging_config(log_config)
        
        if exp_config is not None:
            ConfigValidator.validate_experiment_config(exp_config)
        
        if repro_config is not None:
            ConfigValidator.validate_reproducibility_config(repro_config)
    
    @staticmethod
    def setup_reproducibility(repro_config: ReproducibilityConfig) -> None:
        """
        Setup reproducibility settings based on configuration.
        
        Args:
            repro_config: Reproducibility configuration
        """
        import torch
        import numpy as np
        import random
        
        # Set master seed
        random.seed(repro_config.master_seed)
        np.random.seed(repro_config.master_seed)
        torch.manual_seed(repro_config.master_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(repro_config.master_seed)
            torch.cuda.manual_seed_all(repro_config.master_seed)
        
        # Set deterministic algorithms
        if repro_config.use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        
        # Set benchmark mode
        if repro_config.benchmark_mode:
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    
    @staticmethod
    def create_experiment_variants(base_config: SimulationConfig, 
                                 variants: Dict[str, List[Any]]) -> List[SimulationConfig]:
        """
        Create configuration variants for parameter sweeps.
        
        Args:
            base_config: Base configuration to vary
            variants: Dictionary of parameter names to lists of values
            
        Returns:
            List[SimulationConfig]: List of configuration variants
        """
        import itertools
        from copy import deepcopy
        
        # Get all combinations of variant parameters
        param_names = list(variants.keys())
        param_values = list(variants.values())
        
        configs = []
        for combination in itertools.product(*param_values):
            config = deepcopy(base_config)
            
            # Apply parameter values
            for param_name, param_value in zip(param_names, combination):
                setattr(config, param_name, param_value)
            
            configs.append(config)
        
        return configs
    
    @staticmethod
    def save_config_to_file(config: Any, filepath: str) -> None:
        """
        Save a configuration object to a file.
        
        Args:
            config: Configuration object to save
            filepath: Path to save the configuration
        """
        import json
        from dataclasses import asdict
        
        config_dict = asdict(config)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @staticmethod
    def load_config_from_file(config_class: type, filepath: str) -> Any:
        """
        Load a configuration object from a file.
        
        Args:
            config_class: Configuration class to instantiate
            filepath: Path to load the configuration from
            
        Returns:
            Configuration object instance
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to tuples for tuple fields
        if 'grid_size' in config_dict and isinstance(config_dict['grid_size'], list):
            config_dict['grid_size'] = tuple(config_dict['grid_size'])
        
        if 'hidden_layers' in config_dict and isinstance(config_dict['hidden_layers'], list):
            config_dict['hidden_layers'] = tuple(config_dict['hidden_layers'])
        
        if 'seed_sequence' in config_dict and isinstance(config_dict['seed_sequence'], list):
            config_dict['seed_sequence'] = config_dict['seed_sequence']  # Keep as list
        
        return config_class(**config_dict)


# Hydra Zen app configuration for CLI usage
def hydra_zen_app(simulation: SimulationConfig = SimulationConfig(),
                  network: NetworkConfig = NetworkConfig(), 
                  logging: LoggingConfig = LoggingConfig(),
                  experiment: ExperimentConfig = ExperimentConfig(),
                  reproducibility: ReproducibilityConfig = ReproducibilityConfig()) -> Dict[str, Any]:
    """
    Main Hydra Zen application entry point.
    
    This function serves as the main entry point for Hydra Zen configuration
    and can be used with the Hydra CLI for running experiments.
    
    Args:
        simulation: Simulation configuration
        network: Network configuration
        logging: Logging configuration
        experiment: Experiment configuration
        reproducibility: Reproducibility configuration
        
    Returns:
        Dict containing all configurations
    """
    # Validate all configurations
    ConfigurationManager.validate_all_configs(
        simulation, network, logging, experiment, reproducibility
    )
    
    # Setup reproducibility
    ConfigurationManager.setup_reproducibility(reproducibility)
    
    return {
        "simulation": simulation,
        "network": network,
        "logging": logging,
        "experiment": experiment,
        "reproducibility": reproducibility
    }


# Export main functions and classes
__all__ = [
    "SimulationConfig",
    "NetworkConfig", 
    "LoggingConfig",
    "ExperimentConfig",
    "ReproducibilityConfig",
    "ConfigValidator",
    "ConfigurationManager",
    "hydra_zen_app",
    "create_game_config_from_simulation",
    "get_default_config",
    "validate_all_configs"
]