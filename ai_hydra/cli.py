"""
Command-line interface for the AI Hydra.

This module provides a CLI interface using Hydra Zen for running Snake Game
AI simulations with configurable parameters and experimental setups.
"""

import hydra
from hydra_zen import zen, store
from omegaconf import DictConfig
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config import (
    SimulationConfig, NetworkConfig, LoggingConfig, 
    ExperimentConfig, ReproducibilityConfig,
    ConfigurationManager, hydra_zen_app
)
from .simulation_pipeline import SimulationPipeline, PipelineResult


# Configure Hydra Zen store for CLI
store.add_to_hydra_store()


def setup_logging(log_config: LoggingConfig) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_config.level),
        format=log_config.format,
        filename=log_config.log_file
    )


def save_results(results: PipelineResult, output_dir: Path, experiment_name: str) -> None:
    """Save simulation results to files."""
    import json
    from dataclasses import asdict
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    results_file = output_dir / f"{experiment_name}_results.json"
    with open(results_file, 'w') as f:
        # Convert results to serializable format
        results_dict = {
            "success": results.success,
            "execution_time": results.execution_time,
            "error_message": results.error_message,
            "pipeline_metrics": results.pipeline_metrics,
            "component_statistics": results.component_statistics
        }
        
        if results.game_result:
            results_dict["game_result"] = {
                "final_score": results.game_result.final_score,
                "total_moves": results.game_result.total_moves,
                "game_length_seconds": results.game_result.game_length_seconds,
                "neural_network_accuracy": results.game_result.neural_network_accuracy
            }
        
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"Results saved to {results_file}")


def save_model(pipeline: SimulationPipeline, output_dir: Path, experiment_name: str) -> None:
    """Save trained neural network model."""
    if not pipeline.simulation_config.nn_enabled or not pipeline.oracle_trainer:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / f"{experiment_name}_model.pth"
    
    pipeline.oracle_trainer.save_model(str(model_file))
    print(f"Model saved to {model_file}")


def simple_cli():
    """Simple CLI entry point without Hydra configuration complexity."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Hydra - Parallel Neural Network Snake Game AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  ai-hydra
  
  # Run with custom grid size
  ai-hydra --grid-size 8 8
  
  # Run with more thinking budget
  ai-hydra --budget 100
  
  # Run multiple games
  ai-hydra --num-games 5
        """
    )
    
    parser.add_argument(
        "--grid-size", 
        nargs=2, 
        type=int, 
        default=[10, 10],
        help="Grid size (width height), default: 10 10"
    )
    
    parser.add_argument(
        "--budget", 
        type=int, 
        default=50,
        help="Move budget for tree search, default: 50"
    )
    
    parser.add_argument(
        "--num-games", 
        type=int, 
        default=1,
        help="Number of games to run, default: 1"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed, default: 42"
    )
    
    parser.add_argument(
        "--no-nn", 
        action="store_true",
        help="Disable neural network (tree search only)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create configurations from arguments
    sim_config = SimulationConfig(
        grid_size=tuple(args.grid_size),
        move_budget=args.budget,
        random_seed=args.seed,
        nn_enabled=not args.no_nn
    )
    
    net_config = NetworkConfig()
    
    log_config = LoggingConfig(
        level="DEBUG" if args.verbose else "INFO"
    )
    
    exp_config = ExperimentConfig(
        num_simulations=args.num_games,
        experiment_name="cli_run"
    )
    
    repro_config = ReproducibilityConfig(
        master_seed=args.seed
    )
    
    # Print startup info
    print("🐍 AI Hydra - Parallel Neural Network Snake AI")
    print("=" * 50)
    print(f"🎮 Grid Size: {args.grid_size[0]}x{args.grid_size[1]}")
    print(f"🧠 Move Budget: {args.budget}")
    print(f"🎯 Games to Run: {args.num_games}")
    print(f"🔢 Random Seed: {args.seed}")
    print(f"🤖 Neural Network: {'Enabled' if not args.no_nn else 'Disabled'}")
    print()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, log_config.level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("cli")
    
    try:
        # Validate configurations
        ConfigurationManager.validate_all_configs(
            sim_config, net_config, log_config, exp_config, repro_config
        )
        
        # Setup reproducibility
        ConfigurationManager.setup_reproducibility(repro_config)
        
        # Create pipeline
        from .simulation_pipeline import SimulationPipeline
        pipeline = SimulationPipeline(sim_config, net_config, log_config)
        
        # Initialize pipeline
        if not pipeline.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return 1
        
        # Run simulations
        if exp_config.num_simulations == 1:
            logger.info("🚀 Starting simulation...")
            result = pipeline.run_complete_simulation()
            results = [result]
        else:
            logger.info(f"🚀 Starting {exp_config.num_simulations} simulations...")
            results = pipeline.run_multiple_simulations(exp_config.num_simulations)
        
        # Process results
        successful_results = [r for r in results if r.success]
        logger.info(f"✅ Completed {len(successful_results)}/{len(results)} simulations successfully")
        
        if successful_results:
            # Calculate statistics
            scores = [r.game_result.final_score for r in successful_results if r.game_result]
            
            if scores:
                print("\n📊 Results:")
                print(f"   Average Score: {sum(scores)/len(scores):.1f}")
                print(f"   Best Score: {max(scores)}")
                print(f"   Worst Score: {min(scores)}")
                
                success_count = sum(1 for score in scores if score > 10)
                print(f"   Success Rate: {success_count}/{len(scores)} ({100*success_count/len(scores):.1f}%)")
                
                if success_count > 0:
                    print("🎉 Great! Scores > 10 indicate successful collision avoidance!")
                else:
                    print("💪 Keep trying! Scores > 10 indicate successful AI learning.")
        
        # Shutdown pipeline
        pipeline.shutdown_pipeline()
        logger.info("🏁 Simulation completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        return 1


# Keep the Hydra version for advanced users
@hydra.main(version_base=None, config_path=None, config_name=None)
def run_simulation_cli(cfg: DictConfig = None) -> None:
    """
    Main CLI entry point for running simulations.
    
    This function is called by Hydra with the resolved configuration.
    """
    # Use default configurations if no config provided
    if cfg is None or len(cfg) == 0:
        sim_config = SimulationConfig()
        net_config = NetworkConfig()
        log_config = LoggingConfig()
        exp_config = ExperimentConfig()
        repro_config = ReproducibilityConfig()
    else:
        # Convert DictConfig to dataclass instances
        sim_config = SimulationConfig(**cfg.get('simulation', {}))
        net_config = NetworkConfig(**cfg.get('network', {}))
        log_config = LoggingConfig(**cfg.get('logging', {}))
        exp_config = ExperimentConfig(**cfg.get('experiment', {}))
        repro_config = ReproducibilityConfig(**cfg.get('reproducibility', {}))
    
    # Setup logging
    setup_logging(log_config)
    logger = logging.getLogger("cli")
    
    logger.info(f"Starting experiment: {exp_config.experiment_name}")
    logger.info(f"Configuration: {cfg}")
    
    try:
        # Validate configurations
        ConfigurationManager.validate_all_configs(
            sim_config, net_config, log_config, exp_config, repro_config
        )
        
        # Setup reproducibility
        ConfigurationManager.setup_reproducibility(repro_config)
        
        # Create pipeline
        pipeline = SimulationPipeline(sim_config, net_config, log_config)
        
        # Initialize pipeline
        if not pipeline.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return
        
        # Run simulations
        if exp_config.num_simulations == 1:
            logger.info("Running single simulation")
            result = pipeline.run_complete_simulation()
            results = [result]
        else:
            logger.info(f"Running {exp_config.num_simulations} simulations")
            results = pipeline.run_multiple_simulations(exp_config.num_simulations)
        
        # Process results
        successful_results = [r for r in results if r.success]
        logger.info(f"Completed {len(successful_results)}/{len(results)} simulations successfully")
        
        if successful_results:
            # Calculate aggregate statistics
            scores = [r.game_result.final_score for r in successful_results]
            moves = [r.game_result.total_moves for r in successful_results]
            times = [r.execution_time for r in successful_results]
            
            logger.info(f"Score statistics: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}")
            logger.info(f"Move statistics: min={min(moves)}, max={max(moves)}, avg={sum(moves)/len(moves):.1f}")
            logger.info(f"Time statistics: min={min(times):.2f}s, max={max(times):.2f}s, avg={sum(times)/len(times):.2f}s")
            
            # Save results if requested
            if exp_config.save_results:
                output_dir = Path(exp_config.output_directory)
                
                # Save individual results
                for i, result in enumerate(successful_results):
                    save_results(result, output_dir, f"{exp_config.experiment_name}_run_{i+1}")
                
                # Save aggregate results
                aggregate_results = {
                    "experiment_name": exp_config.experiment_name,
                    "total_simulations": len(results),
                    "successful_simulations": len(successful_results),
                    "configuration": {
                        "simulation": sim_config.__dict__,
                        "network": net_config.__dict__,
                        "logging": log_config.__dict__,
                        "experiment": exp_config.__dict__,
                        "reproducibility": repro_config.__dict__
                    },
                    "aggregate_statistics": {
                        "scores": {"min": min(scores), "max": max(scores), "avg": sum(scores)/len(scores)},
                        "moves": {"min": min(moves), "max": max(moves), "avg": sum(moves)/len(moves)},
                        "times": {"min": min(times), "max": max(times), "avg": sum(times)/len(times)}
                    }
                }
                
                import json
                aggregate_file = output_dir / f"{exp_config.experiment_name}_aggregate.json"
                with open(aggregate_file, 'w') as f:
                    json.dump(aggregate_results, f, indent=2, default=str)
                
                logger.info(f"Aggregate results saved to {aggregate_file}")
            
            # Save model if requested
            if exp_config.save_models:
                output_dir = Path(exp_config.output_directory)
                save_model(pipeline, output_dir, exp_config.experiment_name)
        
        # Shutdown pipeline
        pipeline.shutdown_pipeline()
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


@zen
def main_zen(simulation: SimulationConfig = SimulationConfig(),
             network: NetworkConfig = NetworkConfig(),
             logging: LoggingConfig = LoggingConfig(),
             experiment: ExperimentConfig = ExperimentConfig(),
             reproducibility: ReproducibilityConfig = ReproducibilityConfig()) -> Dict[str, Any]:
    """
    Zen-style main function for programmatic usage.
    
    This function can be used for programmatic execution without Hydra CLI.
    """
    # Setup reproducibility
    ConfigurationManager.setup_reproducibility(reproducibility)
    
    # Create and run pipeline
    pipeline = SimulationPipeline(simulation, network, logging)
    
    if not pipeline.initialize_pipeline():
        raise RuntimeError("Failed to initialize pipeline")
    
    # Run simulations
    if experiment.num_simulations == 1:
        results = [pipeline.run_complete_simulation()]
    else:
        results = pipeline.run_multiple_simulations(experiment.num_simulations)
    
    # Process and return results
    successful_results = [r for r in results if r.success]
    
    pipeline.shutdown_pipeline()
    
    return {
        "total_simulations": len(results),
        "successful_simulations": len(successful_results),
        "results": successful_results,
        "configuration": {
            "simulation": simulation,
            "network": network,
            "logging": logging,
            "experiment": experiment,
            "reproducibility": reproducibility
        }
    }


def create_config_file(config_name: str = "default") -> None:
    """Create a sample configuration file for customization."""
    from .config import SimulationConfig, NetworkConfig, LoggingConfig, ExperimentConfig, ReproducibilityConfig
    
    configs = {
        "simulation": SimulationConfig(),
        "network": NetworkConfig(),
        "logging": LoggingConfig(),
        "experiment": ExperimentConfig(),
        "reproducibility": ReproducibilityConfig()
    }
    
    config_file = Path(f"{config_name}_config.yaml")
    
    import yaml
    from dataclasses import asdict
    
    config_dict = {key: asdict(config) for key, config in configs.items()}
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Sample configuration saved to {config_file}")
    print("You can customize this file and use it with: python -m ai_hydra --config-path . --config-name your_config")


if __name__ == "__main__":
    # Register configurations with Hydra store
    store.add_to_hydra_store()
    
    # Run CLI
    run_simulation_cli()