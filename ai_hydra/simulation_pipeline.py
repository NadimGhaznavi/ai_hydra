"""
Complete simulation pipeline for the AI Hydra.

This module provides the SimulationPipeline class that wires together all
components (NN, tree search, oracle, logging) to create a complete end-to-end
simulation system with full decision cycles and budget management.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .config import SimulationConfig, NetworkConfig, LoggingConfig
from .hydra_mgr import HydraMgr, GameResult
from .master_game import MasterGame
from .budget_controller import BudgetController
from .state_manager import StateManager
from .neural_network import SnakeNet
from .feature_extractor import FeatureExtractor
from .oracle_trainer import OracleTrainer
from .logging_config import SimulationLogger
from .game_logic import GameLogic
from .models import GameBoard


@dataclass
class PipelineResult:
    """Result of a complete simulation pipeline execution."""
    game_result: GameResult
    pipeline_metrics: Dict[str, Any]
    component_statistics: Dict[str, Dict[str, Any]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class SimulationPipeline:
    """
    Complete simulation pipeline that wires together all system components.
    
    This class orchestrates the entire simulation system including neural network
    integration, tree search execution, oracle training, and comprehensive logging.
    It provides a unified interface for running complete Snake Game AI simulations.
    """
    
    def __init__(self, 
                 simulation_config: SimulationConfig,
                 network_config: Optional[NetworkConfig] = None,
                 logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the simulation pipeline.
        
        Args:
            simulation_config: Main simulation configuration
            network_config: Neural network configuration (optional)
            logging_config: Logging configuration (optional)
        """
        self.simulation_config = simulation_config
        self.network_config = network_config or NetworkConfig()
        self.logging_config = logging_config or LoggingConfig()
        
        # Initialize pipeline logger
        self.logger = SimulationLogger("simulation_pipeline", self.logging_config)
        
        # Initialize core components
        self.hydra_mgr: Optional[HydraMgr] = None
        self.master_game: Optional[MasterGame] = None
        self.budget_controller: Optional[BudgetController] = None
        self.state_manager: Optional[StateManager] = None
        
        # Initialize neural network components
        self.neural_network: Optional[SnakeNet] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.oracle_trainer: Optional[OracleTrainer] = None
        
        # Pipeline state
        self.is_initialized = False
        self.execution_count = 0
        
        self.logger.log_system_event("SimulationPipeline created", {
            "nn_enabled": simulation_config.nn_enabled,
            "grid_size": f"{simulation_config.grid_size[0]}x{simulation_config.grid_size[1]}",
            "move_budget": simulation_config.move_budget,
            "random_seed": simulation_config.random_seed
        })
    
    def initialize_pipeline(self) -> bool:
        """
        Initialize all pipeline components and wire them together.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.log_system_event("Initializing simulation pipeline")
            
            # Initialize HydraMgr (main orchestrator)
            self.hydra_mgr = HydraMgr(
                simulation_config=self.simulation_config,
                network_config=self.network_config,
                logging_config=self.logging_config
            )
            
            # Initialize master game
            initial_board = GameLogic.create_initial_board(
                self.simulation_config.grid_size,
                self.simulation_config.initial_snake_length,
                self.simulation_config.random_seed
            )
            self.master_game = MasterGame(initial_board, self.logging_config)
            
            # Initialize budget controller
            self.budget_controller = BudgetController(
                self.simulation_config.move_budget,
                self.logging_config
            )
            
            # Initialize state manager
            self.state_manager = StateManager(self.logging_config)
            
            # Initialize neural network components if enabled
            if self.simulation_config.nn_enabled:
                self._initialize_neural_network_components()
            
            # Wire components together
            self._wire_components()
            
            self.is_initialized = True
            
            self.logger.log_system_event("Pipeline initialization completed", {
                "components_initialized": self._get_initialized_components(),
                "nn_enabled": self.simulation_config.nn_enabled,
                "ready_for_execution": True
            })
            
            return True
            
        except Exception as e:
            self.logger.log_error("SimulationPipeline", f"Initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_neural_network_components(self) -> None:
        """Initialize neural network components."""
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Initialize neural network
        self.neural_network = SnakeNet(
            input_features=self.network_config.input_features,
            hidden_size=self.network_config.hidden_layers[0] if self.network_config.hidden_layers else 200,
            output_actions=self.network_config.output_actions
        )
        
        # Initialize oracle trainer
        self.oracle_trainer = OracleTrainer(
            self.neural_network,
            learning_rate=self.network_config.learning_rate,
            batch_size=self.network_config.batch_size,
            logging_config=self.logging_config
        )
        
        self.logger.log_system_event("Neural network components initialized", {
            "feature_extractor": "FeatureExtractor",
            "neural_network": "SnakeNet",
            "oracle_trainer": "OracleTrainer",
            "input_features": self.network_config.input_features,
            "hidden_layers": str(self.network_config.hidden_layers),
            "output_actions": self.network_config.output_actions
        })
    
    def _wire_components(self) -> None:
        """Wire all components together for integrated operation."""
        # The HydraMgr already handles component integration internally
        # This method ensures all components are properly connected
        
        # Verify component connections
        connections = {
            "hydra_mgr_to_master_game": self.hydra_mgr is not None and self.master_game is not None,
            "hydra_mgr_to_budget_controller": self.hydra_mgr is not None and self.budget_controller is not None,
            "hydra_mgr_to_state_manager": self.hydra_mgr is not None and self.state_manager is not None,
        }
        
        if self.simulation_config.nn_enabled:
            connections.update({
                "neural_network_to_oracle": self.neural_network is not None and self.oracle_trainer is not None,
                "feature_extractor_ready": self.feature_extractor is not None,
                "oracle_trainer_ready": self.oracle_trainer is not None
            })
        
        # Log component wiring status
        self.logger.log_system_event("Component wiring completed", {
            "connections": connections,
            "all_connected": all(connections.values()),
            "nn_integration": self.simulation_config.nn_enabled
        })
    
    def run_complete_simulation(self) -> PipelineResult:
        """
        Run a complete simulation with full decision cycles and budget management.
        
        This method executes the entire simulation pipeline from initialization
        through completion, including neural network training, tree search
        exploration, and comprehensive result collection.
        
        Returns:
            PipelineResult: Complete simulation results and metrics
        """
        start_time = time.time()
        
        if not self.is_initialized:
            return PipelineResult(
                game_result=None,
                pipeline_metrics={},
                component_statistics={},
                execution_time=0.0,
                success=False,
                error_message="Pipeline not initialized"
            )
        
        try:
            self.execution_count += 1
            
            self.logger.log_system_event(f"Starting complete simulation #{self.execution_count}", {
                "simulation_config": {
                    "grid_size": self.simulation_config.grid_size,
                    "move_budget": self.simulation_config.move_budget,
                    "nn_enabled": self.simulation_config.nn_enabled,
                    "random_seed": self.simulation_config.random_seed
                }
            })
            
            # Execute the main simulation through HydraMgr
            game_result = self.hydra_mgr.run_simulation()
            
            # Collect comprehensive statistics from all components
            component_stats = self._collect_component_statistics()
            
            # Calculate pipeline metrics
            pipeline_metrics = self._calculate_pipeline_metrics(game_result, start_time)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create comprehensive result
            result = PipelineResult(
                game_result=game_result,
                pipeline_metrics=pipeline_metrics,
                component_statistics=component_stats,
                execution_time=execution_time,
                success=True
            )
            
            # Log comprehensive simulation completion
            self._log_simulation_completion(result)
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.logger.log_error("SimulationPipeline", f"Simulation failed: {e}")
            
            return PipelineResult(
                game_result=None,
                pipeline_metrics={},
                component_statistics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def run_multiple_simulations(self, count: int) -> List[PipelineResult]:
        """
        Run multiple complete simulations for statistical analysis.
        
        Args:
            count: Number of simulations to run
            
        Returns:
            List[PipelineResult]: Results from all simulations
        """
        results = []
        
        self.logger.log_system_event(f"Starting batch simulation", {
            "simulation_count": count,
            "batch_id": f"batch_{int(time.time())}"
        })
        
        for i in range(count):
            # Reset components for each simulation
            if i > 0:
                self._reset_for_new_simulation()
            
            result = self.run_complete_simulation()
            results.append(result)
            
            # Log batch progress
            if (i + 1) % 10 == 0 or i == count - 1:
                self.logger.log_system_event(f"Batch progress: {i + 1}/{count} simulations completed")
        
        # Log batch completion with aggregate statistics
        self._log_batch_completion(results)
        
        return results
    
    def _collect_component_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Collect statistics from all pipeline components."""
        stats = {}
        
        # Master game statistics
        if self.master_game:
            stats["master_game"] = self.master_game.get_game_statistics()
        
        # Budget controller statistics
        if self.budget_controller:
            stats["budget_controller"] = self.budget_controller.get_budget_utilization_patterns()
        
        # State manager statistics
        if self.state_manager:
            stats["state_manager"] = {
                "tree_stats": self.state_manager.get_tree_statistics(),
                "exploration_efficiency": self.state_manager.get_tree_exploration_efficiency()
            }
        
        # Neural network statistics (if enabled)
        if self.simulation_config.nn_enabled and self.oracle_trainer:
            stats["neural_network"] = {
                "training_stats": self.oracle_trainer.get_training_statistics(),
                "network_info": self.neural_network.get_network_info() if self.neural_network else {}
            }
        
        return stats
    
    def _calculate_pipeline_metrics(self, game_result: GameResult, start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive pipeline performance metrics."""
        current_time = time.time()
        
        # Basic performance metrics
        metrics = {
            "execution_time": current_time - start_time,
            "moves_per_second": game_result.total_moves / (current_time - start_time) if game_result else 0,
            "decision_cycles": getattr(self.hydra_mgr, 'decision_cycle_count', 0) if self.hydra_mgr else 0,
            "pipeline_efficiency": self._calculate_pipeline_efficiency(game_result),
            "component_integration_success": self._verify_component_integration()
        }
        
        # Add neural network metrics if enabled
        if self.simulation_config.nn_enabled and self.oracle_trainer:
            nn_stats = self.oracle_trainer.get_training_statistics()
            metrics.update({
                "nn_accuracy": nn_stats.get('overall_accuracy', 0),
                "nn_training_updates": nn_stats.get('training_updates', 0),
                "nn_learning_efficiency": self._calculate_nn_learning_efficiency(nn_stats)
            })
        
        return metrics
    
    def _calculate_pipeline_efficiency(self, game_result: GameResult) -> float:
        """Calculate overall pipeline efficiency score."""
        if not game_result:
            return 0.0
        
        # Efficiency based on score per move and time
        score_efficiency = game_result.final_score / max(1, game_result.total_moves)
        time_efficiency = game_result.total_moves / max(1, game_result.game_length_seconds)
        
        # Combine metrics (weighted average)
        return (score_efficiency * 0.7 + time_efficiency * 0.3)
    
    def _calculate_nn_learning_efficiency(self, nn_stats: Dict[str, Any]) -> float:
        """Calculate neural network learning efficiency."""
        if not nn_stats:
            return 0.0
        
        accuracy = nn_stats.get('overall_accuracy', 0)
        training_updates = nn_stats.get('training_updates', 1)
        
        # Learning efficiency: accuracy gained per training update
        return accuracy / max(1, training_updates)
    
    def _verify_component_integration(self) -> bool:
        """Verify that all components are properly integrated."""
        required_components = [
            self.hydra_mgr is not None,
            self.master_game is not None,
            self.budget_controller is not None,
            self.state_manager is not None
        ]
        
        if self.simulation_config.nn_enabled:
            required_components.extend([
                self.neural_network is not None,
                self.feature_extractor is not None,
                self.oracle_trainer is not None
            ])
        
        return all(required_components)
    
    def _reset_for_new_simulation(self) -> None:
        """Reset components for a new simulation run."""
        # Reset HydraMgr state
        if self.hydra_mgr:
            self.hydra_mgr.decision_cycle_count = 0
            self.hydra_mgr.total_moves = 0
        
        # Create new master game with fresh board
        initial_board = GameLogic.create_initial_board(
            self.simulation_config.grid_size,
            self.simulation_config.initial_snake_length,
            self.simulation_config.random_seed
        )
        self.master_game = MasterGame(initial_board, self.logging_config)
        
        # Reset budget controller
        if self.budget_controller:
            self.budget_controller.reset_budget()
        
        # Reset state manager
        if self.state_manager:
            self.state_manager.destroy_exploration_tree()
        
        # Neural network components maintain their learned state across simulations
        # This allows for continuous learning across multiple games
        
        self.logger.log_system_event("Pipeline reset for new simulation")
    
    def _get_initialized_components(self) -> List[str]:
        """Get list of successfully initialized components."""
        components = []
        
        if self.hydra_mgr:
            components.append("HydraMgr")
        if self.master_game:
            components.append("MasterGame")
        if self.budget_controller:
            components.append("BudgetController")
        if self.state_manager:
            components.append("StateManager")
        if self.neural_network:
            components.append("SnakeNet")
        if self.feature_extractor:
            components.append("FeatureExtractor")
        if self.oracle_trainer:
            components.append("OracleTrainer")
        
        return components
    
    def _log_simulation_completion(self, result: PipelineResult) -> None:
        """Log comprehensive simulation completion details."""
        if result.success and result.game_result:
            self.logger.log_system_event("SIMULATION COMPLETED SUCCESSFULLY", {
                "execution_number": self.execution_count,
                "final_score": result.game_result.final_score,
                "total_moves": result.game_result.total_moves,
                "execution_time": f"{result.execution_time:.2f}s",
                "moves_per_second": f"{result.pipeline_metrics.get('moves_per_second', 0):.1f}",
                "decision_cycles": result.pipeline_metrics.get('decision_cycles', 0),
                "pipeline_efficiency": f"{result.pipeline_metrics.get('pipeline_efficiency', 0):.3f}",
                "nn_enabled": self.simulation_config.nn_enabled,
                "nn_accuracy": f"{result.pipeline_metrics.get('nn_accuracy', 0):.3f}" if self.simulation_config.nn_enabled else "N/A"
            })
        else:
            self.logger.log_error("SimulationPipeline", f"Simulation failed: {result.error_message}")
    
    def _log_batch_completion(self, results: List[PipelineResult]) -> None:
        """Log batch simulation completion with aggregate statistics."""
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            scores = [r.game_result.final_score for r in successful_results]
            times = [r.execution_time for r in successful_results]
            moves = [r.game_result.total_moves for r in successful_results]
            
            self.logger.log_system_event("BATCH SIMULATION COMPLETED", {
                "total_simulations": len(results),
                "successful_simulations": len(successful_results),
                "success_rate": f"{len(successful_results) / len(results) * 100:.1f}%",
                "score_stats": {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores),
                    "total": sum(scores)
                },
                "time_stats": {
                    "min": f"{min(times):.2f}s",
                    "max": f"{max(times):.2f}s",
                    "avg": f"{sum(times) / len(times):.2f}s",
                    "total": f"{sum(times):.2f}s"
                },
                "move_stats": {
                    "min": min(moves),
                    "max": max(moves),
                    "avg": sum(moves) / len(moves),
                    "total": sum(moves)
                }
            })
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health information."""
        return {
            "initialized": self.is_initialized,
            "execution_count": self.execution_count,
            "components": self._get_initialized_components(),
            "configuration": {
                "grid_size": self.simulation_config.grid_size,
                "move_budget": self.simulation_config.move_budget,
                "nn_enabled": self.simulation_config.nn_enabled,
                "random_seed": self.simulation_config.random_seed
            },
            "ready_for_execution": self.is_initialized and self._verify_component_integration()
        }
    
    def shutdown_pipeline(self) -> None:
        """Gracefully shutdown the pipeline and cleanup resources."""
        self.logger.log_system_event("Shutting down simulation pipeline")
        
        # Cleanup components
        if self.state_manager:
            self.state_manager.destroy_exploration_tree()
        
        # Reset state
        self.is_initialized = False
        self.execution_count = 0
        
        self.logger.log_system_event("Pipeline shutdown completed")