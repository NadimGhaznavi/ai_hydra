"""
Main orchestration system for the AI Hydra.

This module provides the HydraMgr class which coordinates the entire simulation
system including master game management, exploration tree coordination, and
neural network integration.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .models import GameBoard, Move, MoveResult, ExplorationPath, TreeMetrics, GameConfig
from .config import SimulationConfig, NetworkConfig, LoggingConfig
from .logging_config import SimulationLogger
from .game_logic import GameLogic
from .error_handler import ErrorHandler


@dataclass
class GameResult:
    """Result of a complete game simulation."""
    final_score: int
    total_moves: int
    game_length_seconds: float
    tree_metrics: TreeMetrics
    neural_network_accuracy: Optional[float] = None


class HydraMgr:
    """
    Main orchestration system for Snake Game AI simulation.
    
    This class coordinates the entire simulation system including master game
    management, budget-constrained tree search, neural network integration,
    and comprehensive logging.
    """
    
    def __init__(self, simulation_config: SimulationConfig,
                 network_config: Optional[NetworkConfig] = None,
                 logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the HydraMgr with configuration.
        
        Args:
            simulation_config: Main simulation configuration
            network_config: Neural network configuration (optional)
            logging_config: Logging configuration (optional)
        """
        self.simulation_config = simulation_config
        self.network_config = network_config or NetworkConfig()
        self.logging_config = logging_config or LoggingConfig()
        
        # Initialize error handler first
        self.error_handler = ErrorHandler(self.logging_config)
        
        # Initialize logger
        self.logger = SimulationLogger("hydra_mgr", self.logging_config)
        
        # Initialize game configuration
        self.game_config = self._create_game_config()
        
        # Initialize master game
        self.master_game = None
        
        # Initialize components (will be created in run_simulation)
        self.budget_controller = None
        self.state_manager = None
        self.neural_network = None
        self.oracle_trainer = None
        
        # Simulation state
        self.decision_cycle_count = 0
        self.total_moves = 0
        
        self.logger.log_system_event("HydraMgr initialized", {
            "grid_size": self.simulation_config.grid_size,
            "move_budget": self.simulation_config.move_budget,
            "nn_enabled": self.simulation_config.nn_enabled
        })
    
    def _create_game_config(self) -> GameConfig:
        """Create GameConfig from SimulationConfig."""
        return GameConfig(
            grid_size=self.simulation_config.grid_size,
            initial_snake_length=self.simulation_config.initial_snake_length,
            move_budget=self.simulation_config.move_budget,
            random_seed=self.simulation_config.random_seed,
            max_tree_depth=self.simulation_config.max_tree_depth,
            max_moves_multiplier=self.simulation_config.max_moves_multiplier,
            food_reward=self.simulation_config.food_reward,
            collision_penalty=self.simulation_config.collision_penalty,
            empty_move_reward=self.simulation_config.empty_move_reward,
        )
    
    def run_simulation(self) -> GameResult:
        """
        Run a complete Snake Game simulation.
        
        This method orchestrates the entire simulation process including
        initialization, decision cycles, and final result compilation.
        
        Returns:
            GameResult: Complete simulation results
        """
        import time
        start_time = time.time()
        
        self.logger.log_system_event("Starting simulation")
        
        # Initialize all components
        self._initialize_components()
        
        # Main simulation loop
        while not self._is_simulation_complete():
            try:
                # Validate system state before decision cycle
                if not self._validate_system_state():
                    self.logger.log_error("HydraMgr", "System state validation failed, attempting recovery")
                    recovery_result = self.error_handler.handle_error(
                        RuntimeError("System state validation failed"),
                        "hydra_mgr",
                        "state_validation"
                    )
                    if not recovery_result.success:
                        break
                
                # Execute one decision cycle
                optimal_move = self.execute_decision_cycle()
                
                # Apply move to master game
                self.apply_move_to_master(optimal_move)
                
                # Reset for next cycle
                self.reset_exploration_tree()
                
                self.decision_cycle_count += 1
                self.total_moves += 1
                
            except Exception as e:
                # Use error handler for comprehensive error handling
                recovery_result = self.error_handler.handle_error(e, "hydra_mgr", "decision_cycle")
                
                if recovery_result.success and recovery_result.should_retry:
                    # Log recovery and continue
                    self.logger.log_system_event("Decision cycle error recovered", {
                        "error_type": type(e).__name__,
                        "recovery_action": recovery_result.action_taken.value,
                        "message": recovery_result.message
                    })
                    continue
                else:
                    # Log failure and break
                    self.logger.log_error("HydraMgr", f"Unrecoverable error in decision cycle: {e}")
                    break
        
        # Calculate final results
        end_time = time.time()
        simulation_time = end_time - start_time
        
        result = GameResult(
            final_score=self.master_game.get_score(),
            total_moves=self.total_moves,
            game_length_seconds=simulation_time,
            tree_metrics=self._get_tree_metrics(),
            neural_network_accuracy=self._get_nn_accuracy()
        )
        
        # Log comprehensive game summary
        self.log_comprehensive_game_summary()
        
        self.logger.log_system_event("Simulation completed", {
            "final_score": result.final_score,
            "total_moves": result.total_moves,
            "duration": f"{simulation_time:.2f}s",
            "decision_cycles": self.decision_cycle_count,
            "nn_accuracy": f"{result.neural_network_accuracy:.3f}" if result.neural_network_accuracy else "N/A"
        })
        
        return result
    
    def _initialize_components(self) -> None:
        """Initialize all system components with error handling."""
        try:
            # Import here to avoid circular imports
            from .master_game import MasterGame
            from .budget_controller import BudgetController
            from .state_manager import StateManager
            
            # Initialize master game with logging
            initial_board = GameLogic.create_initial_board(
                self.game_config.grid_size,
                self.game_config.initial_snake_length,
                self.game_config.random_seed
            )
            self.master_game = MasterGame(initial_board, self.logging_config, 
                                         self.game_config.max_moves_multiplier)
            
            # Initialize budget controller with logging
            self.budget_controller = BudgetController(self.game_config.move_budget, self.logging_config)
            
            # Initialize state manager with logging
            self.state_manager = StateManager(self.logging_config)
            
            # Initialize neural network components if enabled
            if self.simulation_config.nn_enabled:
                self._initialize_neural_network()
            
            # Verify all components are properly initialized
            self._verify_component_initialization()
            
            self.logger.log_system_event("All components initialized with logging integration")
            
        except Exception as e:
            # Use error handler for component initialization failures
            recovery_result = self.error_handler.handle_error(e, "hydra_mgr", "component_initialization")
            
            if not recovery_result.success:
                raise RuntimeError(f"Failed to initialize components: {e}")
            
            # If recovery suggests retry, attempt initialization again
            if recovery_result.should_retry and recovery_result.action_taken.value == "retry":
                self.logger.log_system_event("Retrying component initialization after error recovery")
                self._initialize_components()  # Recursive retry
    
    def _initialize_neural_network(self) -> None:
        """Initialize neural network components with error handling."""
        try:
            # Import here to avoid circular imports
            from .neural_network import SnakeNet
            from .feature_extractor import FeatureExtractor
            from .oracle_trainer import OracleTrainer
            
            # Initialize feature extractor first
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
                batch_size=self.network_config.batch_size
            )
            
            self.logger.log_system_event("Neural network components initialized", {
                "input_features": self.network_config.input_features,
                "hidden_layers": str(self.network_config.hidden_layers),
                "output_actions": self.network_config.output_actions,
                "learning_rate": self.network_config.learning_rate,
                "batch_size": self.network_config.batch_size
            })
            
        except Exception as e:
            # Handle neural network initialization errors
            recovery_result = self.error_handler.handle_error(e, "neural_network", "initialization")
            
            if recovery_result.success and recovery_result.recovered_data:
                # Disable neural network if recovery suggests it
                if not recovery_result.recovered_data.get('nn_enabled', True):
                    self.simulation_config.nn_enabled = False
                    self.logger.log_system_event("Neural network disabled due to initialization error")
            else:
                # Re-raise if recovery failed
                raise
    
    def execute_decision_cycle(self) -> Move:
        """
        Execute one complete decision cycle with hybrid NN + tree search.
        
        This method implements the hybrid execution system:
        1. Get neural network prediction (if enabled)
        2. Execute tree search exploration starting from NN move
        3. Compare NN prediction with tree search optimal result
        4. Choose tree search result if different from NN prediction
        5. Train neural network based on oracle feedback
        
        Returns:
            Move: The optimal move selected for this cycle
        """
        self.logger.log_system_event(f"Starting decision cycle {self.decision_cycle_count + 1}")
        
        # Step 1: Get neural network prediction if enabled
        nn_prediction = None
        current_board = self.master_game.get_current_board()
        current_score = current_board.score
        
        if self.simulation_config.nn_enabled and self.neural_network:
            nn_prediction = self._get_neural_network_prediction()
            self.logger.log_neural_network_prediction(
                nn_prediction.action.value, 
                0.85,  # Placeholder confidence - will be updated when NN prediction is enhanced
                current_score
            )
        
        # Step 2: Execute tree search exploration
        # If NN is enabled, tree search starts from NN prediction
        exploration_paths = self.execute_simulation_round(nn_prediction)
        
        # Step 3: Evaluate paths and select optimal move
        optimal_path = self.evaluate_exploration_paths(exploration_paths)
        optimal_move = optimal_path.moves[0] if optimal_path.moves else None
        
        if not optimal_move:
            # Fallback to first possible move if no optimal move found
            possible_moves = GameLogic.get_possible_moves(current_board.direction)
            optimal_move = possible_moves[0]
            self.logger.log_warning("HydraMgr", "No optimal move found, using fallback")
        
        # Step 4: Oracle decision logic - always prefer tree search result
        final_decision = optimal_move
        
        # Step 5: Train neural network if enabled and predictions differ
        if self.simulation_config.nn_enabled and nn_prediction and optimal_move:
            self._train_neural_network(nn_prediction, optimal_move)
            
            # Log oracle decision
            self.logger.log_oracle_decision(
                nn_prediction.action.value,
                optimal_move.action.value,
                final_decision.action.value,
                current_score
            )
        
        # Log decision cycle completion with comprehensive summary
        decision_summary = self._create_decision_cycle_summary(
            optimal_path, exploration_paths, nn_prediction, final_decision
        )
        
        self.logger.log_decision_cycle(
            self.decision_cycle_count + 1,
            optimal_path.clone_id if optimal_path else "fallback",
            len(exploration_paths),
            self.budget_controller.get_budget_consumed() if self.budget_controller else 0
        )
        
        # Log comprehensive decision cycle summary
        self.logger.log_system_event("Decision cycle completed", decision_summary)
        
        return final_decision
    
    def _get_neural_network_prediction(self) -> Move:
        """Get neural network prediction for current game state."""
        if not self.neural_network or not self.feature_extractor:
            # Fallback to first possible move if NN not available
            current_board = self.master_game.get_current_board()
            possible_moves = GameLogic.get_possible_moves(current_board.direction)
            return possible_moves[0]
        
        # Extract features from current board
        current_board = self.master_game.get_current_board()
        features = self.feature_extractor.extract_features(current_board)
        
        # Get neural network prediction
        predicted_action_index, confidence = self.neural_network.predict_move(features)
        
        # Convert action index to move
        possible_moves = GameLogic.get_possible_moves(current_board.direction)
        nn_move = possible_moves[predicted_action_index]
        
        # Log neural network prediction details
        self.logger.log_system_event("Neural network prediction details", {
            "predicted_action": nn_move.action.value,
            "confidence": f"{confidence:.3f}",
            "action_index": predicted_action_index,
            "features_shape": str(features.shape)
        })
        
        return nn_move
    
    def execute_simulation_round(self, nn_prediction: Optional[Move] = None) -> List[ExplorationPath]:
        """
        Execute one round of tree search exploration.
        
        This method implements the hybrid execution where all initial clones
        start with the neural network prediction (if provided), then diverge
        for their second moves.
        
        Args:
            nn_prediction: Optional neural network prediction to guide exploration
            
        Returns:
            List[ExplorationPath]: All completed exploration paths
        """
        if not self.state_manager or not self.budget_controller:
            # Return dummy path if components not initialized
            current_board = self.master_game.get_current_board()
            possible_moves = GameLogic.get_possible_moves(current_board.direction)
            
            dummy_path = ExplorationPath(
                moves=possible_moves[:1],
                cumulative_reward=0,
                clone_id="1S",
                parent_id=None,
                depth=1,
                is_complete=True
            )
            return [dummy_path]
        
        # Create initial clones from master board
        current_board = self.master_game.get_current_board()
        initial_clones = self.state_manager.create_initial_clones(current_board)
        
        # Initialize budget for this round
        self.budget_controller.start_new_round()
        
        completed_paths = []
        active_clones = initial_clones.copy()
        
        # Execute exploration rounds until budget exhausted or all clones terminated
        round_number = 1
        while active_clones and self.budget_controller.can_continue_exploration():
            self.logger.log_system_event(f"Executing exploration round {round_number}", {
                "active_clones": len(active_clones),
                "remaining_budget": self.budget_controller.get_remaining_budget()
            })
            
            next_round_clones = []
            
            # Execute moves for all active clones in this round
            for clone in active_clones:
                try:
                    if self.budget_controller.is_budget_exhausted():
                        # Allow current round to complete even if budget exhausted
                        self.logger.log_system_event("Budget exhausted, completing current round")
                    
                    # Determine move for this clone
                    if round_number == 1 and nn_prediction:
                        # First round: all clones use NN prediction
                        move_to_execute = nn_prediction
                        self.logger.log_system_event(f"Clone {clone.get_clone_id()} using NN prediction", {
                            "move": nn_prediction.action.value
                        })
                    else:
                        # Subsequent rounds or no NN: use clone's designated move
                        clone_moves = self._get_clone_moves(clone)
                        move_index = self._get_clone_move_index(clone.get_clone_id())
                        move_to_execute = clone_moves[move_index]
                    
                    # Execute the move with error handling
                    result = clone.execute_move(move_to_execute)
                    self.budget_controller.consume_move()
                    
                    # Log clone step
                    self.logger.log_clone_step(
                        clone.get_clone_id(),
                        result.outcome,
                        result.reward,
                        result.new_board.score
                    )
                    
                    if result.is_terminal:
                        # Clone terminated - add to completed paths
                        path = ExplorationPath(
                            moves=clone.get_path_from_root(),
                            cumulative_reward=clone.get_cumulative_reward(),
                            clone_id=clone.get_clone_id(),
                            parent_id=clone.get_parent_id(),
                            depth=clone.get_depth(),
                            is_complete=True
                        )
                        completed_paths.append(path)
                    else:
                        # Clone survived - create sub-clones for next round
                        if not self.budget_controller.is_budget_exhausted():
                            sub_clones = self.state_manager.create_sub_clones(clone)
                            next_round_clones.extend(sub_clones)
                        else:
                            # Budget exhausted - add surviving clone as completed path
                            path = ExplorationPath(
                                moves=clone.get_path_from_root(),
                                cumulative_reward=clone.get_cumulative_reward(),
                                clone_id=clone.get_clone_id(),
                                parent_id=clone.get_parent_id(),
                                depth=clone.get_depth(),
                                is_complete=True
                            )
                            completed_paths.append(path)
                
                except Exception as e:
                    # Handle clone failure with error handler
                    recovery_result = self.error_handler.handle_clone_failure(
                        clone.get_clone_id(), e, "move_execution"
                    )
                    
                    if recovery_result.success:
                        # Clone failure isolated - continue with remaining clones
                        self.logger.log_system_event("Clone failure handled", {
                            "failed_clone": clone.get_clone_id(),
                            "recovery_action": recovery_result.action_taken.value,
                            "message": recovery_result.message
                        })
                        
                        # Add failed clone as terminated path with penalty
                        failed_path = ExplorationPath(
                            moves=clone.get_path_from_root(),
                            cumulative_reward=clone.get_cumulative_reward() - 100,  # Penalty for failure
                            clone_id=clone.get_clone_id(),
                            parent_id=clone.get_parent_id(),
                            depth=clone.get_depth(),
                            is_complete=True
                        )
                        completed_paths.append(failed_path)
                    else:
                        # Unrecoverable clone failure - log and continue
                        self.logger.log_error("HydraMgr", f"Unrecoverable clone failure: {clone.get_clone_id()}")
                        continue
            
            # Prepare for next round
            active_clones = next_round_clones
            round_number += 1
            
            # Start new round in budget controller
            if active_clones:
                self.budget_controller.start_new_round()
        
        # Add any remaining active clones as completed paths
        for clone in active_clones:
            path = ExplorationPath(
                moves=clone.get_path_from_root(),
                cumulative_reward=clone.get_cumulative_reward(),
                clone_id=clone.get_clone_id(),
                parent_id=clone.get_parent_id(),
                depth=clone.get_depth(),
                is_complete=True
            )
            completed_paths.append(path)
        
        self.logger.log_system_event("Tree search exploration completed", {
            "total_paths": len(completed_paths),
            "rounds_executed": round_number - 1,
            "budget_consumed": self.budget_controller.get_budget_consumed(),
            "exploration_efficiency": self._calculate_exploration_efficiency(completed_paths, round_number - 1)
        })
        
        return completed_paths
    
    def _get_clone_moves(self, clone) -> List[Move]:
        """Get possible moves for a clone based on its current board state."""
        current_board = clone.get_current_board()
        return GameLogic.get_possible_moves(current_board.direction)
    
    def _get_clone_move_index(self, clone_id: str) -> int:
        """
        Get the move index for a clone based on its ID pattern.
        
        Args:
            clone_id: The clone ID (e.g., "1", "2L", "3RS")
            
        Returns:
            int: Move index (0=Left, 1=Straight, 2=Right)
        """
        # For initial clones: "1"->Left, "2"->Straight, "3"->Right
        if clone_id.isdigit():
            clone_num = int(clone_id)
            return (clone_num - 1) % 3  # 1->0, 2->1, 3->2
        
        # For sub-clones, use the last character: L->0, S->1, R->2
        last_char = clone_id[-1]
        if last_char == 'L':
            return 0  # Left turn
        elif last_char == 'S':
            return 1  # Straight
        elif last_char == 'R':
            return 2  # Right turn
        else:
            return 1  # Default to straight
    
    def evaluate_exploration_paths(self, paths: List[ExplorationPath]) -> ExplorationPath:
        """
        Evaluate all exploration paths and select the optimal one.
        
        When multiple paths have the same highest reward, selects the path with
        the fewest moves to promote efficiency (Requirement 5.3).
        
        Args:
            paths: List of completed exploration paths
            
        Returns:
            ExplorationPath: The optimal path
        """
        if not paths:
            raise ValueError("No exploration paths to evaluate")
        
        # Find the highest cumulative reward
        max_reward = max(p.cumulative_reward for p in paths)
        
        # Get all paths with the highest reward
        best_paths = [p for p in paths if p.cumulative_reward == max_reward]
        
        # Among tied paths, select the one with fewest moves (efficiency-based selection)
        optimal_path = min(best_paths, key=lambda p: len(p.moves))
        
        # Log tie-breaking information if multiple paths had the same reward
        tie_breaking_info = {}
        if len(best_paths) > 1:
            tie_breaking_info = {
                "tied_paths_count": len(best_paths),
                "tied_clone_ids": [p.clone_id for p in best_paths],
                "path_lengths": [len(p.moves) for p in best_paths],
                "selected_for_efficiency": True,
                "selected_path_length": len(optimal_path.moves)
            }
            
            self.logger.log_system_event("Efficiency-based tie-breaking applied", tie_breaking_info)
        
        self.logger.log_system_event("Path evaluation completed", {
            "optimal_clone": optimal_path.clone_id,
            "optimal_reward": optimal_path.cumulative_reward,
            "optimal_path_length": len(optimal_path.moves),
            "total_paths": len(paths),
            **tie_breaking_info
        })
        
        return optimal_path
    
    def apply_move_to_master(self, move: Move) -> None:
        """
        Apply the selected move to the master game.
        
        Args:
            move: The move to apply
        """
        if not move:
            self.logger.log_warning("HydraMgr", "No move to apply to master game")
            return
        
        result = self.master_game.apply_move(move)
        
        self.logger.log_master_move(
            move.action.value,
            result.new_board.score
        )
    
    def reset_exploration_tree(self) -> None:
        """Reset the exploration tree for the next decision cycle."""
        if self.budget_controller:
            # Log budget summary before reset
            self.budget_controller.log_budget_summary()
            self.budget_controller.reset_budget()
        
        if self.state_manager:
            # Log tree exploration summary before destruction
            self.state_manager.log_tree_exploration_summary()
            self.state_manager.destroy_exploration_tree()
        
        self.logger.log_system_event("Exploration tree reset with comprehensive logging")
    
    def _train_neural_network(self, nn_prediction: Move, optimal_move: Move) -> None:
        """Train neural network based on tree search results."""
        if not self.oracle_trainer or not self.feature_extractor:
            return
        
        # Extract features from current board
        current_board = self.master_game.get_current_board()
        features = self.feature_extractor.extract_features(current_board)
        current_score = current_board.score
        
        # Compare predictions and generate training sample
        predictions_match = self.oracle_trainer.compare_predictions(nn_prediction, optimal_move, current_score)
        training_sample = self.oracle_trainer.generate_training_sample(
            features, nn_prediction, optimal_move, current_score
        )
        
        # Log oracle comparison details
        self.logger.log_system_event("Oracle comparison details", {
            "nn_prediction": nn_prediction.action.value,
            "optimal_move": optimal_move.action.value,
            "predictions_match": predictions_match,
            "decision": optimal_move.action.value,  # Always use tree search result
            "training_sample_generated": training_sample.was_nn_wrong,
            "pending_samples": len(self.oracle_trainer.training_samples),
            "current_score": current_score
        })
        
        # Update network if there are enough samples
        if len(self.oracle_trainer.training_samples) >= self.oracle_trainer.batch_size:
            self.oracle_trainer.update_network()
            
            # Log training update
            stats = self.oracle_trainer.get_training_statistics()
            self.logger.log_training_update(
                stats['overall_accuracy'],
                stats['total_training_samples']
            )
            
            # Log detailed training statistics
            self.logger.log_system_event("Neural network training update", {
                "overall_accuracy": f"{stats['overall_accuracy']:.3f}",
                "recent_accuracy": f"{stats['recent_accuracy']:.3f}",
                "training_updates": stats['training_updates'],
                "total_samples": stats['total_training_samples'],
                "batch_size": stats['batch_size']
            })
    
    def _is_simulation_complete(self) -> bool:
        """Check if the simulation should terminate."""
        if not self.master_game:
            return True
        
        return GameLogic.is_game_over(self.master_game.get_current_board(), 
                                     self.game_config.max_moves_multiplier)
    
    def _get_tree_metrics(self) -> TreeMetrics:
        """Get tree exploration metrics."""
        return TreeMetrics(
            total_clones_created=0,  # Placeholder
            max_depth_reached=0,     # Placeholder
            budget_consumed=0,       # Placeholder
            paths_evaluated=0,       # Placeholder
            optimal_path=None        # Placeholder
        )
    
    def _get_nn_accuracy(self) -> Optional[float]:
        """Get neural network accuracy if available."""
        if self.oracle_trainer:
            return self.oracle_trainer.get_prediction_accuracy()
        return None
    
    def _create_decision_cycle_summary(self, optimal_path, exploration_paths, nn_prediction, final_decision) -> dict:
        """Create comprehensive decision cycle summary."""
        # Budget metrics
        budget_metrics = self.budget_controller.get_budget_utilization_patterns() if self.budget_controller else {}
        
        # Tree metrics
        tree_metrics = self.state_manager.get_tree_statistics() if self.state_manager else {}
        
        # Path analysis
        path_rewards = [p.cumulative_reward for p in exploration_paths]
        path_depths = [p.depth for p in exploration_paths]
        
        # Neural network metrics
        nn_metrics = {}
        if self.simulation_config.nn_enabled and self.oracle_trainer:
            nn_stats = self.oracle_trainer.get_training_statistics()
            nn_metrics = {
                "nn_enabled": True,
                "nn_prediction": nn_prediction.action.value if nn_prediction else None,
                "final_decision": final_decision.action.value,
                "nn_accuracy": nn_stats.get('overall_accuracy', 0),
                "recent_accuracy": nn_stats.get('recent_accuracy', 0),
                "training_samples": nn_stats.get('total_training_samples', 0),
                "predictions_match": nn_prediction.action == final_decision.action if nn_prediction else False
            }
        else:
            nn_metrics = {"nn_enabled": False}
        
        return {
            "cycle_number": self.decision_cycle_count + 1,
            "optimal_clone": optimal_path.clone_id if optimal_path else "fallback",
            "optimal_reward": optimal_path.cumulative_reward if optimal_path else 0,
            "total_paths": len(exploration_paths),
            "path_rewards": {"min": min(path_rewards) if path_rewards else 0,
                           "max": max(path_rewards) if path_rewards else 0,
                           "avg": sum(path_rewards) / len(path_rewards) if path_rewards else 0},
            "path_depths": {"min": min(path_depths) if path_depths else 0,
                          "max": max(path_depths) if path_depths else 0,
                          "avg": sum(path_depths) / len(path_depths) if path_depths else 0},
            "budget_utilization": f"{budget_metrics.get('current_utilization_rate', 0):.1f}%",
            "rounds_executed": budget_metrics.get('rounds_completed', 0),
            "tree_max_depth": tree_metrics.get('max_depth', 0),
            "tree_survival_rate": f"{tree_metrics.get('survival_rate', 0):.1f}%",
            **nn_metrics
        }
    
    def _calculate_exploration_efficiency(self, paths, rounds_executed) -> dict:
        """Calculate exploration efficiency metrics."""
        if not paths:
            return {"efficiency": 0, "paths_per_round": 0, "reward_efficiency": 0}
        
        total_reward = sum(p.cumulative_reward for p in paths)
        avg_reward = total_reward / len(paths)
        paths_per_round = len(paths) / max(1, rounds_executed)
        
        # Calculate efficiency as reward per path per round
        efficiency = avg_reward * paths_per_round if rounds_executed > 0 else 0
        
        return {
            "efficiency": efficiency,
            "paths_per_round": paths_per_round,
            "avg_reward_per_path": avg_reward,
            "total_reward": total_reward
        }
    
    def log_comprehensive_game_summary(self) -> None:
        """Log comprehensive game summary with all metrics."""
        if not self.master_game:
            return
        
        # Get comprehensive statistics from all components
        game_stats = self.master_game.get_game_statistics()
        budget_patterns = self.budget_controller.get_budget_utilization_patterns() if self.budget_controller else {}
        tree_efficiency = self.state_manager.get_tree_exploration_efficiency() if self.state_manager else {}
        nn_stats = self.oracle_trainer.get_training_statistics() if self.oracle_trainer else {}
        
        # Create comprehensive summary
        comprehensive_summary = {
            "simulation_overview": {
                "total_decision_cycles": self.decision_cycle_count,
                "total_moves": self.total_moves,
                "final_score": game_stats.get('current_score', 0),
                "game_completed": game_stats.get('is_terminal', False)
            },
            "game_performance": {
                "food_eaten": game_stats.get('food_eaten', 0),
                "max_score_achieved": game_stats.get('max_score', 0),
                "final_snake_length": game_stats.get('current_snake_length', 0),
                "game_efficiency": f"{game_stats.get('food_eaten', 0) / max(1, game_stats.get('moves_played', 1)) * 100:.1f}%"
            },
            "budget_efficiency": {
                "avg_budget_utilization": f"{budget_patterns.get('current_utilization_rate', 0):.1f}%",
                "avg_moves_per_round": budget_patterns.get('efficiency_metrics', {}).get('avg_moves_per_round', 0),
                "peak_round_size": budget_patterns.get('efficiency_metrics', {}).get('peak_round_size', 0),
                "total_rounds_across_cycles": budget_patterns.get('rounds_completed', 0)
            },
            "tree_exploration": {
                "generations_completed": tree_efficiency.get('generations_completed', 0),
                "avg_clones_per_generation": tree_efficiency.get('avg_clones_per_generation', 0),
                "avg_max_depth": tree_efficiency.get('avg_max_depth', 0),
                "avg_survival_rate": f"{tree_efficiency.get('avg_survival_rate', 0):.1f}%",
                "exploration_consistency": f"{tree_efficiency.get('exploration_consistency', 0):.1f}%"
            },
            "neural_network": {
                "enabled": self.simulation_config.nn_enabled,
                "overall_accuracy": f"{nn_stats.get('overall_accuracy', 0):.3f}" if nn_stats else "N/A",
                "recent_accuracy": f"{nn_stats.get('recent_accuracy', 0):.3f}" if nn_stats else "N/A",
                "total_predictions": nn_stats.get('total_predictions', 0),
                "training_updates": nn_stats.get('training_updates', 0),
                "training_samples": nn_stats.get('total_training_samples', 0)
            }
        }
        
        # Log the comprehensive summary
        self.logger.log_system_event("COMPREHENSIVE GAME SUMMARY", comprehensive_summary)
        
        # Also log individual component summaries
        if self.master_game:
            self.master_game.log_game_summary()
        
        if self.budget_controller:
            self.budget_controller.log_budget_summary()
        
        if self.state_manager:
            self.state_manager.log_tree_exploration_summary()
    
    def _verify_component_initialization(self) -> None:
        """Verify that all required components are properly initialized."""
        required_components = {
            'master_game': self.master_game is not None,
            'budget_controller': self.budget_controller is not None,
            'state_manager': self.state_manager is not None
        }
        
        if self.simulation_config.nn_enabled:
            required_components.update({
                'neural_network': self.neural_network is not None,
                'feature_extractor': self.feature_extractor is not None,
                'oracle_trainer': self.oracle_trainer is not None
            })
        
        missing_components = [name for name, initialized in required_components.items() if not initialized]
        
        if missing_components:
            error_msg = f"Failed to initialize components: {missing_components}"
            self.logger.log_error("HydraMgr", error_msg)
            raise RuntimeError(error_msg)
        
        self.logger.log_system_event("Component initialization verified", {
            "initialized_components": list(required_components.keys()),
            "all_components_ready": True
        })
    
    def _validate_system_state(self) -> bool:
        """
        Validate the integrity of the entire system state.
        
        Returns:
            bool: True if system state is valid, False if corruption detected
        """
        try:
            # Validate master game state
            if self.master_game:
                current_board = self.master_game.get_current_board()
                if not self.error_handler.validate_game_board_integrity(current_board):
                    return False
            
            # Validate budget controller state
            if self.budget_controller:
                expected_consumed = self.total_moves  # Simple validation
                if not self.error_handler.validate_budget_consistency(self.budget_controller, expected_consumed):
                    return False
            
            # Validate state manager tree structure
            if self.state_manager:
                if not self.state_manager.validate_tree_structure():
                    self.error_handler.handle_state_corruption(
                        "state_manager", 
                        self.state_manager.get_active_clones(),
                        "Invalid tree structure detected"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, "hydra_mgr", "system_state_validation")
            return False