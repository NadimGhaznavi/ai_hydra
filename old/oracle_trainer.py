"""
Oracle trainer for neural network learning from tree search.

This module implements the OracleTrainer class which compares neural network
predictions with tree search results and trains the network when predictions
differ from the optimal tree search decision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .models import Move, MoveAction
from .neural_network import SnakeNet
from .logging_config import SimulationLogger
from .config import LoggingConfig


@dataclass
class TrainingSample:
    """Represents a training sample for the neural network."""
    features: torch.Tensor
    optimal_action: int  # 0=Left, 1=Straight, 2=Right
    nn_prediction: int
    was_nn_wrong: bool


class OracleTrainer:
    """
    Trains neural network using tree search results as oracle.
    
    This class compares neural network predictions with tree search optimal
    results and generates training data when the predictions differ. It maintains
    accuracy statistics and updates the network weights based on oracle feedback.
    """
    
    def __init__(self, neural_network: SnakeNet, learning_rate: float = 0.001, 
                 batch_size: int = 32, logging_config: Optional[LoggingConfig] = None):
        """
        Initialize the oracle trainer.
        
        Args:
            neural_network: The neural network to train
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training updates
            logging_config: Logging configuration (optional)
        """
        self.neural_network = neural_network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize logger
        self.logger = SimulationLogger("oracle_trainer", logging_config)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.neural_network.parameters(), 
            lr=self.learning_rate
        )
        self.loss_function = nn.CrossEntropyLoss()
        
        # Training data storage
        self.training_samples: List[TrainingSample] = []
        
        # Accuracy tracking
        self.total_predictions = 0
        self.correct_predictions = 0
        self.recent_predictions: List[bool] = []  # Last 100 predictions
        self.max_recent_history = 100
        
        # Training statistics
        self.training_updates = 0
        self.total_training_samples = 0
    
    def compare_predictions(self, nn_move: Move, optimal_move: Move, current_score: int) -> bool:
        """
        Compare neural network prediction with tree search optimal result.
        
        Args:
            nn_move: Move predicted by neural network
            optimal_move: Optimal move from tree search
            current_score: Current game score
            
        Returns:
            bool: True if predictions match, False otherwise
        """
        nn_correct = nn_move.action == optimal_move.action
        
        # Update accuracy tracking
        self.total_predictions += 1
        if nn_correct:
            self.correct_predictions += 1
        
        # Update recent predictions history
        self.recent_predictions.append(nn_correct)
        if len(self.recent_predictions) > self.max_recent_history:
            self.recent_predictions.pop(0)
        
        # Log the comparison result
        self.logger.log_oracle_decision(
            nn_move.action.value,
            optimal_move.action.value,
            optimal_move.action.value if not nn_correct else nn_move.action.value,
            current_score
        )
        
        return nn_correct
    
    def generate_training_sample(self, features: torch.Tensor, nn_move: Move, 
                               optimal_move: Move, current_score: int) -> TrainingSample:
        """
        Generate a training sample when NN prediction differs from optimal.
        
        Args:
            features: Game state features used for prediction
            nn_move: Move predicted by neural network
            optimal_move: Optimal move from tree search
            current_score: Current game score
            
        Returns:
            TrainingSample: Training sample for network update
        """
        # Convert moves to action indices
        nn_action = self._move_to_action_index(nn_move)
        optimal_action = self._move_to_action_index(optimal_move)
        
        was_nn_wrong = nn_action != optimal_action
        
        sample = TrainingSample(
            features=features.clone().detach(),
            optimal_action=optimal_action,
            nn_prediction=nn_action,
            was_nn_wrong=was_nn_wrong
        )
        
        # Store sample for batch training
        if was_nn_wrong:
            self.training_samples.append(sample)
            self.total_training_samples += 1
            
        # Log training sample generation
        self.logger.log_training_sample(was_nn_wrong, current_score)
        
        return sample
    
    def update_network(self, training_samples: Optional[List[TrainingSample]] = None) -> None:
        """
        Update neural network weights based on training samples.
        
        Args:
            training_samples: Optional list of samples to train on.
                            If None, uses accumulated samples.
        """
        if training_samples is None:
            training_samples = self.training_samples
        
        if not training_samples:
            return  # No samples to train on
        
        # Filter to only incorrect predictions
        incorrect_samples = [s for s in training_samples if s.was_nn_wrong]
        
        if not incorrect_samples:
            return  # No incorrect predictions to learn from
        
        # Update total training samples counter
        if training_samples is not self.training_samples:
            # Only count new samples if they were passed in externally
            self.total_training_samples += len(incorrect_samples)
        
        # Prepare batch data
        batch_features = torch.stack([s.features.squeeze() for s in incorrect_samples])  # Remove extra dimension
        batch_targets = torch.tensor([s.optimal_action for s in incorrect_samples], 
                                   dtype=torch.long)
        
        # Process in batches
        for i in range(0, len(incorrect_samples), self.batch_size):
            batch_end = min(i + self.batch_size, len(incorrect_samples))
            
            features_batch = batch_features[i:batch_end]
            targets_batch = batch_targets[i:batch_end]
            
            # Forward pass - get raw logits (before softmax)
            self.neural_network.train()
            
            # Get raw logits by calling the network layers directly
            x = features_batch
            x = self.neural_network.relu(self.neural_network.input_layer(x))
            x = self.neural_network.relu(self.neural_network.hidden_layer(x))
            logits = self.neural_network.output_layer(x)  # Raw logits, no softmax
            
            # Calculate loss using raw logits
            loss = self.loss_function(logits, targets_batch)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.training_updates += 1
        
        # Log training update
        if incorrect_samples:
            self.logger.log_training_update(
                self.get_prediction_accuracy(),
                len(incorrect_samples)
            )
        
        # Clear processed samples
        if training_samples is self.training_samples:
            self.training_samples.clear()
    
    def get_prediction_accuracy(self) -> float:
        """
        Get current prediction accuracy.
        
        Returns:
            float: Accuracy as a percentage (0.0 to 1.0)
        """
        if self.total_predictions == 0:
            return 0.0
        
        return self.correct_predictions / self.total_predictions
    
    def get_recent_accuracy(self) -> float:
        """
        Get accuracy for recent predictions (last 100).
        
        Returns:
            float: Recent accuracy as a percentage (0.0 to 1.0)
        """
        if not self.recent_predictions:
            return 0.0
        
        correct_recent = sum(self.recent_predictions)
        return correct_recent / len(self.recent_predictions)
    
    def get_training_statistics(self) -> dict:
        """
        Get comprehensive training statistics.
        
        Returns:
            dict: Training statistics including accuracy, updates, and samples
        """
        return {
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'overall_accuracy': self.get_prediction_accuracy(),
            'recent_accuracy': self.get_recent_accuracy(),
            'training_updates': self.training_updates,
            'total_training_samples': self.total_training_samples,
            'pending_samples': len(self.training_samples),
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
    
    def reset_statistics(self) -> None:
        """Reset all training and accuracy statistics."""
        self.total_predictions = 0
        self.correct_predictions = 0
        self.recent_predictions.clear()
        self.training_updates = 0
        self.total_training_samples = 0
        self.training_samples.clear()
    
    def _move_to_action_index(self, move: Move) -> int:
        """
        Convert a Move to an action index for neural network.
        
        Args:
            move: The move to convert
            
        Returns:
            int: Action index (0=Left, 1=Straight, 2=Right)
        """
        action_mapping = {
            MoveAction.LEFT_TURN: 0,
            MoveAction.STRAIGHT: 1,
            MoveAction.RIGHT_TURN: 2
        }
        
        return action_mapping[move.action]
    
    def _action_index_to_move_action(self, action_index: int) -> MoveAction:
        """
        Convert an action index to a MoveAction.
        
        Args:
            action_index: The action index (0=Left, 1=Straight, 2=Right)
            
        Returns:
            MoveAction: The corresponding move action
        """
        index_mapping = {
            0: MoveAction.LEFT_TURN,
            1: MoveAction.STRAIGHT,
            2: MoveAction.RIGHT_TURN
        }
        
        return index_mapping[action_index]
    
    def save_model(self, filepath: str) -> None:
        """
        Save the neural network model and training state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_statistics': self.get_training_statistics()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved neural network model and training state.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath)
        
        self.neural_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore statistics if available
        if 'training_statistics' in checkpoint:
            stats = checkpoint['training_statistics']
            self.total_predictions = stats.get('total_predictions', 0)
            self.correct_predictions = stats.get('correct_predictions', 0)
            self.training_updates = stats.get('training_updates', 0)
            self.total_training_samples = stats.get('total_training_samples', 0)