"""
Tests for the OracleTrainer class.

This module tests the neural network training functionality that learns
from tree search results, including prediction comparison, training sample
generation, and network weight updates.
"""

import pytest
import torch
import torch.nn as nn
from ai_hydra.oracle_trainer import OracleTrainer, TrainingSample
from ai_hydra.neural_network import SnakeNet
from ai_hydra.models import Move, MoveAction, Direction


class TestOracleTrainer:
    """Test cases for the OracleTrainer class."""
    
    @pytest.fixture
    def neural_network(self):
        """Create a neural network for testing."""
        return SnakeNet()
    
    @pytest.fixture
    def oracle_trainer(self, neural_network):
        """Create an OracleTrainer instance for testing."""
        return OracleTrainer(neural_network, learning_rate=0.001, batch_size=4)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature tensor for testing."""
        return torch.randn(1, 19)  # 19 features as expected by SnakeNet
    
    @pytest.fixture
    def sample_moves(self):
        """Create sample moves for testing."""
        return {
            'left': Move(MoveAction.LEFT_TURN, Direction.left()),
            'straight': Move(MoveAction.STRAIGHT, Direction.up()),
            'right': Move(MoveAction.RIGHT_TURN, Direction.right())
        }
    
    def test_oracle_trainer_initialization(self, neural_network):
        """Test that OracleTrainer initializes correctly."""
        trainer = OracleTrainer(neural_network, learning_rate=0.01, batch_size=16)
        
        assert trainer.neural_network is neural_network
        assert trainer.learning_rate == 0.01
        assert trainer.batch_size == 16
        assert trainer.total_predictions == 0
        assert trainer.correct_predictions == 0
        assert len(trainer.training_samples) == 0
        assert len(trainer.recent_predictions) == 0
        assert trainer.training_updates == 0
        assert trainer.total_training_samples == 0
    
    def test_compare_predictions_correct(self, oracle_trainer, sample_moves):
        """Test prediction comparison when NN is correct."""
        nn_move = sample_moves['left']
        optimal_move = sample_moves['left']
        current_score = 10
        
        result = oracle_trainer.compare_predictions(nn_move, optimal_move, current_score)
        
        assert result is True
        assert oracle_trainer.total_predictions == 1
        assert oracle_trainer.correct_predictions == 1
        assert oracle_trainer.get_prediction_accuracy() == 1.0
        assert len(oracle_trainer.recent_predictions) == 1
        assert oracle_trainer.recent_predictions[0] is True
    
    def test_compare_predictions_incorrect(self, oracle_trainer, sample_moves):
        """Test prediction comparison when NN is incorrect."""
        nn_move = sample_moves['left']
        optimal_move = sample_moves['right']
        current_score = 15
        
        result = oracle_trainer.compare_predictions(nn_move, optimal_move, current_score)
        
        assert result is False
        assert oracle_trainer.total_predictions == 1
        assert oracle_trainer.correct_predictions == 0
        assert oracle_trainer.get_prediction_accuracy() == 0.0
        assert len(oracle_trainer.recent_predictions) == 1
        assert oracle_trainer.recent_predictions[0] is False
    
    def test_generate_training_sample_correct_prediction(self, oracle_trainer, sample_features, sample_moves):
        """Test training sample generation when NN prediction is correct."""
        nn_move = sample_moves['straight']
        optimal_move = sample_moves['straight']
        current_score = 20
        
        sample = oracle_trainer.generate_training_sample(sample_features, nn_move, optimal_move, current_score)
        
        assert isinstance(sample, TrainingSample)
        assert torch.equal(sample.features, sample_features.clone().detach())
        assert sample.optimal_action == 1  # Straight = index 1
        assert sample.nn_prediction == 1   # Straight = index 1
        assert sample.was_nn_wrong is False
        
        # Should not add to training samples since NN was correct
        assert len(oracle_trainer.training_samples) == 0
        assert oracle_trainer.total_training_samples == 0
    
    def test_generate_training_sample_incorrect_prediction(self, oracle_trainer, sample_features, sample_moves):
        """Test training sample generation when NN prediction is incorrect."""
        nn_move = sample_moves['left']      # NN predicted left (index 0)
        optimal_move = sample_moves['right'] # Optimal was right (index 2)
        current_score = 25
        
        sample = oracle_trainer.generate_training_sample(sample_features, nn_move, optimal_move, current_score)
        
        assert isinstance(sample, TrainingSample)
        assert torch.equal(sample.features, sample_features.clone().detach())
        assert sample.optimal_action == 2  # Right = index 2
        assert sample.nn_prediction == 0   # Left = index 0
        assert sample.was_nn_wrong is True
        
        # Should add to training samples since NN was wrong
        assert len(oracle_trainer.training_samples) == 1
        assert oracle_trainer.total_training_samples == 1
        assert oracle_trainer.training_samples[0] is sample
    
    def test_update_network_no_samples(self, oracle_trainer):
        """Test network update with no training samples."""
        initial_params = [p.clone() for p in oracle_trainer.neural_network.parameters()]
        
        oracle_trainer.update_network()
        
        # Parameters should remain unchanged
        for initial, current in zip(initial_params, oracle_trainer.neural_network.parameters()):
            assert torch.equal(initial, current)
        
        assert oracle_trainer.training_updates == 0
    
    def test_update_network_with_samples(self, oracle_trainer, sample_features, sample_moves):
        """Test network update with training samples."""
        # Generate some incorrect predictions to create training samples
        for i in range(5):
            features = torch.randn(1, 19)
            nn_move = sample_moves['left']
            optimal_move = sample_moves['right']
            oracle_trainer.generate_training_sample(features, nn_move, optimal_move, 30 + i)
        
        assert len(oracle_trainer.training_samples) == 5
        
        initial_params = [p.clone() for p in oracle_trainer.neural_network.parameters()]
        
        oracle_trainer.update_network()
        
        # Parameters should have changed
        params_changed = False
        for initial, current in zip(initial_params, oracle_trainer.neural_network.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Network parameters should have been updated"
        assert oracle_trainer.training_updates > 0
        assert len(oracle_trainer.training_samples) == 0  # Should be cleared after update
    
    def test_accuracy_tracking(self, oracle_trainer, sample_moves):
        """Test accuracy tracking over multiple predictions."""
        # Make some correct and incorrect predictions
        predictions = [
            (sample_moves['left'], sample_moves['left'], True),      # Correct
            (sample_moves['left'], sample_moves['right'], False),   # Incorrect
            (sample_moves['straight'], sample_moves['straight'], True), # Correct
            (sample_moves['right'], sample_moves['left'], False),   # Incorrect
            (sample_moves['right'], sample_moves['right'], True),   # Correct
        ]
        
        for nn_move, optimal_move, expected_correct in predictions:
            result = oracle_trainer.compare_predictions(nn_move, optimal_move, 40)
            assert result == expected_correct
        
        assert oracle_trainer.total_predictions == 5
        assert oracle_trainer.correct_predictions == 3
        assert oracle_trainer.get_prediction_accuracy() == 0.6
        assert oracle_trainer.get_recent_accuracy() == 0.6
        assert len(oracle_trainer.recent_predictions) == 5
    
    def test_recent_accuracy_window(self, oracle_trainer, sample_moves):
        """Test that recent accuracy maintains a sliding window."""
        # Fill up more than the max recent history (100)
        for i in range(150):
            # Alternate between correct and incorrect
            if i % 2 == 0:
                oracle_trainer.compare_predictions(sample_moves['left'], sample_moves['left'], 50 + i)
            else:
                oracle_trainer.compare_predictions(sample_moves['left'], sample_moves['right'], 50 + i)
        
        assert oracle_trainer.total_predictions == 150
        assert len(oracle_trainer.recent_predictions) == 100  # Should be capped at max_recent_history
        assert oracle_trainer.get_recent_accuracy() == 0.5  # Should be 50% (alternating pattern)
    
    def test_training_statistics(self, oracle_trainer, sample_features, sample_moves):
        """Test comprehensive training statistics."""
        # Generate some activity
        oracle_trainer.compare_predictions(sample_moves['left'], sample_moves['left'], 60)  # Correct
        oracle_trainer.compare_predictions(sample_moves['left'], sample_moves['right'], 65) # Incorrect
        oracle_trainer.generate_training_sample(sample_features, sample_moves['left'], sample_moves['right'], 65)
        
        stats = oracle_trainer.get_training_statistics()
        
        assert stats['total_predictions'] == 2
        assert stats['correct_predictions'] == 1
        assert stats['overall_accuracy'] == 0.5
        assert stats['recent_accuracy'] == 0.5
        assert stats['training_updates'] == 0  # No updates yet
        assert stats['total_training_samples'] == 1
        assert stats['pending_samples'] == 1
        assert stats['learning_rate'] == 0.001
        assert stats['batch_size'] == 4
    
    def test_reset_statistics(self, oracle_trainer, sample_features, sample_moves):
        """Test resetting all statistics."""
        # Generate some activity
        oracle_trainer.compare_predictions(sample_moves['left'], sample_moves['right'], 70)
        oracle_trainer.generate_training_sample(sample_features, sample_moves['left'], sample_moves['right'], 70)
        
        # Verify there's activity to reset
        assert oracle_trainer.total_predictions > 0
        assert len(oracle_trainer.training_samples) > 0
        
        oracle_trainer.reset_statistics()
        
        assert oracle_trainer.total_predictions == 0
        assert oracle_trainer.correct_predictions == 0
        assert len(oracle_trainer.recent_predictions) == 0
        assert oracle_trainer.training_updates == 0
        assert oracle_trainer.total_training_samples == 0
        assert len(oracle_trainer.training_samples) == 0
        assert oracle_trainer.get_prediction_accuracy() == 0.0
        assert oracle_trainer.get_recent_accuracy() == 0.0
    
    def test_move_to_action_index_conversion(self, oracle_trainer, sample_moves):
        """Test conversion between moves and action indices."""
        assert oracle_trainer._move_to_action_index(sample_moves['left']) == 0
        assert oracle_trainer._move_to_action_index(sample_moves['straight']) == 1
        assert oracle_trainer._move_to_action_index(sample_moves['right']) == 2
    
    def test_action_index_to_move_action_conversion(self, oracle_trainer):
        """Test conversion from action indices to move actions."""
        assert oracle_trainer._action_index_to_move_action(0) == MoveAction.LEFT_TURN
        assert oracle_trainer._action_index_to_move_action(1) == MoveAction.STRAIGHT
        assert oracle_trainer._action_index_to_move_action(2) == MoveAction.RIGHT_TURN
    
    def test_batch_training_with_multiple_samples(self, oracle_trainer, sample_moves):
        """Test batch training with multiple incorrect samples."""
        # Generate multiple training samples
        for i in range(10):
            features = torch.randn(1, 19)
            nn_move = sample_moves['left']
            optimal_move = sample_moves['right']
            oracle_trainer.generate_training_sample(features, nn_move, optimal_move, 80 + i)
        
        assert len(oracle_trainer.training_samples) == 10
        
        initial_updates = oracle_trainer.training_updates
        oracle_trainer.update_network()
        
        # Should have performed multiple batch updates (batch_size = 4, so 3 batches for 10 samples)
        assert oracle_trainer.training_updates > initial_updates
        assert len(oracle_trainer.training_samples) == 0  # Should be cleared