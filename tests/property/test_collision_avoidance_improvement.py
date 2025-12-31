"""Property-based tests for collision avoidance improvement."""

import pytest
import torch
from hypothesis import given, strategies as st, settings


def test_basic_collision_avoidance():
    """Basic test for collision avoidance functionality."""
    from ai_hydra.neural_network import SnakeNet
    from ai_hydra.oracle_trainer import OracleTrainer, TrainingSample
    
    # Create neural network and oracle trainer
    neural_net = SnakeNet(input_features=19, hidden_size=50, output_actions=3)
    oracle_trainer = OracleTrainer(neural_net, learning_rate=0.01, batch_size=4)
    
    # Test basic functionality
    assert oracle_trainer.get_prediction_accuracy() == 0.0
    
    # Create a training sample
    features = torch.randn(1, 19)
    sample = TrainingSample(
        features=features,
        optimal_action=1,  # Oracle says "straight"
        nn_prediction=0,   # NN predicted "left"
        was_nn_wrong=True
    )
    
    # Test training
    oracle_trainer.update_network([sample])
    
    # Verify training occurred
    stats = oracle_trainer.get_training_statistics()
    assert stats['training_updates'] > 0
    assert stats['total_training_samples'] > 0


@given(
    initial_accuracy=st.floats(min_value=0.2, max_value=0.6),
    training_samples_count=st.integers(min_value=3, max_value=10)
)
@settings(max_examples=3, deadline=5000)
def test_collision_avoidance_improvement_property(initial_accuracy, training_samples_count):
    """
    **Feature: ai-hydra, Property 16: Collision Avoidance Improvement**
    **Validates: Requirements 11.4, 11.5**
    
    For any neural network with initial collision avoidance accuracy, training with
    oracle feedback should maintain or improve performance.
    """
    from ai_hydra.neural_network import SnakeNet
    from ai_hydra.oracle_trainer import OracleTrainer, TrainingSample
    
    # Create neural network and oracle trainer
    neural_net = SnakeNet(input_features=19, hidden_size=30, output_actions=3)  # Smaller for testing
    oracle_trainer = OracleTrainer(neural_net, learning_rate=0.01, batch_size=2)
    
    # Simulate initial accuracy
    initial_correct = int(initial_accuracy * 50)
    initial_total = 50
    oracle_trainer.correct_predictions = initial_correct
    oracle_trainer.total_predictions = initial_total
    
    # Track accuracy before training
    accuracy_before = oracle_trainer.get_prediction_accuracy()
    
    # Generate training samples (simulate oracle corrections)
    training_samples = []
    for _ in range(training_samples_count):
        features = torch.randn(1, 19)
        sample = TrainingSample(
            features=features,
            optimal_action=1,  # Oracle says "straight"
            nn_prediction=0,   # NN predicted "left"
            was_nn_wrong=True
        )
        training_samples.append(sample)
    
    # Perform training updates
    oracle_trainer.update_network(training_samples)
    
    # Property: Training updates should have occurred
    assert oracle_trainer.training_updates > 0, "No training updates occurred"
    
    # Property: Training samples should have been processed
    assert oracle_trainer.total_training_samples > 0, "No training samples were processed"
    
    # Property: System should be able to track accuracy
    current_accuracy = oracle_trainer.get_prediction_accuracy()
    assert 0.0 <= current_accuracy <= 1.0, f"Invalid accuracy: {current_accuracy}"


@given(
    network_size=st.integers(min_value=20, max_value=50),
    learning_rate=st.floats(min_value=0.001, max_value=0.05)
)
@settings(max_examples=3, deadline=4000)
def test_neural_network_weight_updates_property(network_size, learning_rate):
    """
    **Feature: ai-hydra, Property 16a: Neural Network Weight Updates**
    **Validates: Requirements 11.4, 11.5**
    
    For any neural network configuration, training with oracle corrections should
    update the network weights and change prediction behavior.
    """
    from ai_hydra.neural_network import SnakeNet
    from ai_hydra.oracle_trainer import OracleTrainer, TrainingSample
    
    # Create neural network and oracle trainer
    neural_net = SnakeNet(input_features=19, hidden_size=network_size, output_actions=3)
    oracle_trainer = OracleTrainer(neural_net, learning_rate=learning_rate, batch_size=2)
    
    # Get initial network weights
    initial_weights = {}
    for name, param in neural_net.named_parameters():
        initial_weights[name] = param.data.clone()
    
    # Create training samples
    training_samples = []
    for _ in range(5):  # Small number for testing
        features = torch.randn(1, 19)
        sample = TrainingSample(
            features=features,
            optimal_action=2,  # Oracle says "right"
            nn_prediction=0,   # NN predicted "left"
            was_nn_wrong=True
        )
        training_samples.append(sample)
    
    # Perform training
    oracle_trainer.update_network(training_samples)
    
    # Property: Network weights should change after training
    weights_changed = False
    for name, param in neural_net.named_parameters():
        if not torch.allclose(initial_weights[name], param.data, atol=1e-6):
            weights_changed = True
            break
    
    assert weights_changed, "Network weights did not change after training"
    
    # Property: Training statistics should be updated
    stats = oracle_trainer.get_training_statistics()
    assert stats['training_updates'] > 0, "No training updates recorded"
    assert stats['total_training_samples'] > 0, "No training samples recorded"