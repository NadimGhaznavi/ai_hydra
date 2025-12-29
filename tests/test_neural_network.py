"""
Property-based tests for neural network components.

This module contains property-based tests that validate the correctness
of feature extraction and neural network functionality.
"""

import random
import torch
from hypothesis import given, strategies as st
import pytest

from ai_hydra.models import GameBoard, Position, Direction
from ai_hydra.feature_extractor import FeatureExtractor
from ai_hydra.neural_network import SnakeNet


def game_boards():
    """Generate random GameBoard instances for property testing."""
    return st.builds(
        GameBoard,
        snake_head=st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
        snake_body=st.lists(
            st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
            min_size=0, max_size=10
        ).map(tuple),
        direction=st.sampled_from([
            Direction.up(), Direction.down(), Direction.left(), Direction.right()
        ]),
        food_position=st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
        score=st.integers(0, 1000),
        random_state=st.builds(random.Random, st.integers(0, 2**32-1)),
        grid_size=st.just((20, 20))
    )


@given(board=game_boards())
def test_feature_vector_consistency(board):
    """
    Feature: ai-hydra, Property: Feature vector consistency
    
    For any GameBoard, identical GameBoards should produce identical feature vectors.
    This test validates that the feature extraction is deterministic and consistent.
    
    **Validates: Requirements 11.2**
    """
    extractor = FeatureExtractor()
    
    # Extract features from the original board
    features1 = extractor.extract_features(board)
    
    # Clone the board and extract features again
    cloned_board = board.clone()
    features2 = extractor.extract_features(cloned_board)
    
    # Features should be identical
    assert torch.equal(features1, features2), "Identical GameBoards should produce identical feature vectors"
    
    # Extract features multiple times from the same board
    features3 = extractor.extract_features(board)
    assert torch.equal(features1, features3), "Multiple extractions from same board should be identical"
    
    # Verify feature vector properties
    assert features1.shape == (19,), f"Feature vector should have 19 dimensions, got {features1.shape}"
    assert features1.dtype == torch.float32, f"Feature vector should be float32, got {features1.dtype}"
    
    # Verify feature ranges
    features_list = features1.tolist()
    
    # First 10 features are boolean (collision and direction flags)
    for i in range(10):
        assert features_list[i] in [0.0, 1.0], f"Boolean feature {i} should be 0.0 or 1.0, got {features_list[i]}"
    
    # Food features (indices 10-11) should be in [-1.0, 1.0] range
    assert -1.0 <= features_list[10] <= 1.0, f"Food dx feature should be in [-1.0, 1.0], got {features_list[10]}"
    assert -1.0 <= features_list[11] <= 1.0, f"Food dy feature should be in [-1.0, 1.0], got {features_list[11]}"
    
    # Snake length features (indices 12-18) should be boolean
    for i in range(12, 19):
        assert features_list[i] in [0.0, 1.0], f"Snake length bit {i-12} should be 0.0 or 1.0, got {features_list[i]}"


@given(board=game_boards())
def test_collision_features_accuracy(board):
    """
    Test that collision features accurately reflect game state.
    
    This test verifies that the collision detection features correctly
    identify potential collisions in all three directions.
    """
    extractor = FeatureExtractor()
    
    # Get collision features
    snake_collisions = extractor.get_collision_features(board, check_snake=True)
    wall_collisions = extractor.get_collision_features(board, check_snake=False)
    
    # Both should return exactly 3 boolean values
    assert len(snake_collisions) == 3, "Snake collision features should have 3 elements"
    assert len(wall_collisions) == 3, "Wall collision features should have 3 elements"
    
    # All values should be boolean
    for collision in snake_collisions + wall_collisions:
        assert isinstance(collision, bool), "Collision features should be boolean"
    
    # Verify collision logic by checking specific positions
    current_dir = board.get_direction()
    head_pos = board.get_snake_head()
    
    # Calculate expected positions
    straight_pos = Position(head_pos.x + current_dir.dx, head_pos.y + current_dir.dy)
    left_dir = current_dir.turn_left()
    left_pos = Position(head_pos.x + left_dir.dx, head_pos.y + left_dir.dy)
    right_dir = current_dir.turn_right()
    right_pos = Position(head_pos.x + right_dir.dx, head_pos.y + right_dir.dy)
    
    positions = [straight_pos, left_pos, right_pos]
    
    # Verify wall collision logic
    for i, pos in enumerate(positions):
        expected_wall_collision = not board.is_position_within_bounds(pos)
        assert wall_collisions[i] == expected_wall_collision, \
            f"Wall collision for direction {i} should be {expected_wall_collision}, got {wall_collisions[i]}"
    
    # Verify snake collision logic
    for i, pos in enumerate(positions):
        expected_snake_collision = pos in board.get_snake_body()
        assert snake_collisions[i] == expected_snake_collision, \
            f"Snake collision for direction {i} should be {expected_snake_collision}, got {snake_collisions[i]}"


@given(board=game_boards())
def test_direction_features_accuracy(board):
    """
    Test that direction features accurately represent current direction.
    
    This test verifies that exactly one direction flag is True and
    it corresponds to the current direction.
    """
    extractor = FeatureExtractor()
    direction_features = extractor.get_direction_features(board)
    
    # Should have exactly 4 features
    assert len(direction_features) == 4, "Direction features should have 4 elements"
    
    # All should be boolean
    for feature in direction_features:
        assert isinstance(feature, bool), "Direction features should be boolean"
    
    # Exactly one should be True
    true_count = sum(direction_features)
    assert true_count == 1, f"Exactly one direction should be True, got {true_count}"
    
    # Verify the correct direction is marked as True
    current_dir = board.get_direction()
    expected_features = [
        current_dir == Direction.up(),
        current_dir == Direction.down(),
        current_dir == Direction.left(),
        current_dir == Direction.right()
    ]
    
    assert direction_features == expected_features, \
        f"Direction features {direction_features} don't match expected {expected_features}"


@given(board=game_boards())
def test_food_features_normalization(board):
    """
    Test that food features are properly normalized to [-1.0, 1.0] range.
    
    This test verifies that food relative position features are correctly
    calculated and normalized.
    """
    extractor = FeatureExtractor()
    food_features = extractor.get_food_features(board)
    
    # Should have exactly 2 features
    assert len(food_features) == 2, "Food features should have 2 elements"
    
    # Both should be floats in [-1.0, 1.0] range
    for feature in food_features:
        assert isinstance(feature, float), "Food features should be floats"
        assert -1.0 <= feature <= 1.0, f"Food feature {feature} should be in [-1.0, 1.0] range"
    
    # Verify calculation logic
    head_pos = board.get_snake_head()
    food_pos = board.get_food_position()
    grid_width, grid_height = board.get_grid_size()
    
    dx = food_pos.x - head_pos.x
    dy = food_pos.y - head_pos.y
    
    expected_dx = max(-1.0, min(1.0, dx / (grid_width / 2.0)))
    expected_dy = max(-1.0, min(1.0, dy / (grid_height / 2.0)))
    
    assert abs(food_features[0] - expected_dx) < 1e-6, \
        f"Food dx feature {food_features[0]} doesn't match expected {expected_dx}"
    assert abs(food_features[1] - expected_dy) < 1e-6, \
        f"Food dy feature {food_features[1]} doesn't match expected {expected_dy}"


@given(board=game_boards())
def test_snake_length_features_binary_representation(board):
    """
    Test that snake length features correctly represent length in binary.
    
    This test verifies that the 7-bit binary representation of snake length
    is accurate and can represent lengths up to 127.
    """
    extractor = FeatureExtractor()
    snake_features = extractor.get_snake_features(board)
    
    # Should have exactly 7 features
    assert len(snake_features) == 7, "Snake length features should have 7 elements"
    
    # All should be boolean
    for feature in snake_features:
        assert isinstance(feature, bool), "Snake length features should be boolean"
    
    # Verify binary representation
    actual_length = len(board.get_all_snake_positions())
    
    # Convert binary features back to integer
    reconstructed_length = 0
    for i, bit in enumerate(snake_features):
        if bit:
            reconstructed_length += 2 ** i
    
    assert reconstructed_length == actual_length, \
        f"Binary representation {snake_features} should equal length {actual_length}, got {reconstructed_length}"
    
    # Verify length is within 7-bit range
    assert 0 <= actual_length <= 127, f"Snake length {actual_length} should be within 7-bit range [0, 127]"


def test_neural_network_architecture():
    """
    Test that SnakeNet has the correct architecture and behavior.
    
    This test verifies the neural network structure and basic functionality.
    """
    # Create network with default parameters
    net = SnakeNet()
    
    # Verify architecture
    info = net.get_network_info()
    assert info['input_features'] == 19, "Input layer should have 19 features"
    assert info['hidden_size'] == 200, "Hidden layers should have 200 nodes"
    assert info['output_actions'] == 3, "Output layer should have 3 actions"
    assert info['architecture'] == '19 → 200 → 200 → 3', "Architecture string should match expected"
    
    # Test forward pass with random input
    test_input = torch.randn(1, 19)
    output = net.forward(test_input)
    
    # Verify output shape and properties
    assert output.shape == (1, 3), f"Output shape should be (1, 3), got {output.shape}"
    assert torch.allclose(output.sum(dim=1), torch.ones(1)), "Output probabilities should sum to 1"
    assert torch.all(output >= 0), "All output probabilities should be non-negative"
    assert torch.all(output <= 1), "All output probabilities should be <= 1"


@given(board=game_boards())
def test_neural_network_prediction_consistency(board):
    """
    Test that neural network predictions are consistent for identical inputs.
    
    This test verifies that the neural network produces consistent outputs
    for identical feature vectors.
    """
    extractor = FeatureExtractor()
    net = SnakeNet()
    
    # Extract features
    features = extractor.extract_features(board)
    
    # Make predictions multiple times
    action1, confidence1 = net.predict_move(features)
    action2, confidence2 = net.predict_move(features)
    
    # Predictions should be identical
    assert action1 == action2, "Predictions should be consistent for identical inputs"
    assert abs(confidence1 - confidence2) < 1e-6, "Confidence should be consistent for identical inputs"
    
    # Verify prediction properties
    assert 0 <= action1 <= 2, f"Predicted action {action1} should be in range [0, 2]"
    assert 0 <= confidence1 <= 1, f"Confidence {confidence1} should be in range [0, 1]"
    
    # Test probability output
    probabilities = net.get_move_probabilities(features)
    assert probabilities.shape == (3,), f"Probabilities shape should be (3,), got {probabilities.shape}"
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0)), "Probabilities should sum to 1"
    assert torch.all(probabilities >= 0), "All probabilities should be non-negative"
    
    # The predicted action should correspond to the highest probability
    max_prob_action = torch.argmax(probabilities).item()
    assert action1 == max_prob_action, "Predicted action should match highest probability action"
    assert abs(confidence1 - probabilities[action1].item()) < 1e-6, "Confidence should match probability"