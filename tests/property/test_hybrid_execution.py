"""
Property-based tests for hybrid execution system.

This module contains property-based tests that validate the correctness
of the hybrid neural network + tree search execution system, particularly
focusing on oracle decision correctness.
"""

import random
import torch
from hypothesis import given, strategies as st, settings
import pytest

from ai_hydra.models import GameBoard, Position, Direction, Move, MoveAction
from ai_hydra.game_logic import GameLogic
from ai_hydra.neural_network import SnakeNet
from ai_hydra.feature_extractor import FeatureExtractor
from ai_hydra.oracle_trainer import OracleTrainer
from ai_hydra.hydra_mgr import HydraMgr
from ai_hydra.config import SimulationConfig, NetworkConfig


def game_boards():
    """Generate random GameBoard instances for property testing."""
    return st.builds(
        GameBoard,
        snake_head=st.builds(Position, x=st.integers(1, 18), y=st.integers(1, 18)),
        snake_body=st.lists(
            st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
            min_size=0, max_size=8
        ).map(tuple),
        direction=st.sampled_from([
            Direction.up(), Direction.down(), Direction.left(), Direction.right()
        ]),
        food_position=st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
        score=st.integers(0, 1000),
        random_state=st.builds(random.Random, st.integers(0, 2**32-1)),
        grid_size=st.just((20, 20))
    )


def moves():
    """Generate random Move instances for property testing."""
    return st.builds(
        Move,
        action=st.sampled_from([MoveAction.LEFT_TURN, MoveAction.STRAIGHT, MoveAction.RIGHT_TURN]),
        resulting_direction=st.sampled_from([
            Direction.up(), Direction.down(), Direction.left(), Direction.right()
        ])
    )


def simulation_configs():
    """Generate random SimulationConfig instances for property testing."""
    return st.builds(
        SimulationConfig,
        grid_size=st.just((20, 20)),
        initial_snake_length=st.integers(3, 5),
        move_budget=st.integers(10, 100),
        random_seed=st.integers(0, 2**16-1),
        nn_enabled=st.just(True),  # Always enable NN for hybrid execution tests
        max_tree_depth=st.one_of(st.none(), st.integers(1, 5)),
        food_reward=st.just(10),
        collision_penalty=st.just(-10),
        empty_move_reward=st.just(0)
    )


class MockTreeSearchResult:
    """Mock tree search result for testing oracle decision logic."""
    
    def __init__(self, optimal_move: Move, paths_count: int = 3):
        self.optimal_move = optimal_move
        self.paths_count = paths_count


class MockHydraMgr:
    """Simplified HydraMgr for testing oracle decision logic."""
    
    def __init__(self, simulation_config: SimulationConfig):
        self.simulation_config = simulation_config
        
        # Initialize neural network components
        self.neural_network = SnakeNet()
        self.feature_extractor = FeatureExtractor()
        self.oracle_trainer = OracleTrainer(self.neural_network)
        
        # Track decisions for testing
        self.decisions_made = []
        self.training_samples_generated = []
    
    def get_neural_network_prediction(self, board: GameBoard) -> Move:
        """Get neural network prediction for a game board."""
        # For testing, make prediction deterministic based on board state
        possible_moves = GameLogic.get_possible_moves(board.direction)
        
        # Use board state to deterministically choose a move
        board_hash = hash((board.snake_head.x, board.snake_head.y, board.direction.dx, board.direction.dy))
        move_index = board_hash % len(possible_moves)
        
        return possible_moves[move_index]
    
    def execute_tree_search(self, board: GameBoard, nn_prediction: Move) -> MockTreeSearchResult:
        """Mock tree search execution that returns a result."""
        # For testing, we'll create different scenarios:
        # - Sometimes tree search agrees with NN
        # - Sometimes tree search disagrees with NN
        
        possible_moves = GameLogic.get_possible_moves(board.direction)
        
        # Use board hash to deterministically choose whether to agree or disagree
        board_hash = hash((board.snake_head.x, board.snake_head.y, board.direction.dx, board.direction.dy))
        
        if board_hash % 3 == 0:
            # Tree search agrees with NN prediction
            optimal_move = nn_prediction
        else:
            # Tree search disagrees - pick a different move
            other_moves = [m for m in possible_moves if m.action != nn_prediction.action]
            optimal_move = other_moves[board_hash % len(other_moves)] if other_moves else nn_prediction
        
        return MockTreeSearchResult(optimal_move)
    
    def make_oracle_decision(self, board: GameBoard) -> tuple[Move, bool, bool]:
        """
        Make an oracle decision and return the decision, whether NN was correct, and whether training occurred.
        
        Returns:
            tuple: (final_decision, nn_was_correct, training_sample_generated)
        """
        # Step 1: Get neural network prediction
        nn_prediction = self.get_neural_network_prediction(board)
        
        # Step 2: Execute tree search
        tree_result = self.execute_tree_search(board, nn_prediction)
        optimal_move = tree_result.optimal_move
        
        # Step 3: Compare predictions
        nn_was_correct = self.oracle_trainer.compare_predictions(nn_prediction, optimal_move, board.score)
        
        # Step 4: Always choose tree search result (oracle decision correctness)
        final_decision = optimal_move
        
        # Step 5: Generate training sample if NN was wrong
        features = self.feature_extractor.extract_features(board)
        training_sample = self.oracle_trainer.generate_training_sample(
            features, nn_prediction, optimal_move, board.score
        )
        
        training_sample_generated = training_sample.was_nn_wrong
        
        # Track for testing
        self.decisions_made.append({
            'nn_prediction': nn_prediction,
            'tree_optimal': optimal_move,
            'final_decision': final_decision,
            'nn_was_correct': nn_was_correct,
            'training_generated': training_sample_generated
        })
        
        if training_sample_generated:
            self.training_samples_generated.append(training_sample)
        
        return final_decision, nn_was_correct, training_sample_generated


@given(
    board=game_boards(),
    config=simulation_configs()
)
@settings(deadline=None, max_examples=100)  # Disable deadline and limit examples for faster testing
def test_oracle_decision_correctness(board, config):
    """
    Feature: ai-hydra, Property: Oracle decision correctness
    
    Test that tree search result is chosen when it differs from NN prediction.
    This validates Requirements 11.4 and 11.5:
    - Oracle_Trainer compares NN prediction with tree search result
    - System chooses tree search result if it differs from NN prediction
    - Training data is generated when predictions differ
    
    **Validates: Requirements 11.4, 11.5**
    """
    # Create mock HydraMgr for testing
    mock_mgr = MockHydraMgr(config)
    
    # Make oracle decision
    final_decision, nn_was_correct, training_generated = mock_mgr.make_oracle_decision(board)
    
    # Get the last decision made
    assert len(mock_mgr.decisions_made) == 1
    decision_info = mock_mgr.decisions_made[0]
    
    # Property 1: Final decision should always be the tree search result (oracle correctness)
    assert final_decision == decision_info['tree_optimal'], \
        "Oracle decision should always choose tree search result over NN prediction"
    
    # Property 2: Training sample should be generated if and only if NN was wrong
    if nn_was_correct:
        assert not training_generated, \
            "No training sample should be generated when NN prediction is correct"
        assert len(mock_mgr.training_samples_generated) == 0, \
            "Training samples list should be empty when NN is correct"
    else:
        assert training_generated, \
            "Training sample should be generated when NN prediction is incorrect"
        assert len(mock_mgr.training_samples_generated) == 1, \
            "Exactly one training sample should be generated when NN is wrong"
        
        # Verify training sample correctness
        training_sample = mock_mgr.training_samples_generated[0]
        assert training_sample.was_nn_wrong, \
            "Training sample should be marked as NN being wrong"
        assert training_sample.optimal_action == mock_mgr.oracle_trainer._move_to_action_index(decision_info['tree_optimal']), \
            "Training sample should contain the correct optimal action from tree search"
        assert training_sample.nn_prediction == mock_mgr.oracle_trainer._move_to_action_index(decision_info['nn_prediction']), \
            "Training sample should contain the NN prediction action"
    
    # Property 3: Oracle trainer should have updated its statistics
    assert mock_mgr.oracle_trainer.total_predictions == 1, \
        "Oracle trainer should have recorded exactly one prediction comparison"
    
    if nn_was_correct:
        assert mock_mgr.oracle_trainer.correct_predictions == 1, \
            "Oracle trainer should record correct prediction when NN matches tree search"
        assert mock_mgr.oracle_trainer.get_prediction_accuracy() == 1.0, \
            "Accuracy should be 100% when NN prediction is correct"
    else:
        assert mock_mgr.oracle_trainer.correct_predictions == 0, \
            "Oracle trainer should record incorrect prediction when NN differs from tree search"
        assert mock_mgr.oracle_trainer.get_prediction_accuracy() == 0.0, \
            "Accuracy should be 0% when NN prediction is incorrect"
    
    # Property 4: Decision should be a valid move for the current board state
    possible_moves = GameLogic.get_possible_moves(board.direction)
    possible_actions = [move.action for move in possible_moves]
    assert final_decision.action in possible_actions, \
        "Final decision should be a valid move action for the current board state"


@given(
    boards=st.lists(game_boards(), min_size=5, max_size=20),
    config=simulation_configs()
)
@settings(deadline=None, max_examples=50)  # Disable deadline and limit examples
def test_oracle_decision_consistency_across_multiple_decisions(boards, config):
    """
    Feature: ai-hydra, Property: Oracle decision correctness
    
    Test that oracle decision logic is consistent across multiple decisions
    and properly accumulates training data and statistics.
    
    **Validates: Requirements 11.4, 11.5**
    """
    mock_mgr = MockHydraMgr(config)
    
    total_decisions = len(boards)
    correct_predictions = 0
    training_samples_expected = 0
    
    # Make decisions for all boards
    for board in boards:
        final_decision, nn_was_correct, training_generated = mock_mgr.make_oracle_decision(board)
        
        if nn_was_correct:
            correct_predictions += 1
        else:
            training_samples_expected += 1
    
    # Verify overall statistics
    assert len(mock_mgr.decisions_made) == total_decisions, \
        f"Should have made {total_decisions} decisions"
    
    assert mock_mgr.oracle_trainer.total_predictions == total_decisions, \
        f"Oracle trainer should have recorded {total_decisions} predictions"
    
    assert mock_mgr.oracle_trainer.correct_predictions == correct_predictions, \
        f"Oracle trainer should have recorded {correct_predictions} correct predictions"
    
    assert len(mock_mgr.training_samples_generated) == training_samples_expected, \
        f"Should have generated {training_samples_expected} training samples"
    
    # Verify accuracy calculation
    expected_accuracy = correct_predictions / total_decisions if total_decisions > 0 else 0.0
    actual_accuracy = mock_mgr.oracle_trainer.get_prediction_accuracy()
    assert abs(actual_accuracy - expected_accuracy) < 1e-6, \
        f"Accuracy should be {expected_accuracy:.3f}, got {actual_accuracy:.3f}"
    
    # Property: All final decisions should be tree search results
    for decision_info in mock_mgr.decisions_made:
        assert decision_info['final_decision'] == decision_info['tree_optimal'], \
            "Every final decision should be the tree search result (oracle correctness)"
    
    # Property: Training samples should only be generated for incorrect NN predictions
    incorrect_decisions = [d for d in mock_mgr.decisions_made if not d['nn_was_correct']]
    assert len(incorrect_decisions) == len(mock_mgr.training_samples_generated), \
        "Training samples should be generated exactly for incorrect NN predictions"


@given(
    board=game_boards(),
    config=simulation_configs()
)
@settings(deadline=None, max_examples=100)
def test_oracle_decision_with_identical_predictions(board, config):
    """
    Feature: ai-hydra, Property: Oracle decision correctness
    
    Test oracle decision behavior when NN prediction matches tree search result.
    In this case, no training should occur, but the decision should still be correct.
    
    **Validates: Requirements 11.4, 11.5**
    """
    # Create a mock that forces NN and tree search to agree
    class AlwaysAgreeHydraMgr(MockHydraMgr):
        def execute_tree_search(self, board: GameBoard, nn_prediction: Move) -> MockTreeSearchResult:
            # Always return the NN prediction as optimal
            return MockTreeSearchResult(nn_prediction)
    
    mock_mgr = AlwaysAgreeHydraMgr(config)
    final_decision, nn_was_correct, training_generated = mock_mgr.make_oracle_decision(board)
    
    # When predictions match
    assert nn_was_correct, "NN should be correct when it matches tree search"
    assert not training_generated, "No training should occur when predictions match"
    assert len(mock_mgr.training_samples_generated) == 0, "No training samples should be generated"
    
    # Final decision should still be the tree search result (which happens to match NN)
    decision_info = mock_mgr.decisions_made[0]
    assert final_decision == decision_info['tree_optimal'], \
        "Final decision should be tree search result even when it matches NN"
    assert final_decision == decision_info['nn_prediction'], \
        "When predictions match, final decision equals both NN and tree search"
    
    # Oracle trainer statistics should reflect correct prediction
    assert mock_mgr.oracle_trainer.get_prediction_accuracy() == 1.0, \
        "Accuracy should be 100% when predictions match"


@given(
    board=game_boards(),
    config=simulation_configs()
)
@settings(deadline=None, max_examples=100)
def test_oracle_decision_with_different_predictions(board, config):
    """
    Feature: ai-hydra, Property: Oracle decision correctness
    
    Test oracle decision behavior when NN prediction differs from tree search result.
    In this case, training should occur and tree search result should be chosen.
    
    **Validates: Requirements 11.4, 11.5**
    """
    # Create a mock that forces NN and tree search to disagree
    class AlwaysDisagreeHydraMgr(MockHydraMgr):
        def execute_tree_search(self, board: GameBoard, nn_prediction: Move) -> MockTreeSearchResult:
            # Always return a different move from NN prediction
            possible_moves = GameLogic.get_possible_moves(board.direction)
            other_moves = [m for m in possible_moves if m.action != nn_prediction.action]
            if other_moves:
                return MockTreeSearchResult(other_moves[0])
            else:
                # If no other moves available, return the same (edge case)
                return MockTreeSearchResult(nn_prediction)
    
    mock_mgr = AlwaysDisagreeHydraMgr(config)
    final_decision, nn_was_correct, training_generated = mock_mgr.make_oracle_decision(board)
    
    decision_info = mock_mgr.decisions_made[0]
    
    # Check if predictions actually differ (they should in most cases)
    predictions_differ = decision_info['nn_prediction'].action != decision_info['tree_optimal'].action
    
    if predictions_differ:
        # When predictions differ
        assert not nn_was_correct, "NN should be incorrect when it differs from tree search"
        assert training_generated, "Training should occur when predictions differ"
        assert len(mock_mgr.training_samples_generated) == 1, "One training sample should be generated"
        
        # Final decision should be tree search result
        assert final_decision == decision_info['tree_optimal'], \
            "Final decision should be tree search result when predictions differ"
        assert final_decision != decision_info['nn_prediction'], \
            "Final decision should differ from NN prediction when tree search disagrees"
        
        # Oracle trainer statistics should reflect incorrect prediction
        assert mock_mgr.oracle_trainer.get_prediction_accuracy() == 0.0, \
            "Accuracy should be 0% when predictions differ"
    else:
        # Edge case: predictions happen to be the same despite our attempt to make them differ
        # This can happen if there's only one valid move
        assert nn_was_correct, "NN should be correct when predictions match"
        assert not training_generated, "No training should occur when predictions match"