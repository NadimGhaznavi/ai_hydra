"""
Property-based tests for GameBoard functionality.

This module contains property-based tests that validate the correctness
of GameBoard operations, particularly focusing on cloning behavior.
"""

import random
import copy
from hypothesis import given, strategies as st
import pytest

from ai_hydra.models import GameBoard, Position, Direction


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
def test_perfect_gameboard_cloning(board):
    """
    Feature: ai-hydra, Property 3: Perfect GameBoard Cloning
    
    For any GameBoard cloning operation, the cloned board should be identical 
    to the source board in all aspects (snake position, body segments, direction, 
    food location, score, random state) and completely independent.
    
    **Validates: Requirements 3.4, 6.4, 7.3, 10.4**
    """
    # Clone the board
    cloned_board = board.clone()
    
    # Test 1: All fields should be identical
    assert cloned_board.snake_head == board.snake_head
    assert cloned_board.snake_body == board.snake_body
    assert cloned_board.direction == board.direction
    assert cloned_board.food_position == board.food_position
    assert cloned_board.score == board.score
    assert cloned_board.grid_size == board.grid_size
    
    # Test 2: Random state should have the same internal state
    # We test this by generating the same sequence from both random states
    original_random = copy.deepcopy(board.random_state)
    cloned_random = copy.deepcopy(cloned_board.random_state)
    
    # Generate a sequence of random numbers from both
    original_sequence = [original_random.random() for _ in range(10)]
    cloned_sequence = [cloned_random.random() for _ in range(10)]
    
    assert original_sequence == cloned_sequence, "Random states should be identical"
    
    # Test 3: Independence - modifying the cloned random state should not affect original
    # Since GameBoard is immutable, we can't directly modify it, but we can verify
    # that the random state objects are different instances
    assert cloned_board.random_state is not board.random_state, "Random state should be independent"
    
    # Test 4: The cloned board should be a different object instance
    assert cloned_board is not board, "Cloned board should be a different instance"
    
    # Test 5: All accessor methods should return identical values
    assert cloned_board.get_snake_head() == board.get_snake_head()
    assert cloned_board.get_snake_body() == board.get_snake_body()
    assert cloned_board.get_direction() == board.get_direction()
    assert cloned_board.get_food_position() == board.get_food_position()
    assert cloned_board.get_score() == board.get_score()
    assert cloned_board.get_grid_size() == board.get_grid_size()
    
    # Test 6: All snake positions should be identical
    assert cloned_board.get_all_snake_positions() == board.get_all_snake_positions()
    
    # Test 7: Position occupation checks should be identical
    test_positions = [
        Position(0, 0), Position(10, 10), Position(19, 19),
        board.snake_head, board.food_position
    ]
    
    for pos in test_positions:
        assert (cloned_board.is_position_occupied_by_snake(pos) == 
                board.is_position_occupied_by_snake(pos))
        assert (cloned_board.is_position_within_bounds(pos) == 
                board.is_position_within_bounds(pos))


@given(board=game_boards())
def test_gameboard_immutability(board):
    """
    Test that GameBoard instances are truly immutable.
    
    This test verifies that GameBoard objects cannot be modified after creation,
    which is essential for safe cloning and state management.
    """
    # Try to modify fields - these should all fail with AttributeError
    with pytest.raises(AttributeError):
        board.snake_head = Position(0, 0)
    
    with pytest.raises(AttributeError):
        board.score = 999
    
    with pytest.raises(AttributeError):
        board.direction = Direction.up()
    
    # The board should remain unchanged after failed modification attempts
    original_head = board.snake_head
    original_score = board.score
    original_direction = board.direction
    
    try:
        board.snake_head = Position(0, 0)
    except AttributeError:
        pass
    
    assert board.snake_head == original_head
    assert board.score == original_score
    assert board.direction == original_direction


@given(board=game_boards())
def test_gameboard_clone_independence(board):
    """
    Test that cloned GameBoard instances are completely independent.
    
    This test verifies that operations on the random state of one board
    do not affect the random state of its clone.
    """
    cloned_board = board.clone()
    
    # Advance the random state of the original board
    original_random = board.random_state
    cloned_random = cloned_board.random_state
    
    # Generate some random numbers from the original
    original_values = [original_random.random() for _ in range(5)]
    
    # The cloned random state should still generate the same sequence as the original would have
    cloned_values = [cloned_random.random() for _ in range(5)]
    
    # They should be identical since they started from the same state
    assert original_values == cloned_values
    
    # But now they should diverge if we continue generating
    more_original = [original_random.random() for _ in range(3)]
    more_cloned = [cloned_random.random() for _ in range(3)]
    
    # These should be the same since both random states were advanced equally
    assert more_original == more_cloned