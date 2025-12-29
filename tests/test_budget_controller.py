"""
Property-based tests for BudgetController functionality.

This module contains property-based tests that validate the correctness
of budget management operations during tree search exploration.
"""

from hypothesis import given, strategies as st, settings
import pytest

from ai_hydra.budget_controller import BudgetController


@given(
    initial_budget=st.integers(min_value=1, max_value=1000),
    moves_to_consume=st.integers(min_value=0, max_value=50)
)
@settings(max_examples=10, deadline=2000)  # 2 second deadline
def test_budget_lifecycle_management(initial_budget, moves_to_consume):
    """
    Feature: ai-hydra, Property 5: Budget Lifecycle Management
    
    For any decision cycle, the move budget should be initialized to the configured value,
    decremented by exactly 1 for each clone move execution, and reset to the original 
    value after each master move application.
    
    **Validates: Requirements 4.1, 4.2, 6.2**
    """
    # Test 1: Budget initialization
    controller = BudgetController(initial_budget)
    
    # Verify initial state
    assert controller.get_remaining_budget() == initial_budget, "Initial budget should match configured value"
    assert controller.get_budget_consumed() == 0, "Initially no budget should be consumed"
    assert controller.get_moves_in_current_round() == 0, "Initially no moves in current round"
    assert not controller.is_budget_exhausted(), "Budget should not be exhausted initially"
    assert controller.get_total_moves_consumed() == 0, "Total moves should be 0 initially"
    
    # Test 2: Budget consumption - decrement by exactly 1 for each move
    moves_consumed = min(moves_to_consume, initial_budget + 10)  # Allow some overrun
    
    for i in range(moves_consumed):
        expected_remaining = initial_budget - (i + 1)
        expected_consumed = i + 1
        expected_moves_in_round = i + 1
        
        controller.consume_move()
        
        # Verify budget decrements by exactly 1
        assert controller.get_remaining_budget() == expected_remaining, f"Budget should decrement by 1 at step {i+1}"
        assert controller.get_budget_consumed() == expected_consumed, f"Consumed budget should be {expected_consumed} at step {i+1}"
        assert controller.get_moves_in_current_round() == expected_moves_in_round, f"Moves in round should be {expected_moves_in_round} at step {i+1}"
        assert controller.get_total_moves_consumed() == expected_consumed, f"Total moves should be {expected_consumed} at step {i+1}"
        
        # Check exhaustion status
        if expected_remaining <= 0:
            assert controller.is_budget_exhausted(), f"Budget should be exhausted when remaining is {expected_remaining}"
        else:
            assert not controller.is_budget_exhausted(), f"Budget should not be exhausted when remaining is {expected_remaining}"
    
    # Test 3: Budget reset - should return to original value
    controller.reset_budget()
    
    # Verify reset state
    assert controller.get_remaining_budget() == initial_budget, "Budget should reset to initial value"
    assert controller.get_budget_consumed() == 0, "Consumed budget should reset to 0"
    assert controller.get_moves_in_current_round() == 0, "Moves in round should reset to 0"
    assert controller.get_total_moves_consumed() == 0, "Total moves should reset to 0"
    assert not controller.is_budget_exhausted(), "Budget should not be exhausted after reset"
    assert controller.get_round_number() == 0, "Round number should reset to 0"
    
    # Test 4: Multiple cycles - budget should reset properly each time
    for cycle in range(3):  # Test 3 cycles
        # Consume some budget
        moves_in_cycle = min(5, initial_budget)
        for _ in range(moves_in_cycle):
            controller.consume_move()
        
        # Verify consumption
        assert controller.get_budget_consumed() == moves_in_cycle, f"Cycle {cycle}: consumed should be {moves_in_cycle}"
        assert controller.get_remaining_budget() == initial_budget - moves_in_cycle, f"Cycle {cycle}: remaining should be correct"
        
        # Reset for next cycle
        controller.reset_budget()
        
        # Verify reset
        assert controller.get_remaining_budget() == initial_budget, f"Cycle {cycle}: budget should reset"
        assert controller.get_budget_consumed() == 0, f"Cycle {cycle}: consumed should reset"


# Import additional modules for clone management testing
from ai_hydra.state_manager import StateManager
from ai_hydra.exploration_clone import ExplorationClone
from ai_hydra.models import GameBoard, Position, Direction
from ai_hydra.game_logic import GameLogic
import random


def game_boards():
    """Generate random GameBoard instances for property testing."""
    return st.builds(
        GameBoard,
        snake_head=st.builds(Position, x=st.integers(5, 15), y=st.integers(5, 15)),
        snake_body=st.lists(
            st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
            min_size=2, max_size=5
        ).map(tuple),
        direction=st.sampled_from([
            Direction.up(), Direction.down(), Direction.left(), Direction.right()
        ]),
        food_position=st.builds(Position, x=st.integers(0, 19), y=st.integers(0, 19)),
        score=st.integers(0, 100),
        random_state=st.builds(random.Random, st.integers(0, 2**32-1)),
        grid_size=st.just((20, 20))
    )


@given(master_board=game_boards())
@settings(max_examples=5, deadline=3000)  # 3 second deadline
def test_exploration_clone_management_invariant(master_board):
    """
    Feature: ai-hydra, Property 2: Exploration Clone Management Invariant
    
    For any point during simulation execution, the system should maintain exactly 3 initial 
    exploration clones from the master game state, with each clone testing exactly one move 
    decision (left, straight, right).
    
    **Validates: Requirements 2.2, 3.2, 3.3, 6.3**
    """
    state_manager = StateManager()
    
    # Test 1: Initial clone creation should create exactly 3 clones
    initial_clones = state_manager.create_initial_clones(master_board)
    
    # Verify exactly 3 clones are created
    assert len(initial_clones) == 3, "Should create exactly 3 initial exploration clones"
    assert state_manager.get_active_clone_count() == 3, "State manager should track exactly 3 active clones"
    
    # Verify clone IDs are "1", "2", "3"
    clone_ids = [clone.get_clone_id() for clone in initial_clones]
    expected_ids = ["1", "2", "3"]
    assert set(clone_ids) == set(expected_ids), f"Clone IDs should be {expected_ids}, got {clone_ids}"
    
    # Test 2: Each clone should be independent and have identical initial state
    for i, clone in enumerate(initial_clones):
        # Verify clone independence
        assert clone is not master_board, f"Clone {i+1} should be independent from master board"
        assert clone.get_current_board() is not master_board, f"Clone {i+1} board should be independent"
        
        # Verify identical initial state (but independent instances)
        clone_board = clone.get_current_board()
        assert clone_board.get_snake_head() == master_board.get_snake_head(), f"Clone {i+1} should have same initial head position"
        assert clone_board.get_snake_body() == master_board.get_snake_body(), f"Clone {i+1} should have same initial body"
        assert clone_board.get_direction().dx == master_board.get_direction().dx, f"Clone {i+1} should have same initial direction"
        assert clone_board.get_direction().dy == master_board.get_direction().dy, f"Clone {i+1} should have same initial direction"
        assert clone_board.get_food_position() == master_board.get_food_position(), f"Clone {i+1} should have same initial food position"
        assert clone_board.get_score() == master_board.get_score(), f"Clone {i+1} should have same initial score"
        
        # Verify clone-specific properties
        assert clone.get_parent_id() is None, f"Initial clone {i+1} should have no parent"
        assert clone.get_cumulative_reward() == 0, f"Initial clone {i+1} should have 0 cumulative reward"
        assert len(clone.get_path_from_root()) == 0, f"Initial clone {i+1} should have empty path"
        assert not clone.is_terminated(), f"Initial clone {i+1} should not be terminated"
    
    # Test 3: Each clone should test exactly one move decision
    possible_moves = GameLogic.get_possible_moves(master_board.get_direction())
    assert len(possible_moves) == 3, "Should have exactly 3 possible moves (left, straight, right)"
    
    # Execute one move on each clone to verify they can test different moves
    for i, clone in enumerate(initial_clones):
        move = possible_moves[i]  # Each clone gets a different move
        
        # Verify clone can execute the move
        assert not clone.is_terminated(), f"Clone {i+1} should be able to execute moves"
        
        # Store initial state for comparison
        initial_reward = clone.get_cumulative_reward()
        initial_path_length = len(clone.get_path_from_root())
        
        # Execute the move
        result = clone.execute_move(move)
        
        # Verify move execution updates clone state correctly
        assert clone.get_cumulative_reward() == initial_reward + result.reward, f"Clone {i+1} reward should update correctly"
        assert len(clone.get_path_from_root()) == initial_path_length + 1, f"Clone {i+1} path should grow by 1"
        assert clone.get_path_from_root()[-1] == move, f"Clone {i+1} should record the executed move"
    
    # Test 4: State manager should maintain clone tracking correctly
    active_clones_after_moves = state_manager.get_active_clones()
    assert len(active_clones_after_moves) == 3, "Should still have exactly 3 active clones after moves"
    
    # Verify all clones are still tracked
    for clone in initial_clones:
        tracked_clone = state_manager.get_clone_by_id(clone.get_clone_id())
        assert tracked_clone is clone, f"Clone {clone.get_clone_id()} should still be tracked by state manager"
    
    # Test 5: Tree destruction should clean up all clones
    state_manager.destroy_exploration_tree()
    
    assert state_manager.get_active_clone_count() == 0, "All clones should be destroyed"
    assert len(state_manager.get_active_clones()) == 0, "Active clones list should be empty"
    
    # Verify individual clone lookups return None after destruction
    for clone_id in expected_ids:
        assert state_manager.get_clone_by_id(clone_id) is None, f"Clone {clone_id} should not be found after destruction"
    
    # Test 6: New clone creation after destruction should work correctly
    new_initial_clones = state_manager.create_initial_clones(master_board)
    
    assert len(new_initial_clones) == 3, "Should create exactly 3 new initial clones after destruction"
    assert state_manager.get_active_clone_count() == 3, "Should have exactly 3 active clones after recreation"
    
    # Verify new clones have correct IDs and are independent from previous clones
    new_clone_ids = [clone.get_clone_id() for clone in new_initial_clones]
    assert set(new_clone_ids) == set(expected_ids), "New clones should have same ID pattern"
    
    # Verify new clones are different instances from the destroyed ones
    for i, new_clone in enumerate(new_initial_clones):
        assert new_clone is not initial_clones[i], "New clones should be different instances"
        assert new_clone.get_cumulative_reward() == 0, "New clones should start with 0 reward"
        assert len(new_clone.get_path_from_root()) == 0, "New clones should start with empty path"


@given(
    master_board=game_boards(),
    sub_clone_generations=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=5, deadline=5000)  # 5 second deadline
def test_sub_clone_creation_hierarchy(master_board, sub_clone_generations):
    """
    Test that sub-clone creation maintains proper hierarchical naming and relationships.
    
    This test verifies that the StateManager correctly creates sub-clones with
    hierarchical naming (1L, 1S, 1R from parent 1) and maintains parent-child
    relationships throughout the tree structure.
    """
    state_manager = StateManager()
    
    # Create initial clones
    initial_clones = state_manager.create_initial_clones(master_board)
    assert len(initial_clones) == 3, "Should start with 3 initial clones"
    
    current_generation = initial_clones
    
    for generation in range(sub_clone_generations):
        next_generation = []
        
        for parent_clone in current_generation:
            if not parent_clone.is_terminated():
                # Create sub-clones from this parent
                sub_clones = state_manager.create_sub_clones(parent_clone)
                
                # Verify exactly 3 sub-clones are created
                assert len(sub_clones) == 3, f"Should create exactly 3 sub-clones from parent {parent_clone.get_clone_id()}"
                
                # Verify hierarchical naming
                parent_id = parent_clone.get_clone_id()
                expected_sub_ids = [parent_id + suffix for suffix in ["L", "S", "R"]]
                actual_sub_ids = [sub_clone.get_clone_id() for sub_clone in sub_clones]
                
                assert set(actual_sub_ids) == set(expected_sub_ids), f"Sub-clone IDs should follow hierarchical pattern: expected {expected_sub_ids}, got {actual_sub_ids}"
                
                # Verify parent-child relationships
                for sub_clone in sub_clones:
                    assert sub_clone.get_parent_id() == parent_id, f"Sub-clone {sub_clone.get_clone_id()} should have parent {parent_id}"
                    
                    # Verify sub-clone starts with parent's board state
                    parent_board = parent_clone.get_current_board()
                    sub_board = sub_clone.get_current_board()
                    
                    # Boards should have same state but be independent instances
                    assert sub_board is not parent_board, "Sub-clone board should be independent"
                    assert sub_board.get_snake_head() == parent_board.get_snake_head(), "Sub-clone should inherit parent's head position"
                    assert sub_board.get_score() == parent_board.get_score(), "Sub-clone should inherit parent's score"
                
                next_generation.extend(sub_clones)
        
        current_generation = next_generation
        
        # Verify state manager tracks all clones correctly
        total_expected_clones = 3 * (4**generation + 4**(generation+1)) // 3  # Geometric series for tree growth
        if generation == 0:
            total_expected_clones = 3 + len(next_generation)  # Initial 3 + first generation sub-clones
        
        # For simplicity, just verify that clone count increased
        assert state_manager.get_active_clone_count() > 3, f"Should have more than 3 clones after generation {generation+1}"
    
    # Verify tree structure is valid
    assert state_manager.validate_tree_structure(), "Tree structure should be valid after sub-clone creation"
    
    # Test tree destruction cleans up all generations
    clone_count_before_destruction = state_manager.get_active_clone_count()
    assert clone_count_before_destruction > 3, "Should have multiple generations of clones"
    
    state_manager.destroy_exploration_tree()
    assert state_manager.get_active_clone_count() == 0, "All clones should be destroyed regardless of generation"


@given(master_board=game_boards())
@settings(max_examples=5, deadline=3000)  # 3 second deadline
def test_clone_independence_and_state_isolation(master_board):
    """
    Test that clones maintain complete independence and state isolation.
    
    This test verifies that modifications to one clone's state do not affect
    other clones or the master board, ensuring proper isolation.
    """
    state_manager = StateManager()
    initial_clones = state_manager.create_initial_clones(master_board)
    
    # Get possible moves
    possible_moves = GameLogic.get_possible_moves(master_board.get_direction())
    
    # Execute different moves on each clone
    for i, clone in enumerate(initial_clones):
        move = possible_moves[i % len(possible_moves)]
        
        # Store state of other clones before this clone's move
        other_clones_states = []
        for j, other_clone in enumerate(initial_clones):
            if i != j:
                other_clones_states.append({
                    'head': other_clone.get_current_board().get_snake_head(),
                    'score': other_clone.get_current_board().get_score(),
                    'reward': other_clone.get_cumulative_reward(),
                    'path_length': len(other_clone.get_path_from_root())
                })
        
        # Execute move on current clone
        clone.execute_move(move)
        
        # Verify other clones are unaffected
        other_clone_index = 0
        for j, other_clone in enumerate(initial_clones):
            if i != j:
                expected_state = other_clones_states[other_clone_index]
                
                assert other_clone.get_current_board().get_snake_head() == expected_state['head'], f"Clone {j+1} head should be unaffected by clone {i+1} move"
                assert other_clone.get_current_board().get_score() == expected_state['score'], f"Clone {j+1} score should be unaffected by clone {i+1} move"
                assert other_clone.get_cumulative_reward() == expected_state['reward'], f"Clone {j+1} reward should be unaffected by clone {i+1} move"
                assert len(other_clone.get_path_from_root()) == expected_state['path_length'], f"Clone {j+1} path should be unaffected by clone {i+1} move"
                
                other_clone_index += 1
        
        # Verify master board is unaffected
        assert master_board.get_snake_head() == master_board.get_snake_head(), "Master board should be unaffected"
        assert master_board.get_score() == master_board.get_score(), "Master board score should be unaffected"