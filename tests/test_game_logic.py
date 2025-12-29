"""
Property-based tests for GameLogic functionality.

This module contains property-based tests that validate the correctness
of GameLogic operations, particularly focusing on immutability behavior.
"""

import random
import copy
from hypothesis import given, strategies as st
import pytest

from ai_hydra.models import GameBoard, Position, Direction, Move, MoveAction
from ai_hydra.game_logic import GameLogic


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
        move_count=st.integers(0, 1000),
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


@given(board=game_boards())
def test_game_board_and_game_logic_immutability(board):
    """
    Feature: ai-hydra, Property 10: Game_Board and Game_Logic Immutability
    
    For any move execution, the Game_Logic module should return a new GameBoard 
    instance without modifying the original GameBoard, maintaining immutability principles.
    
    **Validates: Requirements 10.1, 10.3, 10.5**
    """
    # Create a move from the current direction
    possible_moves = GameLogic.get_possible_moves(board.direction)
    move = possible_moves[0]  # Use the first possible move (left turn)
    
    # Store original board state for comparison
    original_head = board.snake_head
    original_body = board.snake_body
    original_direction = board.direction
    original_food = board.food_position
    original_score = board.score
    original_grid_size = board.grid_size
    
    # Create a deep copy of the random state to compare later
    original_random_state = copy.deepcopy(board.random_state)
    
    # Execute the move
    result = GameLogic.execute_move(board, move)
    
    # Test 1: Original board should be completely unchanged
    assert board.snake_head == original_head, "Original snake head should not change"
    assert board.snake_body == original_body, "Original snake body should not change"
    assert board.direction == original_direction, "Original direction should not change"
    assert board.food_position == original_food, "Original food position should not change"
    assert board.score == original_score, "Original score should not change"
    assert board.grid_size == original_grid_size, "Original grid size should not change"
    
    # Test 2: Random state should be unchanged in the original board
    # We test this by generating the same sequence from both random states
    # First, let's test that the random state itself wasn't modified
    test_random_copy = copy.deepcopy(board.random_state)
    original_sequence = [original_random_state.random() for _ in range(5)]
    current_sequence = [test_random_copy.random() for _ in range(5)]
    assert original_sequence == current_sequence, f"Original random state should be unchanged. Original: {original_sequence}, Current: {current_sequence}"
    
    # Test 3: Result should contain a new GameBoard instance
    assert result.new_board is not board, "Result should contain a new GameBoard instance"
    assert isinstance(result.new_board, GameBoard), "Result should contain a GameBoard"
    
    # Test 4: The new board should be properly constructed
    assert hasattr(result.new_board, 'snake_head'), "New board should have snake_head"
    assert hasattr(result.new_board, 'snake_body'), "New board should have snake_body"
    assert hasattr(result.new_board, 'direction'), "New board should have direction"
    assert hasattr(result.new_board, 'food_position'), "New board should have food_position"
    assert hasattr(result.new_board, 'score'), "New board should have score"
    assert hasattr(result.new_board, 'grid_size'), "New board should have grid_size"
    
    # Test 5: All accessor methods should work on both boards
    # Original board accessors should still work
    assert board.get_snake_head() == original_head
    assert board.get_snake_body() == list(original_body)
    assert board.get_direction() == original_direction
    assert board.get_food_position() == original_food
    assert board.get_score() == original_score
    assert board.get_grid_size() == original_grid_size
    
    # New board accessors should work
    new_head = result.new_board.get_snake_head()
    new_body = result.new_board.get_snake_body()
    new_direction = result.new_board.get_direction()
    new_food = result.new_board.get_food_position()
    new_score = result.new_board.get_score()
    new_grid_size = result.new_board.get_grid_size()
    
    assert isinstance(new_head, Position), "New head should be a Position"
    assert isinstance(new_body, list), "New body should be a list"
    assert isinstance(new_direction, Direction), "New direction should be a Direction"
    assert isinstance(new_food, Position), "New food should be a Position"
    assert isinstance(new_score, int), "New score should be an int"
    assert isinstance(new_grid_size, tuple), "New grid size should be a tuple"


@given(board=game_boards())
def test_game_logic_static_methods_immutability(board):
    """
    Test that GameLogic static methods don't modify input parameters.
    
    This test verifies that all GameLogic static methods maintain immutability
    by not modifying their input parameters.
    """
    # Store original state
    original_head = board.snake_head
    original_body = board.snake_body
    original_direction = board.direction
    original_food = board.food_position
    original_score = board.score
    
    # Test get_possible_moves doesn't modify the direction
    direction_copy = Direction(board.direction.dx, board.direction.dy)
    possible_moves = GameLogic.get_possible_moves(board.direction)
    
    assert board.direction.dx == direction_copy.dx, "Direction should not be modified"
    assert board.direction.dy == direction_copy.dy, "Direction should not be modified"
    assert len(possible_moves) == 3, "Should return 3 possible moves"
    
    # Test create_move doesn't modify the direction
    move = GameLogic.create_move(board.direction, MoveAction.LEFT_TURN)
    assert board.direction.dx == direction_copy.dx, "Direction should not be modified by create_move"
    assert board.direction.dy == direction_copy.dy, "Direction should not be modified by create_move"
    
    # Test is_game_over doesn't modify the board
    is_over = GameLogic.is_game_over(board)
    assert board.snake_head == original_head, "Board should not be modified by is_game_over"
    assert board.snake_body == original_body, "Board should not be modified by is_game_over"
    assert board.direction == original_direction, "Board should not be modified by is_game_over"
    assert board.food_position == original_food, "Board should not be modified by is_game_over"
    assert board.score == original_score, "Board should not be modified by is_game_over"
    
    # Test calculate_reward doesn't modify anything (it's a pure function)
    reward = GameLogic.calculate_reward("EMPTY")
    assert reward == 0, "Empty move should give 0 reward"
    
    reward = GameLogic.calculate_reward("FOOD")
    assert reward == 10, "Food move should give 10 reward"
    
    reward = GameLogic.calculate_reward("WALL")
    assert reward == -10, "Wall collision should give -10 reward"
    
    reward = GameLogic.calculate_reward("SNAKE")
    assert reward == -10, "Snake collision should give -10 reward"


@given(board=game_boards())
def test_execute_move_creates_new_instances(board):
    """
    Test that execute_move always creates new GameBoard instances.
    
    This test specifically focuses on ensuring that execute_move never
    returns the same GameBoard instance, even in edge cases.
    """
    # Get a valid move
    possible_moves = GameLogic.get_possible_moves(board.direction)
    move = possible_moves[1]  # Use straight move to minimize changes
    
    # Execute the move
    result = GameLogic.execute_move(board, move)
    
    # The new board should always be a different instance
    assert result.new_board is not board, "New board should be a different instance"
    
    # Even if the move results in a collision (terminal state),
    # the returned board should still be a different instance
    if result.is_terminal:
        # For terminal states, the new board might be the same as the original
        # in terms of content, but should still be a different instance
        assert result.new_board is not board, "Even terminal results should return new instances"
    
    # Test that multiple executions create different instances
    result2 = GameLogic.execute_move(board, move)
    assert result2.new_board is not board, "Second execution should also create new instance"
    assert result2.new_board is not result.new_board, "Each execution should create unique instances"


@given(
    grid_size=st.tuples(st.integers(5, 30), st.integers(5, 30)),
    initial_length=st.integers(1, 5),
    random_seed=st.integers(0, 2**32-1)
)
def test_create_initial_board_immutability(grid_size, initial_length, random_seed):
    """
    Test that create_initial_board creates proper immutable boards.
    
    This test verifies that the initial board creation follows immutability principles.
    """
    # Create initial board
    board = GameLogic.create_initial_board(grid_size, initial_length, random_seed)
    
    # Test that the board is properly immutable
    with pytest.raises(AttributeError):
        board.score = 999
    
    with pytest.raises(AttributeError):
        board.snake_head = Position(0, 0)
    
    # Test that the board has all required properties
    assert isinstance(board.snake_head, Position), "Head should be a Position"
    assert isinstance(board.snake_body, tuple), "Body should be a tuple (immutable)"
    assert isinstance(board.direction, Direction), "Direction should be a Direction"
    assert isinstance(board.food_position, Position), "Food should be a Position"
    assert board.score == 0, "Initial score should be 0"
    assert board.grid_size == grid_size, "Grid size should match input"
    
    # Test that the snake has the correct initial length
    total_length = 1 + len(board.snake_body)  # head + body segments
    assert total_length == initial_length, f"Snake should have {initial_length} segments"


@given(
    grid_size=st.tuples(st.integers(8, 20), st.integers(8, 20)),
    initial_length=st.integers(3, 5),
    random_seed=st.integers(0, 2**32-1)
)
def test_food_termination_behavior(grid_size, initial_length, random_seed):
    """
    Feature: ai-hydra, Property: Food Termination Behavior
    
    For any game board where the snake can eat food, executing a move that 
    results in food consumption should immediately terminate the clone 
    (is_terminal=True) with a +10 reward.
    
    **Validates: Requirements 4.4 - Clone termination on food consumption**
    """
    # Create initial board
    board = GameLogic.create_initial_board(grid_size, initial_length, random_seed)
    
    # Create a scenario where food is directly in front of snake
    snake_head = board.snake_head
    direction = board.direction
    
    # Calculate position directly in front of snake
    food_position = Position(
        snake_head.x + direction.dx,
        snake_head.y + direction.dy
    )
    
    # Only test if the food position is within bounds and not occupied by snake
    if (board.is_position_within_bounds(food_position) and 
        not board.is_position_occupied_by_snake(food_position)):
        
        # Create new board with food in front of snake
        board_with_food_ahead = GameBoard(
            snake_head=board.snake_head,
            snake_body=board.snake_body,
            direction=board.direction,
            food_position=food_position,
            score=board.score,
            move_count=board.move_count,
            random_state=copy.deepcopy(board.random_state),
            grid_size=board.grid_size
        )
        
        # Create a straight move that will eat the food
        move = Move(action=MoveAction.STRAIGHT, resulting_direction=direction)
        
        # Execute the move
        result = GameLogic.execute_move(board_with_food_ahead, move)
        
        # Verify food termination behavior
        assert result.outcome == "FOOD", f"Expected FOOD outcome, got {result.outcome}"
        assert result.reward == 10, f"Expected +10 reward for food, got {result.reward}"
        assert result.is_terminal == True, f"Food consumption should terminate clone (is_terminal=True)"
        assert result.new_board.score == board.score + 1, f"Score should increase by 1"
        
        # Verify the snake grew (head became part of body, new head at food position)
        assert result.new_board.snake_head == food_position, f"Snake head should move to food position"
        assert len(result.new_board.snake_body) == len(board.snake_body) + 1, f"Snake should grow by 1 segment"
        assert result.new_board.snake_body[0] == board.snake_head, f"Old head should become first body segment"


@given(
    grid_size=st.tuples(st.integers(8, 20), st.integers(8, 20)),
    initial_length=st.integers(3, 5),
    random_seed=st.integers(0, 2**32-1)
)
def test_collision_termination_behavior(grid_size, initial_length, random_seed):
    """
    Feature: ai-hydra, Property: Collision Termination Behavior
    
    For any game board where the snake will collide (wall or self), executing 
    such a move should immediately terminate the clone (is_terminal=True) with 
    a -10 reward.
    
    **Validates: Requirements 4.3 - Clone termination on collision**
    """
    # Create initial board
    board = GameLogic.create_initial_board(grid_size, initial_length, random_seed)
    
    # Create a scenario where snake will hit a wall
    # Place snake at edge position facing the wall
    width, height = grid_size
    
    # Test wall collision - place snake at right edge facing right
    edge_board = GameBoard(
        snake_head=Position(width - 1, height // 2),  # At right edge
        snake_body=board.snake_body,
        direction=Direction.right(),  # Moving toward wall
        food_position=board.food_position,
        score=board.score,
        move_count=board.move_count,
        random_state=copy.deepcopy(board.random_state),
        grid_size=board.grid_size
    )
    
    # Create a straight move that will hit the wall
    move = Move(action=MoveAction.STRAIGHT, resulting_direction=Direction.right())
    
    # Execute the move
    result = GameLogic.execute_move(edge_board, move)
    
    # Verify collision termination behavior
    assert result.outcome == "WALL", f"Expected WALL outcome, got {result.outcome}"
    assert result.reward == -10, f"Expected -10 reward for collision, got {result.reward}"
    assert result.is_terminal == True, f"Collision should terminate clone (is_terminal=True)"
    assert result.new_board.score == edge_board.score, f"Score should not change on collision"


@given(
    grid_size=st.tuples(st.integers(8, 20), st.integers(8, 20)),
    initial_length=st.integers(3, 5),
    random_seed=st.integers(0, 2**32-1),
    max_moves_multiplier=st.integers(1, 10)
)
def test_max_moves_termination_behavior(grid_size, initial_length, random_seed, max_moves_multiplier):
    """
    Feature: ai-hydra, Property: Max Moves Termination Behavior
    
    For any game board where move_count exceeds max_moves * snake_length,
    the game should terminate with is_terminal=True and "MAX_MOVES" outcome.
    
    **Validates: Requirements 4.6 - Game termination on max moves exceeded**
    """
    # Create initial board
    board = GameLogic.create_initial_board(grid_size, initial_length, random_seed)
    
    # Calculate max moves limit
    max_moves_limit = max_moves_multiplier * board.get_snake_length()
    
    # Create a board that is at the max moves limit
    board_at_limit = GameBoard(
        snake_head=board.snake_head,
        snake_body=board.snake_body,
        direction=board.direction,
        food_position=board.food_position,
        score=board.score,
        move_count=max_moves_limit,  # At the limit
        random_state=copy.deepcopy(board.random_state),
        grid_size=board.grid_size
    )
    
    # Verify board is not yet terminated at the limit
    assert not GameLogic.is_game_over(board_at_limit, max_moves_multiplier), "Game should not be over at max moves limit"
    
    # Create a board that exceeds the max moves limit
    board_over_limit = GameBoard(
        snake_head=board.snake_head,
        snake_body=board.snake_body,
        direction=board.direction,
        food_position=board.food_position,
        score=board.score,
        move_count=max_moves_limit + 1,  # Over the limit
        random_state=copy.deepcopy(board.random_state),
        grid_size=board.grid_size
    )
    
    # Verify board is terminated when over the limit
    assert GameLogic.is_game_over(board_over_limit, max_moves_multiplier), "Game should be over when max moves exceeded"
    
    # Test executing a move that would exceed the limit
    # Create a safe move (to empty space) that would normally continue
    center_x, center_y = grid_size[0] // 2, grid_size[1] // 2
    
    safe_board_at_limit = GameBoard(
        snake_head=Position(center_x, center_y),
        snake_body=tuple(Position(center_x - i - 1, center_y) for i in range(initial_length - 1)),
        direction=Direction.right(),  # Moving right (safe direction)
        food_position=Position(0, 0),  # Food far away at corner
        score=board.score,
        move_count=max_moves_limit,  # At the limit, next move will exceed
        random_state=copy.deepcopy(board.random_state),
        grid_size=board.grid_size
    )
    
    # Create a straight move to empty space
    move = Move(action=MoveAction.STRAIGHT, resulting_direction=Direction.right())
    
    # Execute the move that will exceed max moves
    result = GameLogic.execute_move(safe_board_at_limit, move, max_moves_multiplier)
    
    # Verify max moves termination behavior
    assert result.outcome == "MAX_MOVES", f"Expected MAX_MOVES outcome, got {result.outcome}"
    assert result.reward == 0, f"Expected 0 reward for max moves, got {result.reward}"
    assert result.is_terminal == True, f"Max moves exceeded should terminate game (is_terminal=True)"
    assert result.new_board.move_count == max_moves_limit + 1, f"Move count should be incremented"
    assert result.new_board.score == safe_board_at_limit.score, f"Score should not change on max moves"


@given(
    grid_size=st.tuples(st.integers(8, 20), st.integers(8, 20)),
    initial_length=st.integers(3, 5),
    random_seed=st.integers(0, 2**32-1),
    max_moves_multiplier=st.integers(2, 5),
    move_sequence_length=st.integers(1, 10)
)
def test_max_moves_calculation_property(grid_size, initial_length, random_seed, max_moves_multiplier, move_sequence_length):
    """
    Feature: ai-hydra, Property: Max Moves Calculation
    
    For any game configuration, the max moves limit should be calculated as
    max_moves_multiplier * current_snake_length, and should be checked after
    each move execution.
    
    **Validates: Requirements 4.6 - Max moves calculation and checking**
    """
    # Create initial board
    board = GameLogic.create_initial_board(grid_size, initial_length, random_seed)
    
    # Track moves and verify max moves calculation at each step
    current_board = board
    moves_executed = 0
    
    for i in range(min(move_sequence_length, max_moves_multiplier * initial_length)):
        # Calculate current max moves limit
        current_snake_length = current_board.get_snake_length()
        max_moves_limit = max_moves_multiplier * current_snake_length
        
        # Verify the board correctly reports game over status
        should_be_over = current_board.move_count > max_moves_limit
        is_actually_over = GameLogic.is_game_over(current_board, max_moves_multiplier)
        
        if should_be_over:
            assert is_actually_over, f"Game should be over when move_count ({current_board.move_count}) > max_moves_limit ({max_moves_limit})"
            break
        else:
            assert not is_actually_over or current_board.move_count == max_moves_limit, f"Game should not be over when move_count ({current_board.move_count}) <= max_moves_limit ({max_moves_limit})"
        
        # Execute a safe move if possible
        center_x, center_y = grid_size[0] // 2, grid_size[1] // 2
        safe_position = Position(center_x + i, center_y)
        
        # Only continue if we have safe space
        if current_board.is_position_within_bounds(safe_position):
            safe_board = GameBoard(
                snake_head=Position(center_x + i, center_y),
                snake_body=current_board.snake_body,
                direction=Direction.right(),
                food_position=Position(0, 0),  # Food far away
                score=current_board.score,
                move_count=current_board.move_count,
                random_state=copy.deepcopy(current_board.random_state),
                grid_size=current_board.grid_size
            )
            
            move = Move(action=MoveAction.STRAIGHT, resulting_direction=Direction.right())
            result = GameLogic.execute_move(safe_board, move, max_moves_multiplier)
            
            current_board = result.new_board
            moves_executed += 1
            
            # Verify move count incremented
            assert current_board.move_count == board.move_count + moves_executed, f"Move count should increment with each move"
        else:
            break


@given(
    grid_size=st.tuples(st.integers(8, 20), st.integers(8, 20)),
    initial_length=st.integers(3, 5),
    random_seed=st.integers(0, 2**32-1)
)
def test_empty_move_continuation_behavior(grid_size, initial_length, random_seed):
    """
    Feature: ai-hydra, Property: Empty Move Continuation Behavior
    
    For any game board where the snake moves to an empty square (no food, no collision),
    the move should NOT terminate the clone (is_terminal=False) with 0 reward.
    
    **Validates: Requirements 4.5 - Clone continuation on empty moves**
    """
    # Create initial board with plenty of space
    board = GameLogic.create_initial_board(grid_size, initial_length, random_seed)
    
    # Ensure we have a safe move (not hitting food or walls)
    # Move snake to center with food far away
    center_x, center_y = grid_size[0] // 2, grid_size[1] // 2
    
    safe_board = GameBoard(
        snake_head=Position(center_x, center_y),
        snake_body=tuple(Position(center_x - i - 1, center_y) for i in range(initial_length - 1)),
        direction=Direction.right(),  # Moving right (safe direction)
        food_position=Position(0, 0),  # Food far away at corner
        score=board.score,
        move_count=board.move_count,
        random_state=copy.deepcopy(board.random_state),
        grid_size=board.grid_size
    )
    
    # Create a straight move to empty space
    move = Move(action=MoveAction.STRAIGHT, resulting_direction=Direction.right())
    
    # Execute the move
    result = GameLogic.execute_move(safe_board, move)
    
    # Verify empty move continuation behavior
    assert result.outcome == "EMPTY", f"Expected EMPTY outcome, got {result.outcome}"
    assert result.reward == 0, f"Expected 0 reward for empty move, got {result.reward}"
    assert result.is_terminal == False, f"Empty move should NOT terminate clone (is_terminal=False)"
    assert result.new_board.score == safe_board.score, f"Score should not change on empty move"
    
    # Verify snake moved correctly (tail removed, head advanced)
    expected_new_head = Position(center_x + 1, center_y)
    assert result.new_board.snake_head == expected_new_head, f"Snake head should advance"
    assert len(result.new_board.snake_body) == len(safe_board.snake_body), f"Snake length should stay same"
    assert result.new_board.snake_body[0] == safe_board.snake_head, f"Old head should become first body segment"