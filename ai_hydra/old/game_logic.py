"""
Game logic module for Snake Game mechanics.

This module provides pure functions for game mechanics including move execution,
collision detection, and reward calculation. All functions operate on immutable
GameBoard instances and return new instances.
"""

from typing import List, Tuple, Optional
import random
import copy
from .models import GameBoard, Position, Direction, Move, MoveAction, MoveResult


class GameLogic:
    """
    Static methods for Snake Game mechanics and move execution.
    
    This class provides pure functions that operate on GameBoard instances
    without modifying them, maintaining immutability principles throughout
    the game logic operations.
    """
    
    @staticmethod
    def execute_move(board: GameBoard, move: Move, max_moves_multiplier: int = 100) -> MoveResult:
        """
        Execute a move on the game board and return the result.
        
        This method applies a move to the game board, handling snake movement,
        collision detection, food consumption, move counting, and score updates.
        
        Args:
            board: The current game board state
            move: The move to execute
            max_moves_multiplier: Multiplier for max moves calculation
            
        Returns:
            MoveResult: The result of executing the move
        """
        # Increment move count first
        new_move_count = board.move_count + 1
        
        # Check if max moves exceeded BEFORE executing the move
        snake_length = board.get_snake_length()
        max_moves = max_moves_multiplier * snake_length
        if new_move_count > max_moves:
            # Create new board with incremented move count
            max_moves_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                random_state=copy.deepcopy(board.random_state),
                grid_size=board.grid_size
            )
            return MoveResult(
                new_board=max_moves_board,
                reward=0,  # No reward for exceeding max moves
                outcome="MAX_MOVES",
                is_terminal=True
            )
        
        # Calculate new head position
        new_head = Position(
            board.snake_head.x + move.resulting_direction.dx,
            board.snake_head.y + move.resulting_direction.dy
        )
        
        # Check for wall collision
        if not board.is_position_within_bounds(new_head):
            # Create new board instance even for collision (immutability principle)
            collision_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                random_state=copy.deepcopy(board.random_state),
                grid_size=board.grid_size
            )
            return MoveResult(
                new_board=collision_board,
                reward=-10,  # Collision penalty
                outcome="WALL",
                is_terminal=True
            )
        
        # Check for self-collision
        if board.is_position_occupied_by_snake(new_head):
            # Create new board instance even for collision (immutability principle)
            collision_board = GameBoard(
                snake_head=board.snake_head,
                snake_body=board.snake_body,
                direction=board.direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                random_state=copy.deepcopy(board.random_state),
                grid_size=board.grid_size
            )
            return MoveResult(
                new_board=collision_board,
                reward=-10,  # Collision penalty
                outcome="SNAKE",
                is_terminal=True
            )
        
        # Check if food is eaten
        ate_food = new_head == board.food_position
        
        if ate_food:
            # Snake grows - keep all body segments
            new_body = (board.snake_head,) + board.snake_body
            new_score = board.score + 1
            
            # Create a copy of the random state for the new board to maintain immutability
            new_random_state = copy.deepcopy(board.random_state)
            
            # Generate new food position using the copied random state
            new_food_position = GameLogic._generate_food_position(
                new_head, new_body, board.grid_size, new_random_state
            )
            
            # Create new board with grown snake and new food
            new_board = GameBoard(
                snake_head=new_head,
                snake_body=new_body,
                direction=move.resulting_direction,
                food_position=new_food_position,
                score=new_score,
                move_count=new_move_count,
                random_state=new_random_state,
                grid_size=board.grid_size
            )
            
            return MoveResult(
                new_board=new_board,
                reward=10,  # Food reward
                outcome="FOOD",
                is_terminal=True  # Clone terminates immediately when food is eaten (optimal outcome)
            )
        else:
            # Snake moves - remove tail segment
            new_body = (board.snake_head,) + board.snake_body[:-1]
            
            # Create new board with moved snake (copy random state for immutability)
            new_board = GameBoard(
                snake_head=new_head,
                snake_body=new_body,
                direction=move.resulting_direction,
                food_position=board.food_position,
                score=board.score,
                move_count=new_move_count,
                random_state=copy.deepcopy(board.random_state),
                grid_size=board.grid_size
            )
            
            return MoveResult(
                new_board=new_board,
                reward=0,  # Empty move reward
                outcome="EMPTY",
                is_terminal=False
            )
    
    @staticmethod
    def _generate_food_position(head: Position, body: Tuple[Position, ...], 
                               grid_size: Tuple[int, int], 
                               random_state: random.Random) -> Position:
        """
        Generate a new food position that doesn't overlap with the snake.
        
        Args:
            head: Snake head position
            body: Snake body positions
            grid_size: Grid dimensions
            random_state: Random state for deterministic behavior
            
        Returns:
            Position: New food position
        """
        width, height = grid_size
        snake_positions = {head} | set(body)
        
        # Find all available positions
        available_positions = []
        for x in range(width):
            for y in range(height):
                pos = Position(x, y)
                if pos not in snake_positions:
                    available_positions.append(pos)
        
        # If no positions available (shouldn't happen in normal gameplay)
        if not available_positions:
            return Position(0, 0)
        
        # Select random position
        return random_state.choice(available_positions)
    
    @staticmethod
    def create_move(current_direction: Direction, action: MoveAction) -> Move:
        """
        Create a move from the current direction and action.
        
        Args:
            current_direction: Current snake direction
            action: Move action to perform
            
        Returns:
            Move: The resulting move
        """
        if action == MoveAction.LEFT_TURN:
            resulting_direction = current_direction.turn_left()
        elif action == MoveAction.RIGHT_TURN:
            resulting_direction = current_direction.turn_right()
        else:  # STRAIGHT
            resulting_direction = current_direction
        
        return Move(action=action, resulting_direction=resulting_direction)
    
    @staticmethod
    def get_possible_moves(current_direction: Direction) -> List[Move]:
        """
        Get all possible moves from the current direction.
        
        Args:
            current_direction: Current snake direction
            
        Returns:
            List[Move]: All possible moves
        """
        return [
            GameLogic.create_move(current_direction, MoveAction.LEFT_TURN),
            GameLogic.create_move(current_direction, MoveAction.STRAIGHT),
            GameLogic.create_move(current_direction, MoveAction.RIGHT_TURN),
        ]
    
    @staticmethod
    def is_max_moves_exceeded(board: GameBoard, max_moves_multiplier: int = 100) -> bool:
        """
        Check if the maximum moves limit has been exceeded.
        
        Args:
            board: Game board to check
            max_moves_multiplier: Multiplier for calculating max moves limit
            
        Returns:
            bool: True if move count exceeds max_moves_multiplier * snake_length
        """
        snake_length = board.get_snake_length()
        max_moves = max_moves_multiplier * snake_length
        return board.move_count > max_moves
    
    @staticmethod
    def is_game_over(board: GameBoard, max_moves_multiplier: int = 100) -> bool:
        """
        Check if the game is in a terminal state.
        
        Args:
            board: Game board to check
            max_moves_multiplier: Multiplier for max moves calculation
            
        Returns:
            bool: True if game is over
        """
        # Check if snake head is out of bounds
        if not board.is_position_within_bounds(board.snake_head):
            return True
        
        # Check if snake head collides with body
        if board.snake_head in board.snake_body:
            return True
        
        # Check if maximum moves exceeded
        if GameLogic.is_max_moves_exceeded(board, max_moves_multiplier):
            return True
        
        return False
    
    @staticmethod
    def calculate_reward(outcome: str) -> int:
        """
        Calculate reward based on move outcome.
        
        Args:
            outcome: Move outcome ("EMPTY", "FOOD", "WALL", "SNAKE", "MAX_MOVES")
            
        Returns:
            int: Reward value
        """
        reward_map = {
            "EMPTY": 0,
            "FOOD": 10,
            "WALL": -10,
            "SNAKE": -10,
            "MAX_MOVES": 0,  # No reward for exceeding max moves
        }
        return reward_map.get(outcome, 0)
    
    @staticmethod
    def create_initial_board(grid_size: Tuple[int, int], 
                           initial_length: int = 3,
                           random_seed: int = 42) -> GameBoard:
        """
        Create an initial game board with default snake and food placement.
        
        Args:
            grid_size: Dimensions of the game grid
            initial_length: Initial length of the snake
            random_seed: Seed for random number generation
            
        Returns:
            GameBoard: Initial game board
        """
        width, height = grid_size
        random_state = random.Random(random_seed)
        
        # Place snake in the center, moving right
        center_x, center_y = width // 2, height // 2
        snake_head = Position(center_x, center_y)
        
        # Create initial body segments to the left of the head
        snake_body = tuple(
            Position(center_x - i - 1, center_y) 
            for i in range(initial_length - 1)
        )
        
        # Initial direction is right
        direction = Direction.right()
        
        # Generate initial food position
        food_position = GameLogic._generate_food_position(
            snake_head, snake_body, grid_size, random_state
        )
        
        return GameBoard(
            snake_head=snake_head,
            snake_body=snake_body,
            direction=direction,
            food_position=food_position,
            score=0,
            move_count=0,  # Start with 0 moves
            random_state=random_state,
            grid_size=grid_size
        )