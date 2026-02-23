"""
Feature extraction for neural network input.

This module converts GameBoard state into standardized feature vectors
for neural network processing. The feature vector contains 19 features
including collision detection, direction flags, food position, and snake metrics.
"""

import torch
from typing import List
from .models import GameBoard, Position, Direction


class FeatureExtractor:
    """
    Extracts features from GameBoard for neural network input.
    
    Converts game state into a 19-dimensional feature vector containing:
    - Snake collision detection (3 features: straight, left, right)
    - Wall collision detection (3 features: straight, left, right) 
    - Current direction flags (4 features: up, down, left, right)
    - Food relative position (2 features: normalized dx, dy)
    - Snake length binary representation (7 features: up to 127 length)
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_features(self, board: GameBoard) -> torch.Tensor:
        """
        Extract complete feature vector from GameBoard.
        
        Args:
            board: GameBoard instance to extract features from
            
        Returns:
            torch.Tensor: 19-dimensional feature vector
        """
        features = []
        
        # Snake collision features (3 features)
        snake_collisions = self.get_collision_features(board, check_snake=True)
        features.extend(snake_collisions)
        
        # Wall collision features (3 features)  
        wall_collisions = self.get_collision_features(board, check_snake=False)
        features.extend(wall_collisions)
        
        # Direction features (4 features)
        direction_features = self.get_direction_features(board)
        features.extend(direction_features)
        
        # Food relative position features (2 features)
        food_features = self.get_food_features(board)
        features.extend(food_features)
        
        # Snake length binary features (7 features)
        snake_features = self.get_snake_features(board)
        features.extend(snake_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_collision_features(self, board: GameBoard, check_snake: bool = True) -> List[bool]:
        """
        Get collision detection features for three directions relative to current direction.
        
        Args:
            board: GameBoard to check collisions for
            check_snake: If True, check snake collisions; if False, check wall collisions
            
        Returns:
            List[bool]: [straight_collision, left_collision, right_collision]
        """
        current_dir = board.get_direction()
        head_pos = board.get_snake_head()
        
        # Calculate the three possible next positions
        straight_pos = Position(head_pos.x + current_dir.dx, head_pos.y + current_dir.dy)
        
        left_dir = current_dir.turn_left()
        left_pos = Position(head_pos.x + left_dir.dx, head_pos.y + left_dir.dy)
        
        right_dir = current_dir.turn_right()
        right_pos = Position(head_pos.x + right_dir.dx, head_pos.y + right_dir.dy)
        
        positions = [straight_pos, left_pos, right_pos]
        collisions = []
        
        for pos in positions:
            if check_snake:
                # Check snake body collision (excluding head)
                collision = pos in board.get_snake_body()
            else:
                # Check wall collision
                collision = not board.is_position_within_bounds(pos)
            collisions.append(collision)
        
        return collisions
    
    def get_direction_features(self, board: GameBoard) -> List[bool]:
        """
        Get current direction as one-hot encoded features.
        
        Args:
            board: GameBoard to get direction from
            
        Returns:
            List[bool]: [is_up, is_down, is_left, is_right]
        """
        current_dir = board.get_direction()
        
        return [
            current_dir == Direction.up(),
            current_dir == Direction.down(), 
            current_dir == Direction.left(),
            current_dir == Direction.right()
        ]
    
    def get_food_features(self, board: GameBoard) -> List[float]:
        """
        Get food position relative to snake head, normalized.
        
        Args:
            board: GameBoard to get food position from
            
        Returns:
            List[float]: [normalized_dx, normalized_dy] in range [-1.0, 1.0]
        """
        head_pos = board.get_snake_head()
        food_pos = board.get_food_position()
        grid_width, grid_height = board.get_grid_size()
        
        # Calculate relative position
        dx = food_pos.x - head_pos.x
        dy = food_pos.y - head_pos.y
        
        # Normalize to [-1.0, 1.0] range
        normalized_dx = dx / (grid_width / 2.0)
        normalized_dy = dy / (grid_height / 2.0)
        
        # Clamp to [-1.0, 1.0] range
        normalized_dx = max(-1.0, min(1.0, normalized_dx))
        normalized_dy = max(-1.0, min(1.0, normalized_dy))
        
        return [normalized_dx, normalized_dy]
    
    def get_snake_features(self, board: GameBoard) -> List[bool]:
        """
        Get snake length as 7-bit binary representation.
        
        Args:
            board: GameBoard to get snake length from
            
        Returns:
            List[bool]: 7-bit binary representation of snake length (supports up to 127)
        """
        snake_length = len(board.get_all_snake_positions())
        
        # Convert to 7-bit binary (supports lengths up to 127)
        binary_bits = []
        for i in range(7):
            bit = (snake_length >> i) & 1
            binary_bits.append(bool(bit))
        
        return binary_bits