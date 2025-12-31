"""
Property-based tests for deterministic reproducibility.

This module implements Property 11: Deterministic Reproducibility
**Validates: Requirements 7.1, 7.2, 7.5**
"""

import pytest
from hypothesis import given, strategies as st, settings
import torch
import numpy as np
import random

from ai_hydra.game_logic import GameLogic
from ai_hydra.config import ReproducibilityConfig, ConfigurationManager


class TestDeterministicReproducibility:
    """Property-based tests for deterministic reproducibility."""
    
    @given(
        seed=st.integers(min_value=0, max_value=1000),
        grid_size=st.tuples(
            st.integers(min_value=8, max_value=12),
            st.integers(min_value=8, max_value=12)
        )
    )
    @settings(max_examples=10, deadline=5000)
    def test_deterministic_reproducibility_property(self, seed, grid_size):
        """
        **Feature: ai-hydra, Property 11: Deterministic Reproducibility**
        
        *For any* identical seed configuration, multiple simulation runs should produce 
        identical game sequences and decision patterns when using the same GameBoard 
        initialization and Game_Logic operations
        **Validates: Requirements 7.1, 7.2, 7.5**
        """
        # Test core deterministic reproducibility: identical seeds produce identical results
        
        # Setup reproducibility configuration
        repro_config = ReproducibilityConfig(
            master_seed=seed,
            use_deterministic_algorithms=True,
            benchmark_mode=False
        )
        
        # Run 1: Create initial board with seed
        ConfigurationManager.setup_reproducibility(repro_config)
        board1 = GameLogic.create_initial_board(grid_size, 3, seed)
        
        # Run 2: Reset and create identical board with same seed
        ConfigurationManager.setup_reproducibility(repro_config)
        board2 = GameLogic.create_initial_board(grid_size, 3, seed)
        
        # Boards should be identical
        assert board1.snake_head == board2.snake_head, \
            f"Snake heads should be identical: {board1.snake_head} != {board2.snake_head}"
        
        assert board1.snake_body == board2.snake_body, \
            f"Snake bodies should be identical: {board1.snake_body} != {board2.snake_body}"
        
        assert board1.direction == board2.direction, \
            f"Directions should be identical: {board1.direction} != {board2.direction}"
        
        assert board1.food_position == board2.food_position, \
            f"Food positions should be identical: {board1.food_position} != {board2.food_position}"
        
        assert board1.score == board2.score, \
            f"Scores should be identical: {board1.score} != {board2.score}"
        
        # Test move sequence reproducibility
        current_board1 = board1
        current_board2 = board2
        
        # Execute 5 moves with same sequence
        for move_num in range(5):
            moves1 = GameLogic.get_possible_moves(current_board1.direction)
            moves2 = GameLogic.get_possible_moves(current_board2.direction)
            
            # Should have identical possible moves
            assert moves1 == moves2, f"Move {move_num}: possible moves should be identical"
            
            if not moves1:  # No moves available
                break
                
            # Use first move for determinism
            move = moves1[0]
            
            result1 = GameLogic.execute_move(current_board1, move)
            result2 = GameLogic.execute_move(current_board2, move)
            
            # Results should be identical
            assert result1.outcome == result2.outcome, \
                f"Move {move_num}: outcomes should be identical: {result1.outcome} != {result2.outcome}"
            
            assert result1.reward == result2.reward, \
                f"Move {move_num}: rewards should be identical: {result1.reward} != {result2.reward}"
            
            assert result1.is_terminal == result2.is_terminal, \
                f"Move {move_num}: terminal states should be identical"
            
            # New board states should be identical
            assert result1.new_board.snake_head == result2.new_board.snake_head, \
                f"Move {move_num}: new snake heads should be identical"
            
            assert result1.new_board.food_position == result2.new_board.food_position, \
                f"Move {move_num}: new food positions should be identical"
            
            assert result1.new_board.score == result2.new_board.score, \
                f"Move {move_num}: new scores should be identical"
            
            # Update for next iteration
            current_board1 = result1.new_board
            current_board2 = result2.new_board
            
            # Stop if game ended
            if result1.is_terminal:
                break
    
    def test_random_state_isolation(self):
        """
        Test that random state is properly isolated between simulations.
        
        This ensures that each simulation run starts with a clean random state
        and doesn't interfere with other simulations.
        """
        seed = 12345
        
        # Setup reproducibility and capture initial random states
        repro_config = ReproducibilityConfig(master_seed=seed)
        ConfigurationManager.setup_reproducibility(repro_config)
        
        initial_py_random = random.random()
        initial_np_random = np.random.random()
        initial_torch_random = torch.rand(1).item()
        
        # Reset and capture again - should be identical
        ConfigurationManager.setup_reproducibility(repro_config)
        
        reset_py_random = random.random()
        reset_np_random = np.random.random()
        reset_torch_random = torch.rand(1).item()
        
        assert initial_py_random == reset_py_random, "Python random should be deterministic"
        assert initial_np_random == reset_np_random, "NumPy random should be deterministic"
        assert initial_torch_random == reset_torch_random, "PyTorch random should be deterministic"
    
    def test_seed_sequence_determinism(self):
        """
        Test that seed sequences are generated deterministically.
        
        This ensures that the same master seed always produces the same
        sequence of seeds for multi-run experiments.
        """
        master_seed = 98765
        
        # Create multiple reproducibility configs with same master seed
        config1 = ReproducibilityConfig(master_seed=master_seed)
        config2 = ReproducibilityConfig(master_seed=master_seed)
        config3 = ReproducibilityConfig(master_seed=master_seed)
        
        # Seed sequences should be identical
        assert config1.seed_sequence == config2.seed_sequence, "Seed sequences should be identical"
        assert config2.seed_sequence == config3.seed_sequence, "Seed sequences should be identical"
        
        # Different master seed should produce different sequence
        config4 = ReproducibilityConfig(master_seed=master_seed + 1)
        assert config1.seed_sequence != config4.seed_sequence, "Different master seeds should produce different sequences"
    
    @given(
        seed1=st.integers(min_value=0, max_value=1000),
        seed2=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=5)
    def test_different_seeds_produce_different_results(self, seed1, seed2):
        """
        Test that different seeds produce different results (negative test).
        
        This ensures that the deterministic system actually responds to
        different seed values by producing different outcomes.
        """
        # Skip if seeds are the same
        if seed1 == seed2:
            return
        
        grid_size = (8, 8)
        
        # Create boards with different seeds
        board1 = GameLogic.create_initial_board(grid_size, 3, seed1)
        board2 = GameLogic.create_initial_board(grid_size, 3, seed2)
        
        # At least one aspect should be different (food position is most likely)
        # Note: Snake position might be the same, but food position should vary
        different_aspects = [
            board1.snake_head != board2.snake_head,
            board1.food_position != board2.food_position,
            board1.direction != board2.direction
        ]
        
        # At least one aspect should be different with high probability
        # (This is a probabilistic test - very rarely might fail by chance)
        assert any(different_aspects), \
            f"Different seeds should produce different initial states. " \
            f"Seed1: {seed1}, Seed2: {seed2}, " \
            f"Board1 food: {board1.food_position}, Board2 food: {board2.food_position}"