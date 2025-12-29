"""
Property-based tests for efficiency-based path selection.

This module tests the property that among equal-reward paths, 
the system selects the path with the fewest moves to promote efficiency.
"""

import pytest
from hypothesis import given, strategies as st, settings
from ai_hydra.models import ExplorationPath, Move, MoveAction, Direction
from ai_hydra.hydra_mgr import HydraMgr
from ai_hydra.config import SimulationConfig


class MockLogger:
    """Mock logger to avoid initialization issues in tests."""
    def log_system_event(self, *args, **kwargs):
        pass


@st.composite
def exploration_paths_with_ties(draw):
    """Generate exploration paths where some have equal rewards but different lengths."""
    # Generate a common reward value
    common_reward = draw(st.integers(min_value=-50, max_value=50))
    
    # Generate 2-5 paths with the same reward but different lengths
    num_tied_paths = draw(st.integers(min_value=2, max_value=5))
    
    # Generate different path lengths for tied paths
    path_lengths = draw(st.lists(
        st.integers(min_value=1, max_value=10),
        min_size=num_tied_paths,
        max_size=num_tied_paths,
        unique=True  # Ensure different lengths
    ))
    
    tied_paths = []
    for i, length in enumerate(path_lengths):
        # Create moves for this path length
        moves = []
        for j in range(length):
            action = draw(st.sampled_from([MoveAction.LEFT_TURN, MoveAction.STRAIGHT, MoveAction.RIGHT_TURN]))
            direction = draw(st.sampled_from([Direction.up(), Direction.down(), Direction.left(), Direction.right()]))
            moves.append(Move(action, direction))
        
        path = ExplorationPath(
            moves=moves,
            cumulative_reward=common_reward,
            clone_id=f"tied_{i}",
            parent_id=f"parent_{i}",
            depth=length,
            is_complete=True
        )
        tied_paths.append(path)
    
    # Optionally add some paths with different rewards
    num_other_paths = draw(st.integers(min_value=0, max_value=3))
    other_paths = []
    
    for i in range(num_other_paths):
        # Different reward (not equal to common_reward)
        other_reward = draw(st.integers(min_value=-50, max_value=50).filter(lambda x: x != common_reward))
        other_length = draw(st.integers(min_value=1, max_value=5))
        
        moves = []
        for j in range(other_length):
            action = draw(st.sampled_from([MoveAction.LEFT_TURN, MoveAction.STRAIGHT, MoveAction.RIGHT_TURN]))
            direction = draw(st.sampled_from([Direction.up(), Direction.down(), Direction.left(), Direction.right()]))
            moves.append(Move(action, direction))
        
        path = ExplorationPath(
            moves=moves,
            cumulative_reward=other_reward,
            parent_id=f"other_parent_{i}",
            clone_id=f"other_{i}",
            depth=other_length,
            is_complete=True
        )
        other_paths.append(path)
    
    all_paths = tied_paths + other_paths
    
    # Shuffle the paths to ensure order doesn't matter
    draw(st.randoms()).shuffle(all_paths)
    
    return all_paths, tied_paths, common_reward


class TestEfficiencyBasedSelection:
    """Property-based tests for efficiency-based path selection."""
    
    @given(exploration_paths_with_ties())
    @settings(max_examples=50, deadline=2000)
    def test_efficiency_based_selection_property(self, path_data):
        """
        **Property: Efficiency-based path selection**
        **Validates: Requirements 5.3**
        
        For any set of exploration paths where multiple paths have the same highest reward,
        the system should select the path with the fewest moves to promote efficiency.
        """
        all_paths, tied_paths, common_reward = path_data
        
        # Skip if we don't have enough paths to test
        if len(all_paths) < 2:
            return
        
        # Create HydraMgr instance for testing
        config = SimulationConfig(grid_size=(10, 10), move_budget=25, random_seed=42)
        hydra_mgr = HydraMgr(config)
        hydra_mgr.logger = MockLogger()  # Use mock logger
        
        # Find the actual highest reward among all paths
        max_reward = max(path.cumulative_reward for path in all_paths)
        
        # Get all paths with the highest reward
        best_paths = [path for path in all_paths if path.cumulative_reward == max_reward]
        
        # If there's only one best path, efficiency selection doesn't apply
        if len(best_paths) == 1:
            selected_path = hydra_mgr.evaluate_exploration_paths(all_paths)
            assert selected_path == best_paths[0]
            return
        
        # Multiple paths with same highest reward - efficiency selection should apply
        selected_path = hydra_mgr.evaluate_exploration_paths(all_paths)
        
        # Verify the selected path has the highest reward
        assert selected_path.cumulative_reward == max_reward
        
        # Verify the selected path has the minimum length among tied paths
        min_length = min(len(path.moves) for path in best_paths)
        assert len(selected_path.moves) == min_length
        
        # Verify the selected path is one of the best paths
        assert selected_path in best_paths
    
    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=1000)
    def test_tie_breaking_with_identical_rewards(self, num_paths):
        """
        Test tie-breaking when all paths have identical rewards but different lengths.
        
        This is a focused test for the core efficiency-based selection logic.
        """
        # Create paths with identical rewards but different lengths
        common_reward = 15
        paths = []
        
        for i in range(num_paths):
            path_length = i + 1  # Lengths: 1, 2, 3, ..., num_paths
            
            # Create moves for this path
            moves = []
            for j in range(path_length):
                moves.append(Move(MoveAction.STRAIGHT, Direction.up()))
            
            path = ExplorationPath(
                moves=moves,
                cumulative_reward=common_reward,
                clone_id=f"path_{i}",
                parent_id=f"parent_{i}",
                depth=path_length,
                is_complete=True
            )
            paths.append(path)
        
        # Create HydraMgr instance
        config = SimulationConfig(grid_size=(10, 10), move_budget=25, random_seed=42)
        hydra_mgr = HydraMgr(config)
        hydra_mgr.logger = MockLogger()
        
        # Test efficiency-based selection
        selected_path = hydra_mgr.evaluate_exploration_paths(paths)
        
        # Should select the path with length 1 (shortest)
        assert len(selected_path.moves) == 1
        assert selected_path.cumulative_reward == common_reward
        assert selected_path.clone_id == "path_0"  # First path has length 1
    
    def test_no_tie_breaking_when_rewards_differ(self):
        """
        Test that efficiency-based selection only applies when rewards are tied.
        
        When paths have different rewards, the highest reward should win regardless of length.
        """
        paths = [
            ExplorationPath(
                moves=[Move(MoveAction.STRAIGHT, Direction.up())],  # Length 1
                cumulative_reward=5,
                clone_id="short_low",
                parent_id="parent1",
                depth=1,
                is_complete=True
            ),
            ExplorationPath(
                moves=[  # Length 3
                    Move(MoveAction.LEFT_TURN, Direction.left()),
                    Move(MoveAction.STRAIGHT, Direction.up()),
                    Move(MoveAction.RIGHT_TURN, Direction.right())
                ],
                cumulative_reward=20,  # Higher reward
                clone_id="long_high",
                parent_id="parent2",
                depth=3,
                is_complete=True
            )
        ]
        
        # Create HydraMgr instance
        config = SimulationConfig(grid_size=(10, 10), move_budget=25, random_seed=42)
        hydra_mgr = HydraMgr(config)
        hydra_mgr.logger = MockLogger()
        
        # Should select the path with higher reward, not shorter length
        selected_path = hydra_mgr.evaluate_exploration_paths(paths)
        
        assert selected_path.clone_id == "long_high"
        assert selected_path.cumulative_reward == 20
        assert len(selected_path.moves) == 3  # Longer path won due to higher reward
    
    def test_efficiency_selection_with_mixed_rewards(self):
        """
        Test efficiency selection in a mixed scenario with both tied and untied rewards.
        """
        paths = [
            # Two paths with reward 10 but different lengths
            ExplorationPath(
                moves=[Move(MoveAction.STRAIGHT, Direction.up())],  # Length 1
                cumulative_reward=10,
                clone_id="tied_short",
                parent_id="parent1",
                depth=1,
                is_complete=True
            ),
            ExplorationPath(
                moves=[  # Length 2
                    Move(MoveAction.LEFT_TURN, Direction.left()),
                    Move(MoveAction.STRAIGHT, Direction.up())
                ],
                cumulative_reward=10,
                clone_id="tied_long",
                parent_id="parent2",
                depth=2,
                is_complete=True
            ),
            # One path with lower reward
            ExplorationPath(
                moves=[Move(MoveAction.RIGHT_TURN, Direction.right())],  # Length 1
                cumulative_reward=5,
                clone_id="lower_reward",
                parent_id="parent3",
                depth=1,
                is_complete=True
            )
        ]
        
        # Create HydraMgr instance
        config = SimulationConfig(grid_size=(10, 10), move_budget=25, random_seed=42)
        hydra_mgr = HydraMgr(config)
        hydra_mgr.logger = MockLogger()
        
        # Should select the shorter path among the tied highest-reward paths
        selected_path = hydra_mgr.evaluate_exploration_paths(paths)
        
        assert selected_path.clone_id == "tied_short"
        assert selected_path.cumulative_reward == 10
        assert len(selected_path.moves) == 1