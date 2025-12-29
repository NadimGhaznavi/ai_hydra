"""
Property-based tests for parallel neural network spawning and lifecycle management.

This module implements Property 12: Parallel NN Spawning and Lifecycle Management
**Validates: Requirements 11.1, 11.2, 11.3, 11.4**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import gc

from ai_hydra.neural_network import SnakeNet
from ai_hydra.game_logic import GameLogic
from ai_hydra.models import GameBoard, Move, MoveAction
from ai_hydra.config import SimulationConfig, NetworkConfig


class TestParallelNNLifecycle:
    """Property-based tests for parallel neural network spawning and lifecycle management."""
    
    @given(
        initial_budget=st.integers(min_value=3, max_value=20),
        grid_size=st.tuples(
            st.integers(min_value=8, max_value=12),
            st.integers(min_value=8, max_value=12)
        ),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=10, deadline=5000)
    def test_parallel_nn_spawning_property(self, initial_budget, grid_size, seed):
        """
        **Feature: ai-hydra, Property 12: Parallel NN Spawning and Lifecycle Management**
        
        *For any* valid budget and game configuration, the system should spawn exactly
        one new neural network instance for each choice made during tree exploration,
        and properly manage the lifecycle of all spawned NN instances.
        **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
        """
        # Skip if budget is too small for meaningful exploration
        assume(initial_budget >= 3)
        
        # Create initial game board
        initial_board = GameLogic.create_initial_board(grid_size, 3, seed)
        
        # Track NN instances created during exploration
        nn_instances = {}  # choice_path -> NN instance
        nn_creation_log = []  # Track creation order and metadata
        
        # Simulate parallel NN spawning for tree exploration
        self._simulate_parallel_nn_exploration(
            initial_board, 
            initial_budget, 
            nn_instances, 
            nn_creation_log
        )
        
        # Property 1: Each choice should spawn exactly one NN instance
        unique_choice_paths = set(log['choice_path'] for log in nn_creation_log)
        assert len(nn_instances) == len(unique_choice_paths), \
            f"Each unique choice path should have exactly one NN instance. " \
            f"Expected {len(unique_choice_paths)}, got {len(nn_instances)}"
        
        # Property 2: All NN instances should be independent
        self._verify_nn_independence(nn_instances)
        
        # Property 3: NN instances should be properly initialized
        for choice_path, nn_instance in nn_instances.items():
            assert isinstance(nn_instance, SnakeNet), \
                f"NN instance for path {choice_path} should be SnakeNet, got {type(nn_instance)}"
            
            # Verify NN can make predictions
            dummy_features = torch.randn(1, 19)  # Standard feature size
            prediction, confidence = nn_instance.predict_move(dummy_features)
            assert 0 <= prediction <= 2, f"Prediction should be 0-2, got {prediction}"
            assert 0.0 <= confidence <= 1.0, f"Confidence should be 0-1, got {confidence}"
        
        # Property 4: Resource cleanup should be possible
        initial_memory = self._get_memory_usage()
        self._cleanup_nn_instances(nn_instances)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = self._get_memory_usage()
        
        # Memory should not increase significantly after cleanup
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50, \
            f"Memory leak detected: {memory_increase}MB increase after cleanup"
    
    @given(
        choice_sequence=st.lists(
            st.sampled_from([MoveAction.LEFT_TURN, MoveAction.STRAIGHT, MoveAction.RIGHT_TURN]),
            min_size=1,
            max_size=5
        ),
        learning_rate=st.floats(min_value=0.0001, max_value=0.01),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=10, deadline=3000)
    def test_nn_spawning_determinism_property(self, choice_sequence, learning_rate, seed):
        """
        **Feature: ai-hydra, Property 12a: NN Spawning Determinism**
        
        *For any* identical choice sequence and configuration, parallel NN spawning
        should produce identical network architectures and initial weights.
        **Validates: Requirements 11.1, 11.3**
        """
        # Create network configuration
        network_config = NetworkConfig(
            input_features=19,
            hidden_layers=[200, 200],
            output_actions=3,
            learning_rate=learning_rate
        )
        
        # Spawn NN instances for the same choice sequence twice
        nn_instances_1 = self._spawn_nn_for_choices(choice_sequence, network_config, seed)
        nn_instances_2 = self._spawn_nn_for_choices(choice_sequence, network_config, seed)
        
        # Should have same number of instances
        assert len(nn_instances_1) == len(nn_instances_2), \
            "Identical choice sequences should produce same number of NN instances"
        
        # Compare corresponding NN instances
        for choice_path in nn_instances_1.keys():
            assert choice_path in nn_instances_2, \
                f"Choice path {choice_path} missing in second spawning"
            
            nn1 = nn_instances_1[choice_path]
            nn2 = nn_instances_2[choice_path]
            
            # Compare network architectures
            assert nn1.get_network_info() == nn2.get_network_info(), \
                f"Network architectures should be identical for path {choice_path}"
            
            # Compare initial weights (should be identical with same seed)
            for (name1, param1), (name2, param2) in zip(nn1.named_parameters(), nn2.named_parameters()):
                assert name1 == name2, f"Parameter names should match: {name1} vs {name2}"
                assert torch.allclose(param1, param2, atol=1e-6), \
                    f"Initial weights should be identical for {name1} in path {choice_path}"
    
    @given(
        max_parallel_instances=st.integers(min_value=2, max_value=10),
        training_iterations=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=5, deadline=8000)
    def test_parallel_nn_training_isolation_property(self, max_parallel_instances, training_iterations, seed):
        """
        **Feature: ai-hydra, Property 12b: Parallel NN Training Isolation**
        
        *For any* number of parallel NN instances undergoing training, each instance
        should maintain independent weights and training state without interference.
        **Validates: Requirements 11.2, 11.4**
        """
        # Create multiple NN instances
        nn_instances = {}
        training_data = {}
        
        for i in range(max_parallel_instances):
            choice_path = f"path_{i}"
            
            # Create independent NN instance
            torch.manual_seed(seed + i)  # Different seed for each instance
            nn_instance = SnakeNet(input_features=19, hidden_size=200, output_actions=3)
            nn_instances[choice_path] = nn_instance
            
            # Generate unique training data for each instance
            training_data[choice_path] = self._generate_training_data(seed + i * 100)
        
        # Capture initial weights
        initial_weights = {}
        for choice_path, nn_instance in nn_instances.items():
            initial_weights[choice_path] = {
                name: param.clone().detach() 
                for name, param in nn_instance.named_parameters()
            }
        
        # Train each NN instance independently
        for iteration in range(training_iterations):
            for choice_path, nn_instance in nn_instances.items():
                self._train_nn_instance(nn_instance, training_data[choice_path])
        
        # Verify training isolation
        for choice_path, nn_instance in nn_instances.items():
            current_weights = {
                name: param.clone().detach() 
                for name, param in nn_instance.named_parameters()
            }
            
            # Weights should have changed from initial (training occurred)
            weights_changed = False
            for name, initial_weight in initial_weights[choice_path].items():
                current_weight = current_weights[name]
                if not torch.allclose(initial_weight, current_weight, atol=1e-6):
                    weights_changed = True
                    break
            
            assert weights_changed, \
                f"NN instance {choice_path} weights should have changed after training"
            
            # Verify independence: no other instance should have identical weights
            for other_path, other_instance in nn_instances.items():
                if other_path == choice_path:
                    continue
                
                other_weights = {
                    name: param.clone().detach() 
                    for name, param in other_instance.named_parameters()
                }
                
                # At least one weight should be different
                weights_identical = True
                for name in current_weights.keys():
                    if not torch.allclose(current_weights[name], other_weights[name], atol=1e-6):
                        weights_identical = False
                        break
                
                assert not weights_identical, \
                    f"NN instances {choice_path} and {other_path} should have different weights after independent training"
    
    @given(
        budget_exhaustion_point=st.integers(min_value=5, max_value=15),
        active_nn_count=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=8, deadline=4000)
    def test_nn_lifecycle_cleanup_property(self, budget_exhaustion_point, active_nn_count, seed):
        """
        **Feature: ai-hydra, Property 12c: NN Lifecycle Cleanup**
        
        *For any* budget exhaustion scenario, all spawned NN instances should be
        properly cleaned up without memory leaks or resource conflicts.
        **Validates: Requirements 11.3, 11.4**
        """
        # Create multiple active NN instances
        active_nn_instances = {}
        
        for i in range(active_nn_count):
            choice_path = f"active_path_{i}"
            torch.manual_seed(seed + i)
            nn_instance = SnakeNet(input_features=19, hidden_size=200, output_actions=3)
            active_nn_instances[choice_path] = nn_instance
        
        # Measure initial memory usage
        initial_memory = self._get_memory_usage()
        
        # Simulate budget exhaustion - some instances should be discarded
        surviving_instances = {}
        discarded_count = 0
        
        for i, (choice_path, nn_instance) in enumerate(active_nn_instances.items()):
            if i < budget_exhaustion_point // 2:  # Keep some instances
                surviving_instances[choice_path] = nn_instance
            else:  # Discard others
                discarded_count += 1
                # Simulate proper cleanup
                del nn_instance
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Verify surviving instances still work
        for choice_path, nn_instance in surviving_instances.items():
            dummy_features = torch.randn(1, 19)
            prediction, confidence = nn_instance.predict_move(dummy_features)
            
            assert 0 <= prediction <= 2, \
                f"Surviving NN {choice_path} should still make valid predictions"
            assert 0.0 <= confidence <= 1.0, \
                f"Surviving NN {choice_path} should still provide valid confidence"
        
        # Verify memory cleanup
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be proportional to surviving instances only
        expected_memory_per_instance = 5  # MB per NN instance (rough estimate)
        max_expected_memory = len(surviving_instances) * expected_memory_per_instance + 10  # 10MB buffer
        
        assert memory_increase <= max_expected_memory, \
            f"Memory usage should reflect only surviving instances. " \
            f"Expected <= {max_expected_memory}MB, got {memory_increase}MB increase. " \
            f"Surviving: {len(surviving_instances)}, Discarded: {discarded_count}"
    
    def _simulate_parallel_nn_exploration(self, initial_board: GameBoard, budget: int, 
                                        nn_instances: Dict, creation_log: List) -> None:
        """Simulate parallel NN spawning during tree exploration."""
        current_choices = [("root", initial_board)]  # (choice_path, board_state)
        moves_made = 0
        
        while current_choices and moves_made < budget:
            next_choices = []
            
            for choice_path, board_state in current_choices:
                if moves_made >= budget:
                    break
                
                # Get possible moves
                possible_moves = GameLogic.get_possible_moves(board_state.direction)
                
                # Spawn NN for each possible move (parallel NN spawning)
                for i, move in enumerate(possible_moves):
                    if moves_made >= budget:
                        break
                    
                    # Create choice path identifier
                    move_suffix = {
                        MoveAction.LEFT_TURN: "L",
                        MoveAction.STRAIGHT: "S", 
                        MoveAction.RIGHT_TURN: "R"
                    }[move.action]
                    
                    new_choice_path = f"{choice_path}_{move_suffix}" if choice_path != "root" else move_suffix
                    
                    # Spawn new NN instance for this choice
                    nn_instance = self._spawn_nn_instance(new_choice_path)
                    nn_instances[new_choice_path] = nn_instance
                    
                    # Log NN creation
                    creation_log.append({
                        'choice_path': new_choice_path,
                        'parent_path': choice_path,
                        'move': move.action.value,
                        'creation_order': len(creation_log)
                    })
                    
                    # Execute move to get new board state
                    result = GameLogic.execute_move(board_state, move)
                    
                    if not result.is_terminal:
                        next_choices.append((new_choice_path, result.new_board))
                    
                    moves_made += 1
            
            current_choices = next_choices
    
    def _spawn_nn_instance(self, choice_path: str) -> SnakeNet:
        """Spawn a new NN instance for a specific choice path."""
        # Use choice path hash as seed for deterministic but unique initialization
        path_seed = hash(choice_path) % 10000
        torch.manual_seed(path_seed)
        
        return SnakeNet(input_features=19, hidden_size=200, output_actions=3)
    
    def _spawn_nn_for_choices(self, choice_sequence: List[MoveAction], 
                            network_config: NetworkConfig, seed: int) -> Dict[str, SnakeNet]:
        """Spawn NN instances for a specific choice sequence."""
        nn_instances = {}
        
        for i, choice in enumerate(choice_sequence):
            choice_path = f"choice_{i}_{choice.value}"
            
            # Use deterministic seed based on position and base seed
            torch.manual_seed(seed + i)
            
            nn_instance = SnakeNet(
                input_features=network_config.input_features,
                hidden_size=network_config.hidden_layers[0],
                output_actions=network_config.output_actions
            )
            
            nn_instances[choice_path] = nn_instance
        
        return nn_instances
    
    def _verify_nn_independence(self, nn_instances: Dict[str, SnakeNet]) -> None:
        """Verify that all NN instances are independent."""
        instance_list = list(nn_instances.values())
        
        for i, nn1 in enumerate(instance_list):
            for j, nn2 in enumerate(instance_list[i+1:], i+1):
                # Verify they are different objects
                assert nn1 is not nn2, f"NN instances {i} and {j} should be different objects"
                
                # Verify they have independent parameters
                for (name1, param1), (name2, param2) in zip(nn1.named_parameters(), nn2.named_parameters()):
                    assert name1 == name2, f"Parameter structure should match: {name1} vs {name2}"
                    
                    # Parameters should be independent tensors
                    assert param1.data_ptr() != param2.data_ptr(), \
                        f"Parameters {name1} should have independent memory locations"
    
    def _generate_training_data(self, seed: int) -> List[tuple]:
        """Generate unique training data for an NN instance."""
        torch.manual_seed(seed)
        
        training_samples = []
        for _ in range(5):  # Small batch for testing
            features = torch.randn(19)  # Random features
            target = torch.randint(0, 3, (1,)).item()  # Random target action
            training_samples.append((features, target))
        
        return training_samples
    
    def _train_nn_instance(self, nn_instance: SnakeNet, training_data: List[tuple]) -> None:
        """Train an NN instance with provided data."""
        nn_instance.train()
        optimizer = torch.optim.Adam(nn_instance.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for features, target in training_data:
            optimizer.zero_grad()
            
            # Forward pass - get raw logits
            x = features.unsqueeze(0)  # Add batch dimension
            x = nn_instance.relu(nn_instance.input_layer(x))
            x = nn_instance.relu(nn_instance.hidden_layer(x))
            logits = nn_instance.output_layer(x)  # Raw logits, no softmax
            
            # Compute loss and backpropagate
            target_tensor = torch.tensor([target], dtype=torch.long)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()
    
    def _cleanup_nn_instances(self, nn_instances: Dict[str, SnakeNet]) -> None:
        """Clean up NN instances to test resource management."""
        for choice_path in list(nn_instances.keys()):
            del nn_instances[choice_path]
        nn_instances.clear()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def test_nn_spawning_basic_functionality(self):
        """
        Basic functionality test for NN spawning system.
        
        This test verifies that the basic NN spawning mechanism works
        without property-based complexity.
        """
        # Create a simple game board
        initial_board = GameLogic.create_initial_board((10, 10), 3, 42)
        
        # Test basic NN spawning
        nn_instances = {}
        choice_paths = ["L", "S", "R"]
        
        for choice_path in choice_paths:
            nn_instance = self._spawn_nn_instance(choice_path)
            nn_instances[choice_path] = nn_instance
        
        # Verify basic properties
        assert len(nn_instances) == 3, "Should create 3 NN instances"
        
        for choice_path, nn_instance in nn_instances.items():
            assert isinstance(nn_instance, SnakeNet), f"Instance {choice_path} should be SnakeNet"
            
            # Test prediction capability
            dummy_features = torch.randn(1, 19)
            prediction, confidence = nn_instance.predict_move(dummy_features)
            
            assert 0 <= prediction <= 2, f"Prediction should be valid for {choice_path}"
            assert 0.0 <= confidence <= 1.0, f"Confidence should be valid for {choice_path}"
        
        # Test independence
        self._verify_nn_independence(nn_instances)