"""
State management for exploration clones.

This module provides the StateManager class that handles the lifecycle of
exploration clones, including creation, hierarchical naming, and cleanup
operations for the tree search system.
"""

from typing import List, Dict, Optional
import logging
from .models import GameBoard, MoveAction
from .exploration_clone import ExplorationClone
from .game_logic import GameLogic
from .logging_config import SimulationLogger
from .config import LoggingConfig


class StateManager:
    """
    Manages exploration clone lifecycle and relationships.
    
    The StateManager handles the creation of initial exploration clones,
    sub-clone generation with hierarchical naming, and efficient cleanup
    of the exploration tree structure.
    """
    
    def __init__(self, logging_config: Optional[LoggingConfig] = None):
        """Initialize the state manager."""
        self.active_clones: Dict[str, ExplorationClone] = {}
        self.clone_counter = 0
        
        # Initialize comprehensive logging
        self.logger = SimulationLogger("state_manager", logging_config)
        
        # Tree structure tracking
        self.tree_metrics = {
            'total_clones_created': 0,
            'max_depth_reached': 0,
            'active_clones': 0,
            'terminated_clones': 0,
            'root_clones': 0,
            'tree_generations': 0,  # Number of complete tree resets
            'clone_survival_rate': 0.0,
            'average_clone_depth': 0.0
        }
        
        # Clone lifecycle tracking
        self.clone_lifecycle_history = []
        self.tree_generation_history = []
        
        self.logger.log_system_event("StateManager initialized", {
            "tracking_enabled": True,
            "metrics_initialized": True
        })
    
    def create_initial_clones(self, master_board: GameBoard) -> List[ExplorationClone]:
        """
        Create exactly 3 initial exploration clones from the master GameBoard.
        
        This method creates the root-level clones with IDs "1", "2", "3" that
        will test left turn, straight, and right turn moves respectively.
        
        Args:
            master_board: The master game board to clone from
            
        Returns:
            List[ExplorationClone]: List of 3 initial exploration clones
        """
        try:
            # Clear any existing clones
            self.active_clones.clear()
            self.clone_counter = 0
            
            # Create exactly 3 initial clones
            initial_clones = []
            for i in range(3):
                clone_id = str(i + 1)  # "1", "2", "3"
                clone = ExplorationClone(
                    initial_board=master_board,
                    clone_id=clone_id,
                    parent_id=None  # Root clones have no parent
                )
                
                self.active_clones[clone_id] = clone
                initial_clones.append(clone)
                self.clone_counter += 1
                
                # Track clone creation
                self._track_clone_creation(clone_id, None, 0)
                
                self.logger.log_system_event(f"Created initial clone {clone_id}", {
                    "parent_id": None,
                    "depth": 0,
                    "total_clones": self.clone_counter
                })
            
            # Update tree metrics
            self._update_tree_metrics()
            
            self.logger.log_system_event("Initial clones created", {
                "clones_created": len(initial_clones),
                "total_active": len(self.active_clones),
                "tree_generation": self.tree_metrics['tree_generations']
            })
            
            return initial_clones
            
        except Exception as e:
            self.logger.log_error("StateManager", f"Failed to create initial clones: {e}")
            # Return empty list on failure to allow system to handle gracefully
            return []
    
    def create_sub_clones(self, parent_clone: ExplorationClone) -> List[ExplorationClone]:
        """
        Create 3 sub-clones from a parent clone with hierarchical naming.
        
        This method creates sub-clones with IDs following the pattern:
        - Parent "1" creates "1L", "1S", "1R"
        - Parent "1L" creates "1LL", "1LS", "1LR"
        - And so on...
        
        Args:
            parent_clone: The parent clone to create sub-clones from
            
        Returns:
            List[ExplorationClone]: List of 3 sub-clones
        """
        try:
            if parent_clone.is_terminated():
                self.logger.log_warning("StateManager", f"Cannot create sub-clones from terminated clone {parent_clone.get_clone_id()}")
                return []
            
            parent_id = parent_clone.get_clone_id()
            parent_board = parent_clone.create_sub_clone_board()
            parent_depth = parent_clone.get_depth()
            
            # Create sub-clone IDs with hierarchical naming
            sub_clone_suffixes = ["L", "S", "R"]  # Left, Straight, Right
            sub_clones = []
            
            for suffix in sub_clone_suffixes:
                sub_clone_id = parent_id + suffix
                
                # Check for ID conflicts (shouldn't happen with proper usage)
                if sub_clone_id in self.active_clones:
                    self.logger.log_error("StateManager", f"Clone ID conflict: {sub_clone_id} already exists")
                    continue
                
                sub_clone = ExplorationClone(
                    initial_board=parent_board,
                    clone_id=sub_clone_id,
                    parent_id=parent_id
                )
                
                self.active_clones[sub_clone_id] = sub_clone
                sub_clones.append(sub_clone)
                self.clone_counter += 1
                
                # Track clone creation
                self._track_clone_creation(sub_clone_id, parent_id, parent_depth + 1)
                
                self.logger.log_system_event(f"Created sub-clone {sub_clone_id}", {
                    "parent_id": parent_id,
                    "depth": parent_depth + 1,
                    "total_clones": self.clone_counter
                })
            
            # Update tree metrics
            self._update_tree_metrics()
            
            self.logger.log_system_event("Sub-clones created", {
                "parent_clone": parent_id,
                "sub_clones_created": len(sub_clones),
                "new_depth": parent_depth + 1,
                "total_active": len(self.active_clones)
            })
            
            return sub_clones
            
        except Exception as e:
            self.logger.log_error("StateManager", f"Failed to create sub-clones from {parent_clone.get_clone_id()}: {e}")
            # Return empty list on failure to allow system to handle gracefully
            return []
    
    def destroy_exploration_tree(self) -> None:
        """
        Destroy the entire exploration tree efficiently.
        
        This method clears all active clones in any efficient order,
        as the specific order doesn't matter for cleanup purposes.
        """
        clone_count = len(self.active_clones)
        
        if clone_count > 0:
            # Capture tree statistics before destruction
            final_stats = self.get_tree_statistics()
            
            # Track tree generation completion
            self._track_tree_generation_completion(final_stats)
            
            self.logger.log_system_event("Destroying exploration tree", {
                "total_clones": clone_count,
                "max_depth_reached": final_stats["max_depth"],
                "terminated_clones": final_stats["terminated_clones"],
                "tree_generation": self.tree_metrics['tree_generations']
            })
            
            # Log tree exploration summary
            self.logger.log_tree_metrics({
                "total_clones_created": self.tree_metrics['total_clones_created'],
                "max_depth": final_stats["max_depth"],
                "survival_rate": f"{final_stats.get('survival_rate', 0):.1f}%",
                "avg_depth": f"{final_stats.get('average_depth', 0):.1f}",
                "tree_valid": final_stats["tree_valid"]
            })
            
            # Clear all clones (order doesn't matter for cleanup)
            self.active_clones.clear()
            self.clone_counter = 0
            
            # Update metrics
            self.tree_metrics['tree_generations'] += 1
            self._update_tree_metrics()
            
            self.logger.log_system_event("Exploration tree destroyed", {
                "clones_destroyed": clone_count,
                "tree_generation_completed": self.tree_metrics['tree_generations']
            })
        else:
            self.logger.log_system_event("No exploration tree to destroy")
    
    def generate_clone_id(self, parent_id: str, move_type: str) -> str:
        """
        Generate a clone ID based on parent ID and move type.
        
        This method creates hierarchical clone IDs following the pattern:
        parent_id + move_suffix (e.g., "1" + "L" = "1L")
        
        Args:
            parent_id: The parent clone's ID
            move_type: The move type ("L", "S", "R")
            
        Returns:
            str: Generated clone ID
        """
        if move_type not in ["L", "S", "R"]:
            raise ValueError(f"Invalid move type: {move_type}. Must be 'L', 'S', or 'R'")
        
        return parent_id + move_type
    
    def get_active_clones(self) -> List[ExplorationClone]:
        """
        Get all currently active exploration clones.
        
        Returns:
            List[ExplorationClone]: List of all active clones
        """
        return list(self.active_clones.values())
    
    def get_clone_by_id(self, clone_id: str) -> Optional[ExplorationClone]:
        """
        Get a specific clone by its ID.
        
        Args:
            clone_id: The ID of the clone to retrieve
            
        Returns:
            Optional[ExplorationClone]: The clone if found, None otherwise
        """
        return self.active_clones.get(clone_id)
    
    def get_active_clone_count(self) -> int:
        """
        Get the number of currently active clones.
        
        Returns:
            int: Number of active clones
        """
        return len(self.active_clones)
    
    def get_total_clones_created(self) -> int:
        """
        Get the total number of clones created since last reset.
        
        Returns:
            int: Total clones created
        """
        return self.clone_counter
    
    def get_clones_by_depth(self, depth: int) -> List[ExplorationClone]:
        """
        Get all clones at a specific depth in the tree.
        
        Args:
            depth: The depth level to retrieve clones from
            
        Returns:
            List[ExplorationClone]: List of clones at the specified depth
        """
        return [clone for clone in self.active_clones.values() 
                if clone.get_depth() == depth]
    
    def get_root_clones(self) -> List[ExplorationClone]:
        """
        Get all root-level clones (those with no parent).
        
        Returns:
            List[ExplorationClone]: List of root clones
        """
        return [clone for clone in self.active_clones.values() 
                if clone.get_parent_id() is None]
    
    def get_children_of_clone(self, parent_id: str) -> List[ExplorationClone]:
        """
        Get all direct children of a specific clone.
        
        Args:
            parent_id: The parent clone's ID
            
        Returns:
            List[ExplorationClone]: List of child clones
        """
        return [clone for clone in self.active_clones.values() 
                if clone.get_parent_id() == parent_id]
    
    def get_terminated_clones(self) -> List[ExplorationClone]:
        """
        Get all terminated clones.
        
        Returns:
            List[ExplorationClone]: List of terminated clones
        """
        return [clone for clone in self.active_clones.values() 
                if clone.is_terminated()]
    
    def get_active_non_terminated_clones(self) -> List[ExplorationClone]:
        """
        Get all active (non-terminated) clones.
        
        Returns:
            List[ExplorationClone]: List of active clones
        """
        return [clone for clone in self.active_clones.values() 
                if not clone.is_terminated()]
    
    def remove_clone(self, clone_id: str) -> bool:
        """
        Remove a specific clone from the active clones.
        
        Args:
            clone_id: The ID of the clone to remove
            
        Returns:
            bool: True if clone was removed, False if not found
        """
        if clone_id in self.active_clones:
            del self.active_clones[clone_id]
            self.logger.debug(f"Removed clone {clone_id}")
            return True
        else:
            self.logger.warning(f"Attempted to remove non-existent clone {clone_id}")
            return False
    
    def validate_tree_structure(self) -> bool:
        """
        Validate the integrity of the exploration tree structure.
        
        This method checks for common issues like orphaned clones,
        circular references, or invalid ID patterns.
        
        Returns:
            bool: True if tree structure is valid
        """
        issues = []
        
        # Check for orphaned clones (non-root clones with missing parents)
        for clone in self.active_clones.values():
            parent_id = clone.get_parent_id()
            if parent_id is not None and parent_id not in self.active_clones:
                issues.append(f"Clone {clone.get_clone_id()} has missing parent {parent_id}")
        
        # Check for invalid ID patterns
        for clone_id in self.active_clones.keys():
            if not self._is_valid_clone_id(clone_id):
                issues.append(f"Invalid clone ID pattern: {clone_id}")
        
        if issues:
            for issue in issues:
                self.logger.error(f"Tree structure issue: {issue}")
            return False
        
        return True
    
    def _is_valid_clone_id(self, clone_id: str) -> bool:
        """
        Check if a clone ID follows the valid hierarchical pattern.
        
        Valid patterns:
        - Root clones: "1", "2", "3"
        - Sub-clones: "1L", "2S", "3R", "1LL", "1LS", etc.
        
        Args:
            clone_id: The clone ID to validate
            
        Returns:
            bool: True if ID is valid
        """
        if not clone_id:
            return False
        
        # Root clones should be single digits
        if clone_id.isdigit():
            return clone_id in ["1", "2", "3"]
        
        # Sub-clones should start with a digit and contain only L, S, R
        if not clone_id[0].isdigit():
            return False
        
        # Check that all characters after the first digit are L, S, or R
        for char in clone_id[1:]:
            if char not in ["L", "S", "R"]:
                return False
        
        return True
    
    def get_tree_statistics(self) -> dict:
        """
        Get comprehensive statistics about the exploration tree.
        
        Returns:
            dict: Dictionary containing tree statistics
        """
        active_clones = self.get_active_clones()
        terminated_clones = self.get_terminated_clones()
        root_clones = self.get_root_clones()
        
        # Calculate depth statistics
        depths = [clone.get_depth() for clone in active_clones]
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        # Calculate reward statistics
        rewards = [clone.get_cumulative_reward() for clone in active_clones]
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards) if rewards else 0
        
        # Calculate survival rate
        total_clones = len(active_clones) + len(terminated_clones)
        survival_rate = (len(active_clones) / total_clones * 100) if total_clones > 0 else 0
        
        return {
            "total_clones_created": self.tree_metrics['total_clones_created'],
            "active_clones": len(active_clones),
            "terminated_clones": len(terminated_clones),
            "root_clones": len(root_clones),
            "max_depth": max_depth,
            "average_depth": avg_depth,
            "total_cumulative_reward": total_reward,
            "average_reward": avg_reward,
            "survival_rate": survival_rate,
            "tree_valid": self.validate_tree_structure(),
            "tree_generations": self.tree_metrics['tree_generations']
        }
    
    def log_tree_exploration_summary(self) -> None:
        """
        Log a comprehensive tree exploration summary.
        """
        stats = self.get_tree_statistics()
        
        self.logger.log_system_event("Tree exploration summary", {
            "total_clones": stats["total_clones_created"],
            "active": stats["active_clones"],
            "terminated": stats["terminated_clones"],
            "max_depth": stats["max_depth"],
            "avg_depth": f"{stats['average_depth']:.1f}",
            "survival_rate": f"{stats['survival_rate']:.1f}%",
            "total_reward": stats["total_cumulative_reward"],
            "avg_reward": f"{stats['average_reward']:.1f}",
            "tree_generations": stats["tree_generations"]
        })
        
        # Log detailed tree metrics
        self.logger.log_tree_metrics({
            "clones_created": stats["total_clones_created"],
            "max_depth": stats["max_depth"],
            "survival_rate": stats["survival_rate"],
            "avg_depth": stats["average_depth"],
            "tree_valid": stats["tree_valid"]
        })
    
    def _track_clone_creation(self, clone_id: str, parent_id: Optional[str], depth: int) -> None:
        """Track clone creation for lifecycle analysis."""
        self.clone_lifecycle_history.append({
            'clone_id': clone_id,
            'parent_id': parent_id,
            'depth': depth,
            'created_at': len(self.clone_lifecycle_history),
            'is_root': parent_id is None,
            'generation': self.tree_metrics['tree_generations']
        })
        
        # Update total clones created
        self.tree_metrics['total_clones_created'] += 1
    
    def _track_tree_generation_completion(self, final_stats: dict) -> None:
        """Track completion of a tree generation."""
        generation_summary = {
            'generation': self.tree_metrics['tree_generations'],
            'total_clones': final_stats['total_clones_created'],
            'max_depth': final_stats['max_depth'],
            'survival_rate': final_stats['survival_rate'],
            'total_reward': final_stats['total_cumulative_reward'],
            'tree_valid': final_stats['tree_valid']
        }
        
        self.tree_generation_history.append(generation_summary)
    
    def _update_tree_metrics(self) -> None:
        """Update tree metrics based on current state."""
        active_clones = self.get_active_clones()
        terminated_clones = self.get_terminated_clones()
        
        # Calculate current metrics
        depths = [clone.get_depth() for clone in active_clones]
        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0
        
        total_clones = len(active_clones) + len(terminated_clones)
        survival_rate = (len(active_clones) / total_clones * 100) if total_clones > 0 else 0
        
        # Update metrics
        self.tree_metrics.update({
            'max_depth_reached': max(self.tree_metrics['max_depth_reached'], max_depth),
            'active_clones': len(active_clones),
            'terminated_clones': len(terminated_clones),
            'root_clones': len(self.get_root_clones()),
            'clone_survival_rate': survival_rate,
            'average_clone_depth': avg_depth
        })
    
    def get_tree_exploration_efficiency(self) -> dict:
        """
        Get tree exploration efficiency metrics.
        
        Returns:
            dict: Efficiency metrics for tree exploration
        """
        if not self.tree_generation_history:
            return {
                'generations_completed': 0,
                'avg_clones_per_generation': 0,
                'avg_max_depth': 0,
                'avg_survival_rate': 0,
                'exploration_consistency': 0
            }
        
        generations = len(self.tree_generation_history)
        avg_clones = sum(g['total_clones'] for g in self.tree_generation_history) / generations
        avg_depth = sum(g['max_depth'] for g in self.tree_generation_history) / generations
        avg_survival = sum(g['survival_rate'] for g in self.tree_generation_history) / generations
        
        # Calculate exploration consistency (lower variance = more consistent)
        depths = [g['max_depth'] for g in self.tree_generation_history]
        depth_variance = sum((d - avg_depth) ** 2 for d in depths) / generations if generations > 1 else 0
        consistency = max(0, 100 - depth_variance)  # Higher is more consistent
        
        return {
            'generations_completed': generations,
            'avg_clones_per_generation': avg_clones,
            'avg_max_depth': avg_depth,
            'avg_survival_rate': avg_survival,
            'exploration_consistency': consistency,
            'total_clones_across_generations': sum(g['total_clones'] for g in self.tree_generation_history)
        }