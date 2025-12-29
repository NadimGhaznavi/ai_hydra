Architecture Overview
=====================

AI Hydra implements a sophisticated hybrid system that combines neural network predictions with budget-constrained tree search for Snake Game AI decision making.

System Components
-----------------

The system consists of several key components working together:

Core Game Components
~~~~~~~~~~~~~~~~~~~~

**GameBoard**
  Immutable representation of the complete Snake game state including snake position, body segments, direction, food location, score, and random state for deterministic behavior.

**GameLogic**
  Pure functions for game mechanics including move execution, collision detection, game-over conditions, and reward calculations. Operates on GameBoard instances without modifying them.

**MasterGame**
  Maintains the authoritative game state and coordinates with GameLogic for move execution while preserving immutability principles.

Orchestration Components
~~~~~~~~~~~~~~~~~~~~~~~~

**HydraMgr**
  Main orchestration system that manages the entire simulation lifecycle, coordinates decision cycles, and interfaces with all other components.

**BudgetController**
  Manages computational resources during tree search exploration with round-based budget consumption and reset capabilities.

**StateManager**
  Handles exploration clone lifecycle, hierarchy management, and cleanup operations for the dynamic tree structure.

Neural Network Components
~~~~~~~~~~~~~~~~~~~~~~~~~~

**FeatureExtractor**
  Converts GameBoard state into standardized 19-feature vectors for neural network input, including collision detection, direction flags, food location, and snake metrics.

**SnakeNet**
  PyTorch neural network with 19→200→200→3 architecture that processes game state features and outputs move probabilities for left, straight, and right actions.

**OracleTrainer**
  Compares neural network predictions with tree search optimal results, generates training data when predictions differ, and updates network weights.

Tree Search Components
~~~~~~~~~~~~~~~~~~~~~~

**ExplorationClone**
  Executes individual moves using GameLogic, maintains path history from root to current position, and reports results back to the orchestration system.

Decision Flow Architecture
--------------------------

The system follows a structured decision-making process:

1. **Neural Network Prediction Phase**
   
   - Extract 19 features from current GameBoard state
   - Neural network predicts optimal move with confidence score
   - Log prediction for oracle comparison

2. **Tree Search Exploration Phase**
   
   - Initialize budget controller with configured move allowance
   - Create 3 initial exploration clones from master GameBoard
   - Execute budget-constrained tree search with dynamic expansion
   - Allow current round completion even if budget exceeded

3. **Path Evaluation Phase**
   
   - Collect cumulative rewards from all completed exploration paths
   - Select path with highest reward (random selection for ties)
   - Extract first move from optimal path

4. **Oracle Training Phase**
   
   - Compare neural network prediction with tree search result
   - Generate training sample if predictions differ
   - Update neural network weights using tree search as ground truth

5. **Master Game Update Phase**
   
   - Apply selected move to master game through GameLogic
   - Reset exploration tree and budget for next cycle
   - Continue until game termination

Budget-Constrained Tree Search
------------------------------

The tree search algorithm implements several key innovations:

**Round-Based Execution**
  - Executes all active clones in synchronized rounds
  - Allows current round to complete even if budget exceeded
  - Provides predictable computational costs

**Dynamic Tree Expansion**
  - Creates 3 sub-clones from each surviving clone
  - Hierarchical naming system (1, 2, 3 → 1L, 1S, 1R → 1LL, 1LS, 1LR)
  - Natural termination on collision or budget exhaustion

**Reward-Based Evaluation**
  - +10 reward for eating food
  - -10 penalty for collisions (wall or self)
  - 0 reward for empty square movement
  - Cumulative reward tracking across entire paths

Hybrid Neural Network Integration
---------------------------------

The system implements a sophisticated learning approach:

**Oracle-Based Training**
  - Tree search results serve as ground truth for neural network training
  - Only generates training samples when NN prediction differs from optimal
  - Maintains high-quality training data focused on correction

**Feature Engineering**
  - 19-dimensional feature vector captures essential game state information
  - Collision detection features (6): snake and wall collisions in 3 directions
  - Direction features (4): current movement direction flags
  - Food features (2): normalized relative position to food
  - Snake features (7): binary representation of snake length

**Progressive Learning**
  - Neural network learns from tree search corrections over time
  - Gradually reduces reliance on tree search as accuracy improves
  - Maintains tree search validation for critical decisions

Configuration Management
-------------------------

The system uses Hydra Zen for structured configuration:

**Hierarchical Configuration**
  - SimulationConfig: Game parameters and tree search settings
  - NetworkConfig: Neural network architecture and training parameters
  - LoggingConfig: Comprehensive logging and monitoring settings

**Validation and Inheritance**
  - Automatic validation of all configuration parameters
  - Support for configuration inheritance and composition
  - Predefined configurations for different experimental setups

Logging and Monitoring
----------------------

Comprehensive logging system provides detailed insights:

**Structured Logging**
  - Clone step logging with identifiers and outcomes
  - Decision cycle summaries with path evaluation metrics
  - Neural network prediction and training progress
  - System events and error handling

**Performance Monitoring**
  - Budget consumption tracking and efficiency metrics
  - Tree exploration depth and breadth analysis
  - Neural network accuracy progression over time
  - Resource usage and computational performance

Deterministic Reproducibility
-----------------------------

The system ensures reliable experimental reproducibility:

**Seed Management**
  - Master seed controls all random events
  - Perfect random state preservation during cloning
  - Identical seeds produce identical game sequences

**Immutable State Design**
  - All game state modifications create new instances
  - Thread-safe operations throughout the system
  - Reliable state consistency across exploration tree

This architecture provides a robust foundation for research in hybrid AI decision-making systems, budget-constrained search algorithms, and neural network training from tree search oracles.