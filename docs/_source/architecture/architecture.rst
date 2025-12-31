Architecture Overview
=====================

AI Hydra implements a sophisticated hybrid system that combines neural network predictions with budget-constrained tree search for Snake Game AI decision making. The system features a distributed router-based architecture that enables multiple clients to connect to a centralized AI server through an intelligent message routing system.

Network Architecture
--------------------

The AI Hydra system uses a router-based messaging architecture that transforms the traditional direct client-server model into a scalable, distributed system:

.. code-block:: text

    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   TUI Client    │    │ Monitoring Tool │    │   API Client    │
    └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                           ┌─────────▼───────┐
                           │  HydraRouter    │
                           │   (Port 5556)   │
                           └─────────┬───────┘
                                     │
                           ┌─────────▼───────┐
                           │ Headless Server │
                           │ (AI Agent)      │
                           └─────────────────┘

**Key Architecture Benefits:**

* **Scalability**: Multiple clients can connect to a single AI server through the router
* **Distributed Deployment**: Router and server can run on different machines
* **Intelligent Routing**: Messages are routed based on sender type and client registration
* **Fault Tolerance**: Automatic client detection and cleanup with heartbeat management
* **Resource Efficiency**: Centralized AI processing with distributed client access

Router Components
~~~~~~~~~~~~~~~~~

**HydraRouter**
  Central message routing component using ZeroMQ ROUTER socket that manages client connections, routes messages between clients and servers, and provides heartbeat-based client lifecycle management.

**MQClient**
  Generic client class for router communication that supports both client and server roles, provides automatic connection management with reconnection support, handles structured message protocol with timeout management, and includes built-in message format conversion between ZMQMessage and RouterConstants formats.

**Message Format Adapter**
  Built-in component within MQClient that automatically converts between ZMQMessage format (used internally) and RouterConstants format (used for router communication), ensuring transparent operation and backward compatibility.

**RouterConstants**
  Centralized configuration for message types, network settings, and protocol constants that ensures consistent communication across all components and defines the standard message format expected by the router.

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

Router System Components
~~~~~~~~~~~~~~~~~~~~~~~~

**HydraRouter**
  Central message routing system that manages client connections using ZeroMQ ROUTER socket, implements heartbeat-based client registration and lifecycle management, provides intelligent message routing based on sender type, and supports distributed deployment across multiple machines.

**MQClient**
  Generic message queue client that provides unified interface for router communication, supports both client and server connection modes, implements automatic connection management with reconnection support, handles structured JSON message protocol with timeout management, and provides Python context manager support for resource cleanup.

**Message Protocol**
  Standardized JSON message structure with sender identification, client ID tracking, message type classification, timestamp and request ID correlation, and structured data payload for commands, responses, and broadcasts.

**Client Registration System**
  Automatic client registration via heartbeat messages sent every 5 seconds, inactive client detection and removal after 15-second timeout, background task management for heartbeat processing, and graceful client cleanup with resource management.

**CLI Integration**
  Router command (``ai-hydra-router``) with configurable address and port options, updated server command (``ai-hydra-server --router``) for router connection, updated TUI command (``ai-hydra-tui --router``) for router communication, and support for remote deployment with distributed router access.

Decision Flow Architecture
--------------------------

System Initialization
~~~~~~~~~~~~~~~~~~~~~~

The AI Hydra system follows a structured initialization process before beginning decision cycles:

1. **Configuration Loading**: Load and validate Hydra Zen configuration
2. **Component Initialization**: Initialize HydraMgr, Master Game, Budget Controller, State Manager, Neural Network, and Oracle Trainer
3. **Logging Setup**: Configure comprehensive logging system
4. **System Ready**: Prepare for decision cycle execution

Starting State Definition
~~~~~~~~~~~~~~~~~~~~~~~~~

Before each decision cycle begins, the system establishes a well-defined starting state:

**Master Game State:**
- GameBoard with current snake position, body, direction, food location, and score
- Move count tracking for circular pattern detection
- Deterministic random state for reproducible food placement

**Budget Controller State:**
- Move budget initialized to configured value (default: 100 moves)
- Budget consumption counter reset to 0
- Round tracking for budget management

**State Manager State:**
- No active exploration clones (clean slate)
- Clone ID generator reset for new tree
- Tree structure tracking cleared

**Neural Network State:**
- Model loaded with current weights
- Feature extractor ready for game state processing
- Oracle trainer prepared for prediction comparison

Decision Cycle Sequence
~~~~~~~~~~~~~~~~~~~~~~~

The complete decision-making process follows this detailed sequence:

1. **INITIALIZATION**: Reset budget, get master state, prepare components

2. **NN_PREDICTION**: Extract features, get neural network move prediction
   
   - Extract 19 features from current GameBoard state
   - Neural network predicts optimal move with confidence score
   - Log prediction for oracle comparison

3. **TREE_SETUP**: Create initial exploration clones from master state
   
   - Initialize budget controller with configured move allowance
   - Create 3 initial exploration clones from master GameBoard
   - Each clone tests the NN prediction first

4. **EXPLORATION**: Execute rounds of clone moves within budget constraints
   
   - Execute budget-constrained tree search with dynamic expansion
   - Allow current round completion even if budget exceeded
   - Create sub-clones from surviving clones (L, S, R moves)
   - Terminate clones on collision or food consumption

5. **EVALUATION**: Analyze all paths and select optimal move
   
   - Collect cumulative rewards from all completed exploration paths
   - Select path with highest reward (fewest moves for ties)
   - Extract first move from optimal path

6. **ORACLE_TRAINING**: Compare NN vs tree search, update network if needed
   
   - Compare neural network prediction with tree search result
   - Generate training sample if predictions differ
   - Update neural network weights using tree search as ground truth

7. **MASTER_UPDATE**: Apply winning move to authoritative game state
   
   - Apply selected move to master game through GameLogic
   - Confirm state update and log master move application

8. **CLEANUP**: Destroy exploration tree and prepare for next cycle
   
   - Reset exploration tree and budget for next cycle
   - Destroy entire exploration tree structure

9. **TERMINATION_CHECK**: Determine if simulation should continue
   
   - Check if game has ended (collision/max moves)
   - Continue until game termination or user stop

Budget-Constrained Tree Search
------------------------------

The tree search algorithm implements several key innovations:

**Round-Based Execution**
  - Executes all active clones in synchronized rounds
  - Allows current round to complete even if budget exceeded
  - Provides predictable computational costs
  - Tracks moves per round for logging and analysis

**Dynamic Tree Expansion**
  - Creates 3 sub-clones from each surviving clone
  - Hierarchical naming system (1, 2, 3 → 1L, 1S, 1R → 1LL, 1LS, 1LR)
  - Natural termination on collision or budget exhaustion
  - Efficient cleanup in any order after decision cycle

**Reward-Based Evaluation**
  - +10 reward for eating food (terminates clone optimally)
  - -10 penalty for collisions (wall or self)
  - 0 reward for empty square movement
  - Cumulative reward tracking across entire paths
  - Tie-breaking by fewest moves for efficiency

**Budget Management Flow**
  - Initialize budget to configured value (default: 100 moves)
  - Decrement budget by 1 for each clone move execution
  - Complete current round even if budget exhausted
  - Reset budget after each decision cycle
  - Track budget utilization for performance metrics

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