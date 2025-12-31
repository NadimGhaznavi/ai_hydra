Decision Flow Architecture
=========================

This document provides detailed information about the AI Hydra decision flow architecture, including the 9-state decision cycle, neural network integration, and budget management system.

Overview
--------

The AI Hydra decision flow implements a sophisticated hybrid system that combines neural network predictions with budget-constrained tree search. The system operates through a structured 9-state decision cycle that ensures optimal move selection while maintaining computational efficiency.

**Key Features:**

* **9-State Decision Cycle**: Structured progression through initialization, prediction, exploration, evaluation, training, and cleanup
* **Neural Network Integration**: NN predictions guide initial exploration and learn from tree search results
* **Budget-Constrained Search**: Predictable computational costs with round-based execution
* **Oracle Training**: Continuous learning from tree search optimal results
* **Deterministic Reproducibility**: Consistent results across identical configurations

System Initialization
---------------------

Before beginning decision cycles, the system follows a structured initialization process:

Initialization Sequence
~~~~~~~~~~~~~~~~~~~~~~~

1. **Configuration Loading**
   
   * Load and validate Hydra Zen configuration object
   * Verify all required parameters and value ranges
   * Handle configuration errors with detailed validation messages

2. **Component Initialization**
   
   * **HydraMgr**: Main orchestrator for decision cycle management
   * **Master Game**: Authoritative game state container
   * **Budget Controller**: Resource management with round-based tracking
   * **State Manager**: Clone lifecycle and hierarchy management
   * **Neural Network**: Feature extraction and move prediction
   * **Oracle Trainer**: Learning system for NN improvement
   * **Logging System**: Comprehensive monitoring and debugging

3. **System Ready State**
   
   * All components initialized and cross-validated
   * Logging system active for decision cycle monitoring
   * Ready to begin first decision cycle

Starting State Definition
~~~~~~~~~~~~~~~~~~~~~~~~~

Each decision cycle begins with a well-defined starting state:

**Master Game State:**

* Current GameBoard with snake position, body, direction, food location, score
* Move count tracking for pattern detection and game progression
* Deterministic random state for reproducible food placement

**Budget Controller State:**

* Move budget initialized to configured value (default: 100 moves)
* Budget consumption counter reset to 0
* Round tracking system prepared for new cycle

**State Manager State:**

* No active exploration clones (clean slate for new tree)
* Clone ID generator reset for hierarchical naming
* Tree structure tracking cleared from previous cycle

**Neural Network State:**

* Model loaded with current trained weights
* Feature extractor ready for 19-dimensional vector processing
* Oracle trainer prepared for prediction comparison and learning

Decision Cycle States
---------------------

The decision cycle progresses through 9 distinct states, each with specific responsibilities and validation requirements:

State 1: INITIALIZATION
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Prepare all components for new decision cycle

**Operations:**

* Reset budget controller to configured value (100 moves)
* Retrieve current GameBoard state from Master Game
* Clear any residual state from previous cycle
* Initialize performance tracking metrics

**Validation:**

* Budget controller shows full budget available
* Master Game provides valid, non-terminal GameBoard
* All components report ready status

**Logging**: Decision cycle start, initial game state, budget initialization

State 2: NN_PREDICTION
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Generate neural network move prediction

**Operations:**

* Extract 19-dimensional feature vector from current GameBoard
* Process features through neural network (19→200→200→3 architecture)
* Generate move probabilities for [Left, Straight, Right] actions
* Select highest probability move as NN prediction

**Feature Vector Components:**

* **Collision Features (6)**: Snake and wall collision detection in 3 directions
* **Direction Features (4)**: Current movement direction one-hot encoding
* **Food Features (2)**: Normalized relative position to food
* **Snake Features (7)**: Binary representation of snake length (up to 127)

**Validation:**

* Feature vector has exactly 19 dimensions
* All feature values within expected ranges
* NN prediction is valid move (0, 1, or 2)

**Logging**: NN prediction, confidence score, feature vector summary

State 3: TREE_SETUP
~~~~~~~~~~~~~~~~~~~

**Purpose**: Initialize exploration tree structure

**Operations:**

* Create 3 initial exploration clones from master GameBoard
* Each clone assigned to test one move direction (Left, Straight, Right)
* All clones initially execute the NN-predicted move
* Initialize clone hierarchy with proper naming (1, 2, 3)

**Clone Configuration:**

* **Clone 1**: Tests NN prediction, then Left turn for second move
* **Clone 2**: Tests NN prediction, then Straight for second move  
* **Clone 3**: Tests NN prediction, then Right turn for second move

**Validation:**

* Exactly 3 initial clones created successfully
* All clones have identical starting GameBoard state
* Clone IDs properly assigned and tracked

**Logging**: Clone creation, initial clone states, tree initialization

State 4: EXPLORATION
~~~~~~~~~~~~~~~~~~~~

**Purpose**: Execute budget-constrained tree search

**Operations:**

* Execute rounds of clone moves within budget constraints
* For each active clone in current round:
  
  * Execute clone's assigned move via GameLogic
  * Decrement budget by 1 move
  * Log clone step with outcome and reward
  * Handle termination or create sub-clones

* Continue until budget exhausted OR all clones terminated
* Allow current round completion even if budget exceeded

**Round-Based Execution:**

* **Round 1**: Execute initial clones (1, 2, 3) with NN prediction
* **Round 2**: Execute sub-clones from survivors (1L, 1S, 1R, 2L, 2S, 2R, etc.)
* **Round N**: Continue until natural termination or budget exhaustion

**Clone Outcomes:**

* **Collision (-10 reward)**: Terminate clone, record final path
* **Food (+10 reward)**: Terminate clone optimally, record path
* **Empty (0 reward)**: Continue if budget allows, create 3 sub-clones

**Validation:**

* Budget decrements correctly for each move
* Clone states remain consistent
* Terminated clones properly recorded

**Logging**: Each clone step, budget consumption, round progression

State 5: EVALUATION
~~~~~~~~~~~~~~~~~~~

**Purpose**: Analyze exploration paths and select optimal move

**Operations:**

* Collect all completed exploration paths with cumulative rewards
* Calculate total reward for each path from root to termination
* Identify path(s) with highest cumulative reward
* Break ties by selecting path with fewest moves (efficiency)
* Extract first move from optimal path as decision

**Path Evaluation Criteria:**

1. **Primary**: Highest cumulative reward
2. **Tie-breaker**: Fewest moves (most efficient)
3. **Final tie-breaker**: Deterministic selection (e.g., leftmost)

**Validation:**

* At least one completed path exists
* All paths have valid cumulative rewards
* Optimal path selection is deterministic

**Logging**: Path evaluation results, optimal path selection, tie-breaking

State 6: ORACLE_TRAINING
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Compare NN prediction with tree search result and update network

**Operations:**

* Compare NN prediction with optimal move from tree search
* If predictions differ:
  
  * Generate training sample (features, optimal_move)
  * Update neural network weights using backpropagation
  * Record training sample for accuracy tracking

* If predictions match:
  
  * Record successful prediction for accuracy metrics
  * No network update required

**Training Process:**

* **Loss Function**: CrossEntropyLoss for move classification
* **Optimizer**: Adam with learning rate 0.001
* **Batch Size**: Single sample (online learning)
* **Update Frequency**: Only when NN prediction is incorrect

**Validation:**

* Training only occurs when predictions differ
* Network weights update successfully
* Accuracy metrics properly tracked

**Logging**: Oracle comparison, training sample generation, accuracy updates

State 7: MASTER_UPDATE
~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Apply optimal move to authoritative game state

**Operations:**

* Apply optimal move to master GameBoard via GameLogic
* Receive new GameBoard state with updated snake position, score, etc.
* Validate new game state for consistency
* Update master game with new authoritative state

**Game State Updates:**

* **Snake Position**: New head position, updated body
* **Score**: Incremented if food consumed
* **Food Position**: New food placement if consumed
* **Game Status**: Check for terminal conditions

**Validation:**

* New game state is valid and consistent
* Score updates correctly reflect game events
* Terminal conditions properly detected

**Logging**: Master move application, new game state, score changes

State 8: CLEANUP
~~~~~~~~~~~~~~~~

**Purpose**: Destroy exploration tree and prepare for next cycle

**Operations:**

* Destroy entire exploration tree structure
* Release memory from all exploration clones
* Reset clone ID generator for next cycle
* Clear tree structure tracking data

**Cleanup Process:**

* **Efficient Destruction**: Clones destroyed in any order for performance
* **Memory Management**: Explicit cleanup to prevent memory leaks
* **State Reset**: All exploration state cleared for fresh start

**Validation:**

* All exploration clones properly destroyed
* Memory usage returns to baseline
* State manager ready for next cycle

**Logging**: Tree destruction, memory cleanup, state reset

State 9: TERMINATION_CHECK
~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Determine if simulation should continue

**Operations:**

* Check master game for terminal conditions
* Evaluate collision status (wall or self-collision)
* Check for maximum move limits or other stop conditions
* Decide whether to continue or end simulation

**Terminal Conditions:**

* **Wall Collision**: Snake hits grid boundary
* **Self Collision**: Snake hits its own body
* **Maximum Moves**: Configured move limit reached
* **User Stop**: External stop command received

**Validation:**

* Terminal condition detection is accurate
* Game state properly reflects terminal status
* Simulation control logic functions correctly

**Logging**: Terminal condition check, simulation status, final results

Budget Management System
------------------------

The budget management system ensures predictable computational costs while allowing optimal exploration:

Budget Lifecycle
~~~~~~~~~~~~~~~~

1. **Initialization**: Set budget to configured value (default: 100 moves)
2. **Consumption**: Decrement by 1 for each clone move execution
3. **Round Completion**: Allow current round to finish even if budget exceeded
4. **Evaluation**: Proceed to path evaluation when budget exhausted
5. **Reset**: Restore full budget for next decision cycle

Budget States
~~~~~~~~~~~~~

**Active Budget (Budget > 0)**

* Continue exploration with new clone creation
* Execute all active clones in current round
* Create sub-clones from surviving clones

**Exhausted Budget (Budget ≤ 0)**

* Complete current round of active clones
* No new clone creation allowed
* Proceed to path evaluation after round completion

**Reset Budget**

* Restore to original configured value
* Ready for next decision cycle
* Reset round tracking counters

Round-Based Execution
~~~~~~~~~~~~~~~~~~~~~

The system uses round-based execution to ensure fair exploration:

**Round Structure:**

* All active clones execute moves simultaneously in each round
* Budget decrements by number of active clones per round
* Surviving clones generate sub-clones for next round
* Terminated clones contribute paths to evaluation

**Round Completion:**

* Current round always completes even if budget exceeded
* Ensures all clones in round get equal exploration opportunity
* Prevents bias toward earlier clones in execution order

Neural Network Integration
-------------------------

The neural network provides intelligent guidance for tree search exploration:

Feature Extraction
~~~~~~~~~~~~~~~~~

The system extracts 19 standardized features from each GameBoard state:

**Collision Detection Features (6 total):**

* Snake collision in straight direction (relative to current direction)
* Snake collision in left turn direction
* Snake collision in right turn direction
* Wall collision in straight direction
* Wall collision in left turn direction
* Wall collision in right turn direction

**Direction Features (4 total):**

* Current direction is Left (boolean)
* Current direction is Right (boolean)
* Current direction is Up (boolean)
* Current direction is Down (boolean)

**Food Location Features (2 total):**

* Normalized X distance to food (-1.0 to 1.0)
* Normalized Y distance to food (-1.0 to 1.0)

**Snake Length Features (7 total):**

* Binary representation of snake length (supports up to 127 segments)
* Bit 0: Length & 1
* Bit 1: Length & 2
* ...
* Bit 6: Length & 64

Network Architecture
~~~~~~~~~~~~~~~~~~~

**Input Layer**: 19 features → 200 neurons (ReLU activation)
**Hidden Layer**: 200 neurons → 200 neurons (ReLU activation)  
**Output Layer**: 200 neurons → 3 actions (Softmax activation)

**Output Interpretation:**

* Index 0: Left turn probability
* Index 1: Straight probability
* Index 2: Right turn probability

Oracle Training System
~~~~~~~~~~~~~~~~~~~~~

The oracle trainer implements continuous learning from tree search results:

**Training Trigger**: NN prediction differs from tree search optimal move

**Training Process**:

1. Generate training sample (feature_vector, optimal_move)
2. Compute loss using CrossEntropyLoss
3. Perform backpropagation to update weights
4. Track accuracy metrics for monitoring

**Learning Benefits**:

* Reduces reliance on expensive tree search over time
* Improves decision speed while maintaining quality
* Adapts to specific game patterns and strategies

Performance Monitoring
---------------------

The decision flow architecture includes comprehensive performance monitoring:

Timing Metrics
~~~~~~~~~~~~~

* **Decision Cycle Duration**: Total time for complete 9-state cycle
* **State Durations**: Individual timing for each of the 9 states
* **Exploration Efficiency**: Paths evaluated per unit time
* **Budget Utilization**: Percentage of budget consumed per cycle

Quality Metrics
~~~~~~~~~~~~~~

* **Neural Network Accuracy**: Percentage of correct NN predictions
* **Tree Search Depth**: Average and maximum exploration depth
* **Path Quality**: Average reward of selected optimal paths
* **Collision Avoidance**: Success rate avoiding terminal conditions

Resource Metrics
~~~~~~~~~~~~~~~

* **Memory Usage**: Peak memory during tree exploration
* **CPU Utilization**: Processing load during decision cycles
* **Clone Count**: Number of active clones per round
* **Budget Efficiency**: Moves per decision quality improvement

Error Handling and Recovery
---------------------------

The decision flow architecture includes robust error handling:

State-Level Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each state includes specific error handling and recovery mechanisms:

* **INITIALIZATION**: Configuration validation, component readiness checks
* **NN_PREDICTION**: Feature extraction validation, network error handling
* **TREE_SETUP**: Clone creation validation, memory allocation checks
* **EXPLORATION**: Move execution validation, budget consistency checks
* **EVALUATION**: Path validation, tie-breaking consistency
* **ORACLE_TRAINING**: Training sample validation, network update verification
* **MASTER_UPDATE**: Game state validation, consistency checks
* **CLEANUP**: Memory cleanup verification, state reset validation
* **TERMINATION_CHECK**: Terminal condition validation, simulation control

Recovery Strategies
~~~~~~~~~~~~~~~~~~

* **Retry with Backoff**: Temporary failures retry with exponential backoff
* **Graceful Degradation**: Reduce tree size or budget on resource constraints
* **State Rollback**: Return to previous valid state on corruption detection
* **Fallback Decisions**: Use simple heuristics if complex systems fail

This comprehensive decision flow architecture ensures reliable, efficient, and high-quality decision making for the AI Hydra Snake game agent.