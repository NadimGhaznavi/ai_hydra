API Reference
=============

This section provides detailed documentation for all modules and classes in the AI Hydra agent.

Core Models
-----------

.. automodule:: ai_hydra.models
   :members:
   :undoc-members:
   :show-inheritance:

Configuration System
--------------------

.. automodule:: ai_hydra.config
   :members:
   :undoc-members:
   :show-inheritance:

Router System
-------------

The Hydra Router system provides ZeroMQ-based message routing for distributed deployments.

.. automodule:: ai_hydra.router
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.mq_client
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.router_constants
   :members:
   :undoc-members:
   :show-inheritance:

Logging System
--------------

.. automodule:: ai_hydra.logging_config
   :members:
   :undoc-members:
   :show-inheritance:

Game Logic
----------

.. automodule:: ai_hydra.game_logic
   :members:
   :undoc-members:
   :show-inheritance:

Main Orchestrator
-----------------

The HydraMgr class implements the enhanced 9-state decision cycle architecture for optimal move selection.

**Decision Cycle States:**

1. **INITIALIZATION**: Reset budget and prepare components
2. **NN_PREDICTION**: Generate neural network move predictions  
3. **TREE_SETUP**: Create initial exploration clones
4. **EXPLORATION**: Execute budget-constrained tree search
5. **EVALUATION**: Analyze paths and select optimal move
6. **ORACLE_TRAINING**: Update NN from tree search results
7. **MASTER_UPDATE**: Apply move to authoritative game state
8. **CLEANUP**: Destroy exploration tree
9. **TERMINATION_CHECK**: Determine simulation continuation

.. automodule:: ai_hydra.hydra_mgr
   :members:
   :undoc-members:
   :show-inheritance:

Master Game
-----------

.. automodule:: ai_hydra.master_game
   :members:
   :undoc-members:
   :show-inheritance:

Budget Management
-----------------

.. automodule:: ai_hydra.budget_controller
   :members:
   :undoc-members:
   :show-inheritance:

State Management
----------------

.. automodule:: ai_hydra.state_manager
   :members:
   :undoc-members:
   :show-inheritance:

Exploration Clones
------------------

.. automodule:: ai_hydra.exploration_clone
   :members:
   :undoc-members:
   :show-inheritance:

Neural Network Components
-------------------------

.. automodule:: ai_hydra.neural_network
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.feature_extractor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.oracle_trainer
   :members:
   :undoc-members:
   :show-inheritance:

ZeroMQ Communication Layer
--------------------------

.. automodule:: ai_hydra.zmq_server
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.zmq_protocol
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.headless_server
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.zmq_client_example
   :members:
   :undoc-members:
   :show-inheritance:

TUI Client
----------

.. automodule:: ai_hydra.tui.client
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: ai_hydra.tui.game_board
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Pipeline
-------------------

.. automodule:: ai_hydra.simulation_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Error Handling
--------------

The Hydra Router system provides comprehensive error handling with detailed context and debugging information.

Exception Hierarchy
~~~~~~~~~~~~~~~~~~~

.. automodule:: ai_hydra.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Message Validation
~~~~~~~~~~~~~~~~~~

.. automodule:: ai_hydra.validation
   :members:
   :undoc-members:
   :show-inheritance:

Error Handler
~~~~~~~~~~~~~

.. automodule:: ai_hydra.error_handler
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
----------------------

.. automodule:: ai_hydra.cli
   :members:
   :undoc-members:
   :show-inheritance:

Token Tracking System
---------------------

The token tracking system provides comprehensive monitoring of AI token usage within the Kiro IDE environment.

Core Token Tracker
~~~~~~~~~~~~~~~~~~~

.. automodule:: ai_hydra.token_tracker.tracker
   :members:
   :undoc-members:
   :show-inheritance:

Data Models
~~~~~~~~~~~

.. automodule:: ai_hydra.token_tracker.models
   :members:
   :undoc-members:
   :show-inheritance:

CSV Writer
~~~~~~~~~~

.. automodule:: ai_hydra.token_tracker.csv_writer
   :members:
   :undoc-members:
   :show-inheritance:

Metadata Collection
~~~~~~~~~~~~~~~~~~~

.. automodule:: ai_hydra.token_tracker.metadata_collector
   :members:
   :undoc-members:
   :show-inheritance:

Error Handling
~~~~~~~~~~~~~~

.. automodule:: ai_hydra.token_tracker.error_handler
   :members:
   :undoc-members:
   :show-inheritance: