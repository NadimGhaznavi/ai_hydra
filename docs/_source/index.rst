AI Hydra Documentation
=======================

Welcome to the AI Hydra documentation. This system implements an AI agent that uses a **parallel neural network exploration system** to play the classic video Snake Game. AI that achieves mastery over basic collision avoidance through exhaustive exploration and increases the chance of finding food.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   architecture
   decision_flow
   zmq_protocol
   deployment
   testing
   version_update_procedure
   troubleshooting
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Terminal User Interface:

   tui_getting_started
   tui_architecture
   tui_controls

.. toctree::
   :maxdepth: 2
   :caption: Project Specifications:

   requirements
   design
   tasks

.. toctree::
   :maxdepth: 2
   :caption: TUI Client Specifications:

   tui_requirements
   tui_design
   tui_tasks

Overview
--------

AI Hydra addresses the fundamental problem where legacy AI regularly scores below 10 by implementing a "deep thinking" architecture that explores multiple NN learning trajectories simultaneously before making each move.

**Core Architecture**: Every decision spawns multiple concurrent NN instances that explore different learning paths through a budget-constrained tree search. The system exhausts its move budget analyzing alternatives before making a single move in the master game, prioritizing decision quality over speed.

Key Features
------------

* **Parallel Neural Network Exploration**: Spawn multiple NN instances for each decision
* **Budget-Constrained Tree Search**: Efficient exploration within computational limits  
* **Collision Avoidance Mastery**: Achieve consistent scores > 10 through deep thinking
* **Deterministic Reproducibility**: Seed-controlled randomness for reliable experiments
* **ZeroMQ Headless Operation**: Complete message-based control without GUI dependencies
* **Terminal User Interface**: Real-time visualization and control via Textual TUI
* **Hydra Zen Configuration**: Flexible parameter management for concurrent systems
* **Comprehensive Testing**: Property-based tests with timeout protection

Quick Start
-----------

.. code-block:: python

   from ai_hydra import HydraMgr
   from ai_hydra.config import SimulationConfig, NetworkConfig
   
   # Create configuration for parallel NN exploration
   sim_config = SimulationConfig(
       grid_size=(10, 10),
       move_budget=100,  # Budget for parallel exploration
       nn_enabled=True,
       random_seed=42
   )
   
   net_config = NetworkConfig(
       input_features=19,
       hidden_layers=(200, 200),
       output_actions=3,
       learning_rate=0.001
   )
   
   # Initialize the parallel NN system
   hydra_mgr = HydraMgr(sim_config, net_config)
   
   # Run simulation with concurrent NN exploration
   result = hydra_mgr.run_simulation()
   print(f"Final score: {result.final_score}")

Headless Operation
------------------

.. code-block:: python

   from ai_hydra.headless_server import HeadlessServer
   
   # Start headless AI agent
   server = HeadlessServer(port=5555)
   server.start()
   
   # Control via ZeroMQ messages from any client
   # See zmq_client_example.py for client implementation

Architecture
------------

The system consists of several key components:

* **HydraMgr**: Main orchestration system managing parallel NN instances and exploration
* **GameBoard**: Immutable game state representation with perfect cloning
* **GameLogic**: Pure functions for game mechanics and move execution
* **Neural Network**: PyTorch-based move prediction with concurrent spawning
* **Tree Search**: Budget-constrained exploration with parallel NN evaluation
* **Oracle Trainer**: Learning system that improves NN from tree search results
* **ZeroMQ Server**: Headless communication layer for remote control
* **TUI Client**: Terminal user interface for real-time visualization and control

Performance Expectations
-------------------------

* **Decision Quality over Speed**: System prioritizes thorough analysis over fast moves
* **Collision Avoidance**: Consistent scores > 10 through exhaustive exploration
* **Resource Intensive**: Expects slow execution due to parallel NN exploration
* **Scalable Budget**: Computational cost scales with move budget allocation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`