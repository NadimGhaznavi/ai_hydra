AI Hydra: Hybrid Neural Network + Tree Search System
===================================================

AI Hydra is an advanced Snake game AI that combines neural network predictions 
with tree search validation to achieve superior performance through hybrid 
decision-making and continuous learning.

Quick Links
-----------

📚 **End User Documentation**
   Complete guides for using AI Hydra, from installation to advanced features.
   
   * :doc:`end_user/getting_started` - Installation and setup
   * :doc:`end_user/quickstart` - Run your first simulation
   * :doc:`end_user/troubleshooting` - Common issues and solutions
   * :doc:`end_user/tui_getting_started` - Terminal user interface guide
   * :doc:`end_user/tui_controls` - TUI controls and navigation

🏗️ **Architecture & Code Documentation**
   Technical documentation for developers and researchers.
   
   * :doc:`architecture/architecture` - System design overview
   * :doc:`architecture/api_reference` - Complete API documentation
   * :doc:`architecture/decision_flow` - Decision-making algorithms
   * :doc:`architecture/design` - Detailed system design
   * :doc:`architecture/ai_documentation_manager` - AI Documentation Manager system architecture
   * :doc:`architecture/tui_architecture` - TUI system architecture
   * :doc:`architecture/zmq_protocol` - ZeroMQ communication protocol
   * :doc:`architecture/router_message_protocol_fix` - Router message format standardization

⚙️ **Operations Runbook**
   Procedures for project management and maintenance using Kiro IDE.
   
   * :doc:`runbook/development_standards` - Development standards and directory layout guidelines
   * :doc:`runbook/token_tracking` - Complete token usage monitoring guide
   * :doc:`runbook/version_update_procedure` - Version management and release procedures
   * :doc:`runbook/sdlc_procedures` - Software development lifecycle management
   * :doc:`runbook/deployment_maintenance` - Production deployment and maintenance
   * :doc:`runbook/deployment` - Deployment instructions
   * :doc:`runbook/testing` - Testing procedures and standards
   * :doc:`runbook/runbook` - General operational procedures
   * :doc:`runbook/requirements` - Project requirements
   * :doc:`runbook/tasks` - Implementation tasks
   * :doc:`runbook/token_tracking_implementation_status` - Token tracking status

Overview
--------

AI Hydra addresses the fundamental problem where legacy AI regularly scores below 10 by implementing a "deep thinking" architecture that explores multiple neural network learning trajectories simultaneously before making each move.

**Core Architecture**: Every decision spawns multiple concurrent neural network instances that explore different learning paths through a budget-constrained tree search. The system exhausts its move budget analyzing alternatives before making a single move in the master game, prioritizing decision quality over speed.

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
* **Token Tracking System**: Monitor AI token usage across all interactions

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

System Architecture
-------------------

The system consists of several key components:

* **HydraMgr**: Main orchestration system managing parallel NN instances and exploration
* **GameBoard**: Immutable game state representation with perfect cloning
* **GameLogic**: Pure functions for game mechanics and move execution
* **Neural Network**: PyTorch-based move prediction with concurrent spawning
* **Tree Search**: Budget-constrained exploration with parallel NN evaluation
* **Oracle Trainer**: Learning system that improves NN from tree search results
* **ZeroMQ Server**: Headless communication layer for remote control
* **TUI Client**: Terminal user interface for real-time visualization and control
* **Token Tracker**: Comprehensive AI token usage monitoring and analysis

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