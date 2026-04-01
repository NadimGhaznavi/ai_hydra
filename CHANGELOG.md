# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


---

## [Unreleased]

### Added
- Made *Max Moves Penalty* setting model specific and customized the value for the *GRU* model.
- Enhanced stagnation alert cleared message; includes the new highscore.
- Support for CUDA and MPS.
- New *GRU Model* defaults.

### Fixed
- Increase the *critical stagnation threshold* every time a *critical threshold event* is raised, **including the first time.**

---

## [0.23.3] - 2026-03-29 @ 11:01 - HOTFIX - Working Man Release

### Added
- **Enhanced Snapshot Report**
  - Increased the granularity of the memory and epsilon nice tables from every 500 to every 100 episodes.
  - Added the *current gear* to the memory bucket table.
  - Added the *recent mean and median* to the mean and median table.
- Created sane baseline configurations for the *Linear*, *RNN*, and *GRU* that avoid destructive training behaviour and promote better recovery behaviour under stagnation and hard reset conditions.

### Fixed
- Modify the `ATHDataStore` to return the complete list of buckets, even if some of them are empty.
- Complete overhaul of the *ATH Memory* modules to ensure they are correct.
  - Added a LOT of asserts and data validation steps.
  - Basically improved the stability and safety of the *Replay Memory*.
- `EpsilonNice` wasn't interacting correctly with the `GameLogic` for collision detection
  - Refactored the `GameLogic:step()` function to isolate the collision detection functions from the *reward* calculations
  - Updated the `GameLogic:would_collide()` method (used by `EpsilonNice`) to use the refactored `GameLogic:step()` changes.
- **Fixed Multiple Config Propagation Bugs**
  - Ensured that the *stagnation threshold* and *critical stagnation threshold* TUI settings made their way into the `TrainMgr` module.
  - Ensured that the *Epsilon Nice* TUI settings made their way to the `EpsilonNicePolicy` module.

---

## [0.23.2] - 2026-03-28 @ 14:13 - HOTFIX

### Fixed
- When first starting, the memory buckets are not all created. This means that for the first few games, the bucket_counts may be unexpectedly short, which causes issues on the client side in the snapshot report, which expects the data to be a specific size. The simple fix is to have the `ATHDataMgr` check that the bucket count is correct before sending the metrics data.

---

## [0.23.1] - 2026-03-28 @ 11:38 - HOTFIX

### Added
- Better support for Linear models in the *Snapshot* report.
- Updated *Snapshot* report documentation example for the [PyPI](https://pypi.org/project/ai-hydra/) and [RTD](https://ai-hydra.readthedocs.io/en/latest/) documentation.

### Fixed
- Corrected syntax error that caused a TUI crash when a *Snapshot* report was run for an RNN or GRU model.

---

## [0.23.0] - 2026-03-28 @ 01:55 - Centerfold Release

### Added
- **Additional Configurable Simulation Settings**
  - **Epsilon Nice Settings**
    - Added TUI controls for setting the `NICE_P_VALUE` and `NICE_STEPS` values.
  - **ATH Replay Memory Settings**
    - Added TUI controls for setting the `DOWNSHIFT_COUNT_THRESHOLD`, `MAX_FRAMES`, `MAX_GEAR`, `MAX_HARD_RESET_EPISODES`, `MAX_STAGNANT_EPISODES`, `MAX_TRAINING_FRAMES`, `NUM_COOLDOWN_EPISODES`, and `UPSHIFT_COUNT_THRESHOLD` values.
  - **Reward Settings**
    - Added TUI constrols for setting the `FOOD_REWARD`, `COLLISION_PENALTY`, `MAX_MOVES_PENALTY`, `EMPTY_MOVE_REWARD`, `CLOSER_TO_FOOD`, `FURTHER_FROM_FOOD`, and `MAX_MOVES_MULTIPLIER` values.
  - Wired these settings through the entire stack i.e. enabled support in the `SimCfg`, `HydraMgr`, `ATHMemory`, `ATHCommon`, `ATHDataMgr`, `ATHDataStore`, `ATHGearbox`, and `EpsilonNicePolicy` classes.
- Updated [PyPI](https://pypi.org/project/ai-hydra/) and [RTD](https://ai-hydra.readthedocs.io/en/latest/) documentation.
  - Added detailed configuration information with screenshots.

### Changed
- Removed `THRESHOLD_BUCKETS`. Instead the system uses the last three bucket and limits the MAX_BUCKETS to have a **minimum** of 3.
- Reworked how the *Linear Model* runs; made it consistent with the *GRU* and *RNN*:
  - Training now happens at the end of the episode.
  - The Linear model now uses the *ATH Replay Memory*
- Enabled the *layers* TUI option for the Linear model; it denotes the number of hidden layers that the NN contains.
- Changed the *Snapshot Report* to use the TUI labels instead of the backend variable names.

### Fixed
- Set `MAX_MOVES_PENALTY` to `-10` (not to `0`)


## [0.22.0] - 2026-03-26 @ 18:14 - Tweaked Layout Release 

### Added
- Complete network configuration is now visible in the TUI. The network configuration now shows:
  - Hydra router address
  - Hydra router port
  - Hydra router heartbeat port
  - Hydra router status
  - Simulation server address
  - Simulation server ZeroMQ pub port
  - A placeholder for the simulation server status (not yet implemented feature)
- Updated screenshots in the documentation that show the new TUI layout.

### Changed
- Changed the TUI layout (again) to allow for additonal settings.
  - Move the *random seed* and *move delay* to the left side of the TUI
  - Moved the *Settings* into a `TabbedContent` widget
    - Old settings are now in the *Config* tab
    - Old, partial network config moved from the left side to a *Network* tab

### Fixed
- Updatd the [RTD](https://ai-hydra.readthedocs.io/en/latest/) documentation to reflect support for custom, distributed deployments.

---

## [0.21.0] - 2025-03-26 @ 08:23 - The Islands Release

### Added
- Support for fully distributed AI Hydra
- Added additional switches 
- Updated [PyPI](https://pypi.org/project/ai-hydra/) documentation to reflect support for custom, distributed deployments.

### Fixed
- Fixed startup args parsing bus in `HydraMgr`
- Issue where a port conflict was hidden

---

## [0.20.1] - 2025-03-25 @ 10:10 - HOTFIX

### Fixed
- Set *NICE STEPS* when the Linear model is selected

---

## [0.20.0] - 2025-03-25 @ 8:05 - The "I am GRU Release"

### Added
- Updated *Event Log* screenshot for the docs.
- **Stagnation Hander**
  - Introduced the concept of a **stagnation** event that is implemented in the `TrainMgr`.
  - A **stagnation event** is defined as occurring when no new high score has been achieved within a set number of episodes.
  - There are **normal** and **critical** stagnation alerts that are sent to the *ATH Gearbox* and *Nice Policy* object.
    - The *Nice Policy* activates the `EpsilonNiceAlgo` when a threshold is received and deactivates it when the event is cleared.
    - The *ATH Gearbox* gears down when a **normal** stagnation alert is received, unless it has recently shifted up.
    - The *ATH Gearbox* does a *radical* downshift to a sequence of 8 when a **critical** alert is received.
- A **Gated Recurrent Network - GRU** has been added.
- **Epsilon Nice** Policy and Algorithm
  - See the README for more information on this new feature.

### Changed
- Updated [PyPI](https://pypi.org/project/ai-hydra/) and [RTD](https://ai-hydra.readthedocs.io/en/latest/) documentation.

### Fixed
- Scores image link in README
- `HydraLog` bug that caused duplicate log messages after a `reset`.

---

## [0.19.1] - 2026-03-21 14:37 - HOTFIX

- **Grown-Ups make mistakes too**

### Fixed
- Release glitch.

---

## [0.19.0] - 2026-03-21 14:13 - Grown-Up Release

- **Professional Docs**
- **Pause/Resume/Reset Buttons**

### Added
- A description of the displayed `ATH_GEARBOX` setting in the *Snapshot Report*.
- Complete rewrite of the [PyPI](https://pypi.org/project/ai-hydra/) and [RTD](https://ai-hydra.readthedocs.io/en/latest/) documentation.
- **Pause/Resume** buttons
  - The *Hydra Client* now includes *Pause* and *Resume* buttons.
  - The visibility is implemented correctly, so only one button is ever visible.
  - This feature also works when *Hydra Client* connects to (un)paused, running simulation.
- **Reset** button to reset the simulation.
- Configurable *random seed*
- Sane defaults for the *RTH Replay Memory* and the *RNN Model*.

---

## [0.18.0] - 2026-03-20 09:21 - Nigel Mansell Release

- **Relentless pace**
- **Controlled aggression**
- **Occasional wild bursts**
- **No fear of the edge**

*White-knuckle,high-grip performance*

### Added
- An *event type* field to the *Hydra MQ* `EventMsg`, used by `HydraEventMQ`.
- Colored the *Move Delay*, *Current Epsilon*, RNN *Batch Size*, and RNN *Sequence Length* settings, because they change during a simulation run.
- A *Memory* widget that displays the size of (number or stored sequences) and relative health (cells are shaded) of the 20 *Adaptive Replay Memory Buckets*
  - Published new telemetry information to populate this widget.
- Two new report sections in the **Snapshot Report**:
  1. A table showing `Epoch`, `Gear`, `Seq Length`, `Batch Size`, `Mean`, and `Median` values every 500 episodes.
  2. A table showing the bucket sizes for the 20 *ATH Replay Memory* buckets every 500 epochs.
- **Adaptive Memory Horizon - Replay Memory Automatic Gearbox**
  - See the [documentation](https://pypi.org/project/ai-hydra/) for more information on this feature


### Changed
- Moved the *Turbo Mode* switch to the left side of the TUI.
- **Refactored the ATHReplayMemory**
  - A `ATHDataStore` to house the *Replay Memory* data.
  - A `ATHDataMgr` to manage the contents of the `ATHDataStore`
  - Created a `ATHGearbox` class that dynamically changes the sequence length and batch size of the training data based on the state of the `ATHDataStore`.
  - A `ATHMemory` class to orchestrate the other new *ATH* classes
  - A small `ATHCommon` module to house the `GearMeta` and `Episode` dataclasses.
  - Minor refactor of *ATH* Constants.

### Fixed
- Batch size and sequence length are now correct in the TUI.
  - For RNNs these are set on the server size by the *Adaptive Replay Memory*
  - The *Adaptive Replay Memory* changes these as the game proceeds. These changes are published on the ZeroMQ *Events* topic. The client is a subscriber to this topic. The client correctly updates the values in the TUI when they change on the server.


---

## [0.17.0] - 2026-03-18 14:41

### Added
- **HydraClient TUI Updates and Additions**
  - Removed the *Update Config* button; changes to settings are instant now.
  - Replace the *Turbo Mode* checkbox with a *Switch* widget.
  - Tweaked the TUI layout.
  - Moved the one line status widget into an *Event Log* tab.
- **ZeroMQ Events Topic**
  - Added another publishing topic. This topic is a PUB/SUB ZMQ topic that publishes events (e.g. *Epsilon depleted*, *ATHReplayMemory gear change*)
  - These events are displayed in the new *HydraClient* `EventLog` tab.
    - **HydraEventMQ** and **EventMsg** classes
      - This class has been deployed in the `EpsilonAlgo` as a simple use case.
      - The class encapsulates all of the ZeroMQ code from the application.
      - Modules can now simply `self.event.publish(EventMsg(level=DHydraLog.INFO, message=msg))`
        - Messages are published on a ZeroMQ PUB/SUB socket
        - The `HydraClient` now subscribe to the new *Events Topic* and publishes them  in the new *Event Log*
  - This opens the door for enhanced reporting and event correlation.
- **Adaptive Temporal Horizon Replay Memory**
  - This addresses the *structural limitation* in the previous `ReplayMemory`.
  - The `ATHReplayMemory` is used by temporally aware models such as the *RNN*.
  - This replay memory implements an *adaptive* scaling feature:
    - The initial `sequence_length/batch_size` is set to `4/128`.
    - When a threshold is reached, `ATHReplayMemory` *gears up* to `8/64`, `16/32`, `32/16`, and finally to a `sequence_length/batch_size` of `64/8`.

### Changed
- Split the `ReplayMemory` into `ATHReplayMemory` and `SimpleReplayMemory`.
  - `SimpleReplayMemory` is used with the *Linear* model and
  - `ATHReplayMemory` is used with the *RNN*.
- Moved *ZeroMQ* constant definitions out of `DHydra` and into a new `DHydraMQ` module.


### Known Issues
- The introduction of the `ATPReplayMemory` is a significant architectural shift.
- The *Replay Memory* and the *RNN* now need to be (re)tuned.
- The next release will include more TUI side visibility into the *Replay Memory*

---

## [v0.16.1] - 2026-03-17 23:11

### Fixed
- Fixed image references in the README.md.

---

## [v0.16.0] - 2026-03-17 16:13

### Added
- Updated documentation and screenshots.
- **More Simulation Settings in the TUI**
  - *RNN Tau* for the `RNNTrainer`.
  - *Discount/Gamma* for the `RNNTrainer` and `LinearTrainer`.
  - *Batch Size* for the `RNNTrainer` and `LinearTrainer`.
  - *Sequence Length* for the `ReplayMemory` when the *RNN* model is selected.
- **Metrics Class**
  - Renamed the old `HydraMetrics` to `HydraSnapshot` and created a new `HydraMetrics`
  - When telemetry data arrives on the client, it is loaded into this object.
  - The class is used by the new `GameScorePlot`, `LossPlot`, `ScoresDistPlot` plots, and the `HydraSnapshot` class.

### Changed
- The layout of the TUI.
- **Complete Rewrite of Plotting**
  - Replaced the monolithic `TabbedPlot` with `GameScorePlot`, `LossPlot`, and `ScoresDistPlot`.
  - The old `TabbedPlot` also managed the data used by the plots; this functionality has been moved into the `HydraMetrics` class.
  - The new architecture is cleaner, clearer, and easily extensible.

### Fixed

- Images referenced in the README.md.

## Known Issue

- **Structural ReplayMemory Bug**
  - The `ReplayMemory` that houses the training data needs a *warm-up* time to build a diverse training data set.
  - For performance reasons, the *RNN's* training data is stored in *chunks* that correspond to the RNN's *sequence_length*.
  - Previously, if an entire game's transitions did not make a complete *sequence*, then it was discarded. For a *heavy* RNN (large hidden layer size), which is slow to train, this resulted in a scenario where the replay memory never *warmed up*.
  - This will be addressed in the next release.

---

## [0.15.0] - 2026-03-14 16:23

### Added
- Added a *current epsilon* column to the *high score event* table.
- Added the *RNN sequence length* and *batch size* to the snapshot file.
- **More Simulation Settings in the TUI**
  - The model is set with a drop-down menu (RNN or Linear)
  - Settings are prepopulated with sane defaults
  - Some defaults are model-specific, so if the user changes the model, those settings are reset to the (Linear or RNN) model defaults.
  - New settings include:
    - Number of nodes in the model's hidden layers
    - The *p-value* of each model's dropout layer
    - For the *RNN* the number of *RNN layers*
- Updated the [PyPI](https://pypi.org/project/ai-hydra/) and [Read-the-Docs](https://ai-hydra.readthedocs.io/en/latest/) documentation.

### Changed
- Removed the *Look Ahead* feature: Taking a page from the Bitter Lesson....
  - Updated a ton of components to support this.
- Rebuilt the plots and added new ones!
- The default RNN hyperparameters
- With the *look ahead* feature removed, I rewrote the entire Plotting widget it now includes plots that show
  - Sliding window showing game score per episode - line plot
  - Sliding window showing average game score over several episodes - line plot
  - Score distribution histogram - bar plot
  - Overall simulation loss - line plot
  - Sliding window showing loss - line plot
  - Highscores scatterplot
- Changed the TUI layout to allow for more settings in the existing screen size
- Some work on a clean shutdown, I'm making progress (this is low on my priority list)

### Removed
- I removed the *Look Ahead* feature. It goes against what Richard Sutton calls the *Bitter Lesson*: Don't inject human ideas about how the AI **should** behave. Instead, trust the whole process and enrich the environment instead.


---

## [0.14.1] = 2026-03-12 10:29

### Fixed
- Release glitch.

---

## [0.14.0] - 2026-03-12 10:19

### Added
- Stacktrace support for HydraClientMQ exceptions. Useful when expected topic data is missing.
- A `HydraLog` to the `HydraBaseMQ` class to support structured logging.
- Improved `HydraMgr` startup console log: 
  - Added a line indicating which model (Linear or RNN) is being used.
  - Added a line from the `ReplayMemory` indicating if it's using Linear/RNN memory type.
  
- **A RNN model**:
  - Created `nnet/models/RNNModel.py`, the new RNN!
  - Created a drop-down menu in the TUI to select the model type (Linear or RNN).
  - Updated the `SimCfg` to support passing the model.
  - Created `nnet/RNNTrainer.py` to support the RNN.
  - When the *RNN* is selected from the dropdown menu the Look Ahead p-value default sets to 0
- **Learning Rate**:
  - Added the *learning rate* to the TUI.
  - Handled the learning rate in the `HydraMgr`.
  - Configured the TUI so that the default learning rate is based on the selected model.
- **New ZeroMQ Scores Topic**:
  - Moved all score telementry (current score, highscore with and without lookahead) into this topic.
  - Published on a per-step basis to capture *outlier* scores where the snake has one lucky game and the highscore jumps multiple points.
  - Updated the `HydraClientMQ` and `HydraServerMQ` to support the new topic.
- **Batched ZeroMQ Messages**
  - The simulations ran so fast, that the per-step score telemtry ZeroMQ messages flooded the client. It couldn't keep up and since ZeroMQ PUB/SUB does not guarantee delivery, telemetry data was being dropped.
  - A new ZeroMQ PUB/SUB topic created and batched.
  - Per-Episode telemetry is also batched; AI Hydra runs episodes per second...
- **Hydra Metrics**
  - An initial draft of a Hydra Metrics class is introduced in this release.
  - Added a *Snapshot* button that creates a text file that contains the summary of the simulation run.

### Changed
- Modified the `Trainer` to accept a learning rate.
- Modifed the `SimCfg` to support passing the learning rate.
- Modifed the `HydraMgr` to handle the new `MODEL_TYPE` and `LEARNING_RATE` settings.
- Set the criterion in the `Trainer` to an appropriate default based on the selected model.
- **ReplayMemory**:
  - Implemented a completely new *ReplayMemory* class.
  - Uses the (SQLite) `DBMgr` to support very large ReplayMemory without running out of RAM.
  - Support different memory types:
    - None: No replay memory
    - *Random Game*: Return an ordered list of transitions for a randomly selected game.
    - *Random Frames*: Return a list of random transitions from random games.
- Renamed `Trainer` to `LinearTrainer` to distinguish it from the `RNNTrainer`
- Updated the `ReplayMemory` to support the new `RNN`:
  - The `ReplayMemory` stores the transition data in a format that supports the `RNNTrainer` *batch size* and *sequence size* making simulations run **BLAZINGLY** fast.

### Fixed
- Removed unnecessary f-string formatting in the TUI code.
- Fixed a bug where the lookahead highscore was display in the "no lookahead" highscore tab. The "no lookahead" highscore was being ignored.
- Removed a bug that was sending per-episode telementry every step.
- Improved `HydraClient` shutdown (it still needs work).

---

## [0.13.0] - 2026-03-08

### Added
- Console widget to display messages (e.g. *Simulation started...*, *Unable to connect to simulation server...*)
- **Handshake Feature**:
  - On TUI startup *Handshake* and *Quit* buttons are now visible
  - When the user clicks the *Handshake* button either:
    - A console message appears, *"Unable to connect to simulation server..."*
    - A console message appears, *"Connected to simulation server..."*, and the *Handshake* button is replaced with a *Start* button.
    - A console message appears, *"Connected to running simulation..."*, and the *Handshake* button is replaced with a *Update Config* button
- **Read-Only vs. Read-Write Settings Feature**:
  - When the TUI first starts, the labels for the settings are visible, but the actual input fields (and turbo checkbos) are not.
  - When the simulation is stopped the input fields and the turbo checkbox are visible. They are populated with default values.
  - When the simulation is running, the settings input fields are replaced by labels that show the selected settings.
  - Currently, the user can change the *Move Delay* and change the *Turbo* setting while a simulation is running. These remain configurable during simulation runs.
- **SimCfg configuration object**
  - This object exists on the client and the server and houses the settings.
  - The settings from the TUI are put into this object and it is passed to the server when the user starts a simulation.
  - If the TUI connectes to a running simulation, then the SimCfg object is passed back to the client and the TUI is updated with the running simulation settings.

### Changed
- Refactored `HydraMq` into a base `HydraBaseMQ`, `HydraClientMQ`, and `HydraServerMQ`.
- Moved the *Turbo* button to be another setting, a checkbox, into the `Settings` widget.
- Changed *Runtime Values* to *Settings and Runtime Values*
- Updated the documentation on the PyPI and Read-The-Docs sites.

Final note: I'm encoutering a dead-lock when I try to do a clean shutdown. I have a kludge in place, but it usually requires the user to hit Control-C after clicking *Quit* in the TUI. This bug will be addressed in the next release.

---

## [0.12.1] - 2026-03-06 

### Added
- TUI setting for initial epsilon value

### Fixed
- Add missing CONFIG constant

---

## [0.12.0] - 2026-03-05 

### Added
I've setup lookahead with a default p-value of 0.25, so it's only active 25% of the time.
- Tabbed plots featuring:
  - Score histogram with all scores
  - Score histogram with scores when lookahead is enabled
  - Score histogram with scores when lookahead is disabled

---

## [0.11.0] - 2026-03-05

## Added
- Added a configuration class. The client sets the configuration, passed it to the server where its used. The client can update runtime settings during a run. 

## Changed
- Refactored, moved and renamed modules and updated dependent modules:
  - Renamed `constants/DNet.py` to `constants/DNNet.py`
  - Refactored `nnet/LinearPolicy` into `nnet/LinearPolicy` and `nnet/models/LinearModel`
  - Moved `EpsilonPolicy.py`, `HydraPolicy.py`, `LinearPolicy.py`, and `LookaheadPolicy.py` into `nnet/Policy`

## Fixed
- Bug in configuration management where Turbo (deterministic and FAST mode) couldn't properly be set before the simulation started.

---

## [0.10.0] - 2026-03-04 10:38

Complete rewrite based on the [Hydra Router](https://pypi.org/project/hydra-router/) codebase. The code is fully functional now.

## Added 
- Updated the documentation for PyPI and Read-The-Docs.
- The user can now select *Turbo* mode at the beginning of the session, before starting the simulation.

## Changed
- Moved the location of the highscore in the TUI to the highscore widget.

## Fixed
- Added pytorch `torch.manual_seed(DHydra.RANDOM_SEED)` throughout the code to make the *HydraMgr* fully deterministic.
- Score is displaying again.
- Made the simulation fully deterministic **if** the user runs it in *Turbo* mode.

---

# [Unreleased] 

### Added
- **Hydra Router Examples**: Basic client-server example demonstrating router communication patterns
  - Added `hydra-router/examples/basic_client_server.py` with comprehensive client-server communication example
  - Demonstrates server responding to client messages through router with proper async/await patterns
  - Shows ZMQMessage usage, router integration, and proper error handling
  - Includes clear setup instructions and user feedback with emoji indicators
  - Provides practical template for building router-based applications
- **Documentation Updates**: Enhanced Hydra Router documentation with new example integration
  - Updated `hydra-router/README.md` with basic example section and improved quick start guide
  - Enhanced `docs/_source/end_user/hydra_router_configuration.rst` with basic client-server example documentation
  - Added examples section to development workflow with clear usage instructions
  - Integrated new example with existing router demo examples for comprehensive coverage
  - Improved client usage examples with proper async patterns and error handling
- **Standalone Hydra Router Package**: Complete extraction of router functionality as independent PyPI package
  - Created `hydra-router/` directory with standalone package structure following PyPI standards
  - Implemented `pyproject.toml` with proper package metadata, dependencies, and `ai-hydra-router` console script entry point
  - Extracted and enhanced all router components without ai-hydra dependencies:
    - `hydra_router/zmq_protocol.py`: Standalone ZMQMessage and MessageType implementation
    - `hydra_router/router_constants.py`: RouterConstants and RouterLabels for message format definitions
    - `hydra_router/exceptions.py`: Complete custom exception hierarchy with detailed context
    - `hydra_router/validation.py`: Comprehensive message validation framework with detailed error reporting
    - `hydra_router/mq_client.py`: Generic MQClient library with automatic format conversion between ZMQMessage and RouterConstants
    - `hydra_router/router.py`: Enhanced HydraRouter with client registry, message routing, and background task management
    - `hydra_router/cli.py`: Command-line interface with argument parsing, logging setup, and user-friendly output
  - Created comprehensive package documentation:
    - `README.md`: Installation instructions, quick start guide, architecture overview, and usage examples
    - `LICENSE`: MIT license for open source distribution
    - `.gitignore`: Python project gitignore with comprehensive exclusions
  - Implemented test infrastructure:
    - `tests/conftest.py`: Shared test fixtures and configuration for pytest
    - `tests/unit/test_router_constants.py`: Unit tests for RouterConstants module
    - `tests/unit/test_zmq_protocol.py`: Unit tests for ZMQMessage and MessageType functionality
  - Created usage examples:
    - `examples/basic_client_server.py`: Complete client-server communication example with multiple clients
    - `examples/simple_heartbeat_test.py`: Heartbeat mechanism demonstration with single and multiple clients
  - Package provides `ai-hydra-router` command for standalone router deployment
  - All components are fully independent with no ai-hydra dependencies, enabling use in any Python project
  - Maintains backward compatibility with existing ai-hydra router usage patterns
- **Hydra Router Examples**: Basic client-server example demonstrating router communication patterns
  - Added `hydra-router/examples/basic_client_server.py` with comprehensive client-server communication example
  - Demonstrates server responding to client messages through router with proper async/await patterns
  - Shows ZMQMessage usage, router integration, and proper error handling
  - Includes clear setup instructions and user feedback with emoji indicators
  - Provides practical template for building router-based applications
- **Message Validation System**: Comprehensive message validation framework for RouterConstants format compliance
  - Added MessageValidator class with detailed error reporting and field validation in `hydra-router/src/hydra_router/validation.py`
  - Implemented validation utilities with strict and lenient validation modes
  - Created singleton validator pattern for performance optimization with validation caching
  - Added comprehensive field type validation and timestamp validation with reasonable bounds checking
  - Support for ZMQMessage format detection and conversion guidance
  - Provides detailed validation error information for debugging and troubleshooting
  - Integrated exception hierarchy with validation system for enhanced error context
  - Added comprehensive field validation including required fields (sender, elem, data, client_id, timestamp) and optional fields (request_id)
  - Implemented sender type validation against allowed values (HydraClient, HydraServer, HydraRouter)
  - Added message element validation against comprehensive list of supported message types
  - Created validation error context with expected format specification for comparison
- **Documentation Updates**: Enhanced documentation for message validation framework integration
  - Updated `docs/_source/architecture/api_reference.rst` with comprehensive MessageValidator documentation
  - Enhanced `docs/_source/architecture/hydra_router_system.rst` Message_Validator component description with detailed feature list
  - Expanded message validation error handling section with comprehensive validation features and error reporting details
  - Updated `docs/_source/end_user/hydra_router_configuration.rst` with message validation framework section including validation features and error debugging
  - Enhanced `docs/_source/end_user/troubleshooting.rst` MessageValidationError section with detailed debugging information, validation utilities, and comprehensive error examples
  - Added validation debugging examples and utilities documentation for troubleshooting message format issues
  - Documented validation modes (lenient vs strict) and performance optimization features
  - Included comprehensive error handling examples and validation best practices

### Changed
- **Router Integration Tests**: Comprehensive refactoring of router integration tests for enhanced validation
  - Completely rewrote router integration tests to use actual components instead of mocks
  - Added TestRouterIntegration class with router startup/shutdown, client registry, and message validation tests
  - Added TestRouterErrorHandling class for error condition testing and TestRouterPerformance class for performance validation
  - Removed complex mock-based testing in favor of direct component integration testing
  - Enhanced test coverage for HydraRouter, MQClient, and message format conversion with proper async/await patterns
  - Improved test organization and cleanup procedures for more reliable test execution
  - **✅ IMPLEMENTED**: Enhanced integration test suite with comprehensive component testing
    - `test_router_startup_and_shutdown()`: Validates router lifecycle management and component initialization
    - `test_client_registry_operations()`: Tests client registration, heartbeat updates, and removal operations
    - `test_message_validation_integration()`: Validates message format compliance and error reporting
    - `test_mq_client_format_conversion()`: Tests bidirectional message format conversion between ZMQMessage and RouterConstants
    - `test_message_type_mapping_completeness()`: Ensures all message types have proper mapping coverage
    - `test_background_task_management()`: Validates background task lifecycle and cleanup procedures
    - `test_invalid_message_format_handling()`: Tests error handling for malformed messages
    - `test_unsupported_message_type_handling()`: Validates handling of unsupported message types
    - `test_client_registration_error_handling()`: Tests error conditions in client registration
    - `test_multiple_client_registration_performance()`: Performance testing with 100 concurrent client registrations
    - `test_message_conversion_performance()`: Performance validation for 1000 message conversions
  - **Testing Architecture Improvements**: Replaced 251-line mock-based test suite with 279-line real component testing
    - Eliminated MockAsyncSocket and complex mock setup in favor of actual HydraRouter instances
    - Used port 0 binding for automatic port allocation to avoid conflicts in test environments
    - Implemented proper async/await patterns with comprehensive error handling and cleanup
    - Added performance benchmarks with specific timing requirements (< 1 second for bulk operations)
    - Enhanced test documentation with clear descriptions of validation scope and expected behavior
- **Router Startup Process**: Enhanced router startup process and error handling
  - Updated main_async to use new router.start() method instead of start_background_tasks()
  - Added comprehensive exception handling with logging for router failures
  - Updated docstring to reflect enhanced router functionality
  - Improved error reporting with stack traces for debugging
- **Architecture Documentation**: Updated Hydra Router system architecture documentation for enhanced startup process
  - Updated `docs/_source/architecture/hydra_router_system.rst` startup sequence section
  - Enhanced operational procedures to reflect new router.start() method
  - Documented improved initialization control and error reporting capabilities
  - Maintained consistency between code implementation and architecture documentation

### Added
- **Message Validation System**: Comprehensive message validation framework for RouterConstants format compliance
  - Added MessageValidator class with detailed error reporting and field validation
  - Implemented validation utilities with strict and lenient validation modes
  - Created singleton validator pattern for performance optimization
  - Integrated exception hierarchy with validation system for enhanced error context
  - Added comprehensive field type validation and timestamp validation
  - Support for ZMQMessage format detection and conversion guidance
  - Provides detailed validation error information for debugging and troubleshooting
- **Router Demo Examples**: Comprehensive router demonstration with multiple clients and server interaction
  - Added `examples/test_router_demo.py` demonstrating complete router functionality with DemoServer and DemoClient classes
  - Includes step-by-step demo scenario showing router startup, client/server connections, message routing, and heartbeat monitoring
  - Demonstrates proper async/await patterns, error handling, and graceful shutdown procedures
  - Provides practical example of HydraRouter, MQClient, ZMQMessage, and RouterConstants integration
  - Shows real-world usage patterns for router system with comprehensive logging and status updates
- **Documentation Updates**: Enhanced examples and user documentation for router demo integration
  - Updated `examples/README.md` with comprehensive router demo documentation including expected output examples
  - Enhanced `docs/_source/end_user/getting_started.rst` to feature router demo as primary example with detailed workflow explanation
  - Added router demo examples section to `docs/_source/end_user/hydra_router_configuration.rst` with usage instructions
  - Provided complete output examples showing router startup, client connections, message routing, and cleanup procedures
  - Integrated router demo documentation with existing user guides and configuration documentation
- **Exception Hierarchy**: Comprehensive router exception hierarchy with detailed context and debugging information
  - Implemented HydraRouterError base exception with context support for enhanced error debugging
  - Added MessageValidationError for message format validation failures with invalid message details
  - Created ConnectionError for network connection issues with address, port, and client ID context
  - Implemented ClientRegistrationError for client management failures with operation tracking
  - Added MessageFormatError for format conversion failures with source/target format details
  - Created RouterConfigurationError for invalid configuration with config key/value validation
  - Implemented HeartbeatError for heartbeat mechanism failures with timeout tracking
  - Added RoutingError for message routing failures with sender/target and routing rule context
  - All exceptions follow consistent error handling patterns with proper inheritance hierarchy
- **Message Validation System**: Complete message validation framework for RouterConstants format compliance
  - Implemented MessageValidator class with comprehensive RouterConstants format validation
  - Added detailed error reporting with specific field validation and type checking
  - Created validation utilities with strict and lenient validation modes
  - Integrated exception hierarchy with validation system for enhanced error context
  - Added singleton validator pattern for performance optimization
  - Implemented validation caching and comprehensive field type validation
- **Hydra Router Specification Complete**: Comprehensive specification for standalone router component extraction
  - **Requirements Document**: Created detailed requirements (`.kiro/specs/hydra-router/requirements.md`) with 8 main requirements covering centralized message routing, generic MQClient library, message format standardization, heartbeat monitoring, comprehensive validation, flexible routing rules, scalable connection management, and configuration flexibility
  - **Design Document**: Created comprehensive design (`.kiro/specs/hydra-router/design.md`) with high-level architecture, message flow diagrams, component specifications, data models, error handling framework, correctness properties, testing strategy, CLI interface, package integration, and deployment examples
  - **Implementation Tasks**: Created detailed tasks document (`.kiro/specs/hydra-router/tasks.md`) with 6-phase implementation plan covering infrastructure setup, message format implementation, MQClient library development, HydraRouter core implementation, integration testing, and documentation
  - **Task Details**: Defined 24 specific tasks with priorities, time estimates, acceptance criteria, and implementation details across phases from core infrastructure to deployment
  - **Testing Strategy**: Established comprehensive testing approach with unit tests (95% coverage), property-based tests (100+ examples), integration tests, and end-to-end workflow validation
  - **Dependencies and Risk Assessment**: Documented external dependencies (ZeroMQ, asyncio, pytest, hypothesis, sphinx), internal component dependencies, and mitigation strategies for high-risk areas
  - **Success Criteria**: Provided functional requirements, non-functional performance requirements, and deployment requirements for implementation validation
  - **Future Roadmap**: Added post-implementation tasks for AI Hydra integration and future enhancements (multi-server support, message persistence, advanced routing)
  - **Specification Status**: ✅ **COMPLETE** - Ready for implementation with all requirements, design, and tasks fully documented
- **Hydra Router Implementation**: Phase 1 and 2 implementation complete - Core infrastructure and message validation
  - **Enhanced Router Core**: Implemented comprehensive `ai_hydra/router.py` with enhanced HydraRouter class featuring modular architecture with ClientRegistry, MessageRouter, BackgroundTaskManager, and MessageValidator components
  - **Client Registry Management**: Implemented thread-safe ClientRegistry class with client registration, heartbeat tracking, automatic pruning, and client type classification supporting hundreds of concurrent connections
  - **Message Routing Logic**: Enhanced MessageRouter class with intelligent routing between clients and servers, error handling, and extensible routing rules for flexible communication patterns
  - **Background Task Management**: Added BackgroundTaskManager for client pruning, health monitoring, and resource cleanup with graceful shutdown and configurable task lifecycle
  - **Enhanced Error Handling**: Comprehensive error logging with detailed context, validation error details, and debugging hints for troubleshooting malformed messages and connection issues
  - **Backward Compatibility**: Maintained compatibility with existing router interface while adding enhanced capabilities and improved performance
  - **Unit Tests**: Created comprehensive test suite (`tests/unit/test_router_validation.py`, `tests/unit/test_router_exceptions.py`) with 95%+ coverage for validation and exception handling
  - **Integration Tests**: Implemented integration test suite (`tests/integration/test_router_integration.py`) validating component interactions, performance benchmarks, and error handling scenarios
  - **Message Format Conversion**: Enhanced MQClient with robust bidirectional conversion between ZMQMessage and RouterConstants formats with validation and error recovery
  - **Performance Optimization**: Optimized message processing, client tracking, and validation with efficient data structures and asynchronous processing
  - **Status**: ✅ **PHASES 1-2 COMPLETE** - Core infrastructure setup and message validation framework implemented and tested, ready for Phase 3 (MQClient Library Enhancement)
- **Router Demo Examples**: Comprehensive router demonstration with multiple clients and server interaction
  - Added `examples/test_router_demo.py` demonstrating complete router functionality with DemoServer and DemoClient classes
  - Includes step-by-step demo scenario showing router startup, client/server connections, message routing, and heartbeat monitoring
  - Demonstrates proper async/await patterns, error handling, and graceful shutdown procedures
  - Provides practical example of HydraRouter, MQClient, ZMQMessage, and RouterConstants integration
  - Shows real-world usage patterns for router system with comprehensive logging and status updates

### Changed
- **Documentation Updates**: Updated documentation to reflect Hydra Router specification completion
  - **Architecture Documentation**: Updated `docs/_source/architecture/hydra_router_system.rst` with specification completion status and comprehensive overview of requirements, design, and implementation components
  - **Requirements Documentation**: Added Requirement 48 to `docs/_source/runbook/requirements.rst` documenting the completion of Hydra Router specification with all acceptance criteria marked as completed
  - **Tasks Documentation**: Added Phase 17 to `docs/_source/runbook/tasks.rst` documenting the completion of Hydra Router specification tasks including requirements, design, implementation plan, and specification integration
  - **Main Documentation Index**: Updated `docs/_source/index.rst` to indicate Hydra Router specification completion status
  - **Status Indicators**: Added ✅ **COMPLETE** status indicators throughout documentation to show specification readiness for implementation
  - **Implementation Readiness**: Documented that Hydra Router specification is ready for development with all requirements defined, architecture designed, and implementation plan detailed
- **Hydra Router Implementation Tasks**: Enhanced PyPI package structure specification
  - Updated Task 1.1 to specify standalone PyPI package creation with `ai-hydra-router` executable
  - Added detailed PyPI package directory structure with modern `src/` layout
  - Included console script configuration for command-line interface
  - Enhanced acceptance criteria for pip installability and PyPI distribution
  - Added PyPI-specific documentation requirements and packaging standards
- **Hydra Router Requirements**: Clarified single server architecture and future extensibility
  - Updated introduction to specify support for multiple clients and single server architecture
  - Added clarification about zero or one server support with multiple clients
  - Noted extensibility for multiple servers in future versions
  - Improved accuracy of system architecture description in requirements document
- **Architecture Documentation**: Updated Hydra Router system architecture documentation for consistency
  - Modified system overview to reflect single server with multiple clients architecture
  - Updated key features to specify single server connection with future extensibility
  - Clarified Hydra_Router component description for accurate client/server relationship
  - Updated Client_Registry description to reflect single server connection tracking
  - Modified routing rules to reflect client-to-server forwarding for single server
- **Configuration Documentation**: Updated Hydra Router configuration guide for single server architecture
  - Modified overview to clarify multiple clients connecting to single server through router
  - Updated routing configuration section to note server-to-server as reserved for future use
  - Added comments in configuration examples to clarify future multiple server support
- **Requirements Documentation**: Updated requirements specification for single server architecture
  - Modified Requirement 38 acceptance criteria to specify zero or one server support
  - Updated Requirement 41 to include server connection status tracking alongside clients
  - Revised Requirement 43 to reflect client-to-server routing and future extensibility
  - Enhanced requirements to accurately reflect current single server limitations
- **Hydra Router Design**: Clarified design as integrated AI Hydra component
  - Updated overview to emphasize integration with existing ai-hydra PyPI package rather than standalone system
  - Clarified router as reusable component that leverages existing proven components (HydraRouter, MQClient, RouterConstants)
  - Updated key design features to reflect package integration while maintaining reusability focus
  - Referenced router-message-protocol-fix enhancements in error handling improvements
  - Maintained emphasis on generic router system usable by any project needing message routing
- **Hydra Router Implementation Tasks**: Enhanced PyPI package structure specification
  - Updated Task 1.1 to specify standalone PyPI package creation with `ai-hydra-router` executable
  - Added detailed PyPI package directory structure with modern `src/` layout
  - Included console script configuration for command-line interface
  - Enhanced acceptance criteria for pip installability and PyPI distribution
  - Added PyPI-specific documentation requirements and packaging standards

### Added
- **Hydra Router Design Document**: Comprehensive design document for ZeroMQ-based message routing system
  - Created detailed design document (`.kiro/specs/hydra-router/design.md`) with complete system architecture and implementation guidance
  - Documented high-level and message flow architecture with Mermaid diagrams showing client-router-server interactions
  - Defined core components: HydraRouter (central message router), MQClient (generic client library), RouterConstants (message format definitions)
  - Specified automatic message format conversion between internal ZMQMessage and RouterConstants formats
  - Documented comprehensive error handling and validation pipeline with detailed error logging and recovery mechanisms
  - Included deployment configuration examples and network topology diagrams
  - Added complete testing strategy covering unit, property-based, integration, and performance tests
  - Designed for future extensibility with multi-server support, custom client types, and message protocol extensions
  - Provides implementation foundation for standalone ZeroMQ-based message routing system
- **Documentation Updates**: Updated Hydra Router system architecture documentation for design consistency
  - Updated `docs/_source/architecture/hydra_router_system.rst` to align with comprehensive design document
  - Enhanced system overview to emphasize standalone component nature and message format conversion
  - Updated key design features to reflect complete independence and generic client library capabilities
  - Refined component descriptions to match detailed design specifications for HydraRouter, MQClient, and RouterConstants
  - Updated `docs/_source/architecture/api_reference.rst` to include router system modules (router, mq_client, router_constants)
  - Maintained consistency between specification documents and architecture documentation
  - Ensured documentation reflects current single server architecture with future multi-server extensibility
  - Designed for future extensibility with multi-server support, custom client types, and message protocol extensions
  - Provides implementation foundation for standalone ZeroMQ-based message routing system
- **Hydra Router System Documentation**: Comprehensive documentation for the ZeroMQ-based message routing system
  - Created detailed requirements document (`.kiro/specs/hydra-router/requirements.md`) with 10 main requirements covering centralized message routing, MQClient library, message format standardization, heartbeat monitoring, error handling, routing rules, scalability, configuration, backward compatibility, and monitoring
  - Added comprehensive glossary with 11 key terms for router system components including Hydra_Router, MQClient, RouterConstants, Message_Format_Adapter, Heartbeat_Monitor, Client_Registry, Message_Validator, ZMQMessage, RouterConstants_Format, Client_Type, and Message_Routing
  - Established acceptance criteria for each requirement with specific technical details and user stories for system architects, developers, integrators, administrators, and operators
  - Added user stories for each requirement to clarify stakeholder needs and use cases
  - Provides foundation for implementing robust ZeroMQ-based message routing system with reusable router pattern across different projects
- **Requirements Documentation Updates**: Added Requirements 38-47 covering comprehensive Hydra Router system
  - Requirement 38: Centralized Message Routing with ZeroMQ ROUTER socket support
  - Requirement 39: Generic MQClient Library with unified interface for client/server communication
  - Requirement 40: Message Format Standardization and Conversion with automatic format adaptation
  - Requirement 41: Heartbeat Monitoring and Client Tracking with automatic lifecycle management
  - Requirement 42: Comprehensive Message Validation and Error Handling with detailed error reporting
  - Requirement 43: Flexible Routing Rules and Message Broadcasting with configurable patterns
  - Requirement 44: Scalable Connection Management supporting hundreds of concurrent connections
  - Requirement 45: Configuration and Deployment Flexibility for different environments
  - Requirement 46: Backward Compatibility and Migration Support for smooth transitions
  - Requirement 47: Monitoring and Observability with comprehensive metrics and logging
- **Architecture Documentation**: Created comprehensive Hydra Router system architecture documentation
  - Added `docs/_source/architecture/hydra_router_system.rst` with complete system architecture
  - Documented core router components, message processing components, and client management components
  - Detailed message flow architecture including client registration, message routing, and format conversion flows
  - Comprehensive message format specifications for RouterConstants and ZMQMessage formats
  - Routing rules and patterns documentation with default behavior and configurable options
  - Error handling and recovery mechanisms with validation errors and connection management
  - Performance and scalability considerations with connection scalability and resource management
  - Monitoring and observability features with real-time metrics and structured logging
  - Deployment configurations for development, production, and cloud environments
  - Security considerations including network security and message security
  - Future enhancements and scalability improvements planning
- **Configuration Documentation**: Created comprehensive Hydra Router configuration guide
  - Added `docs/_source/end_user/hydra_router_configuration.rst` with complete configuration instructions
  - Router configuration with CLI options, configuration file formats, and environment variables
  - Client configuration for MQClient, server, and TUI client setup
  - Deployment configurations for local development, production, Docker, and Kubernetes
  - Security configuration with access control and encryption options
  - Monitoring configuration with metrics collection and health checks
  - Troubleshooting configuration with debug options and common issues
  - Configuration validation and best practices
- **Documentation Structure Updates**: Enhanced main documentation index with router system links
  - Updated `docs/_source/index.rst` to include Hydra Router configuration guide in end user documentation
  - Added Hydra Router system architecture to architecture documentation section
  - Integrated router documentation with existing documentation structure
  - Maintained consistent navigation and organization

### Changed
- **Hydra Router Requirements**: Clarified single server architecture and future extensibility
  - Updated introduction to specify support for multiple clients and single server architecture
  - Added clarification about zero or one server support with multiple clients
  - Noted extensibility for multiple servers in future versions
  - Improved accuracy of system architecture description in requirements document

### Added
- **Hydra Router Specifications**: Comprehensive requirements document for ZeroMQ-based message routing system
  - Created detailed requirements document with 10 main requirements covering centralized message routing, MQClient library, message format standardization, heartbeat monitoring, error handling, routing rules, scalability, configuration, backward compatibility, and monitoring
  - Defined comprehensive glossary with 10 key terms for router system components including Hydra_Router, MQClient, RouterConstants, Message_Format_Adapter, and others
  - Established acceptance criteria for each requirement with specific technical details and user stories
  - Added user stories for each requirement to clarify stakeholder needs and use cases for system architects, developers, integrators, administrators, and operators
  - Provides foundation for implementing robust ZeroMQ-based message routing system with reusable router pattern across different projects

### Changed
- **AI Documentation Manager Requirements**: Simplified and focused requirements specification for improved clarity and implementation
  - Simplified introduction to focus on documentation organization extraction from token tracker spec
  - Reduced glossary from 6 terms to 2 essential terms (Main_Documentation_Page, Project_Documentation)
  - Consolidated 7 complex requirements into 2 focused requirements covering documentation structure organization and content migration
  - Removed detailed acceptance criteria for build systems, quality assurance, steering integration, and automated maintenance
  - Streamlined acceptance criteria to be more focused and implementable
  - Enhanced clarity by removing redundant and overly complex requirements
- **Documentation Updates**: Updated requirements specification and architecture documentation to reflect AI Doc Manager changes
  - Updated `docs/_source/runbook/requirements.rst` with consolidated Requirements 36-37 covering AI Documentation Manager system
  - Removed Requirements 38-42 (Build System, Quality Assurance, Steering Integration, Automated Maintenance, Integration/Extensibility) to focus on core functionality
  - Renumbered requirements for consistency after consolidation
  - Updated `docs/_source/architecture/ai_documentation_manager.rst` with simplified system architecture focusing on structure organization and content migration
  - Removed complex automated systems (Content Organizer, Build System, Quality Assurance Engine) in favor of focused migration approach
  - Updated system responsibilities to emphasize documentation structure organization and safe content migration
  - Modified architecture to reflect project ecosystem organization with three-tier structure per project

### Added
- **AI Documentation Manager Specification**: Comprehensive requirements document for automated documentation management system
  - Created detailed requirements document with 7 main requirements covering documentation structure, content organization, and system integration
  - Defined acceptance criteria for build system, quality assurance, and automated maintenance workflows
  - Established integration requirements with existing development tools and version control systems
  - Added comprehensive glossary of key terms and components for documentation management system
  - Provides foundation for implementing automated documentation organization and maintenance workflows
- **Documentation Updates**: AI Documentation Manager requirements integration and architecture documentation
  - Updated `docs/_source/runbook/requirements.rst` with Requirements 36-37 covering focused documentation management system
  - Added Requirements 36-37 covering Documentation Structure Organization and Content Migration
  - Created `docs/_source/architecture/ai_documentation_manager.rst` with simplified system architecture documentation
  - Documented focused approach with Documentation Structure Organization, Main Documentation Page, and Content Migration System components
  - Added three-tier documentation structure (End User, Developer, Operations) for project ecosystem organization
  - Included content management workflow, integration points, configuration management, and automated maintenance systems
  - Established performance and scalability considerations with plugin architecture for future extensibility
  - Integrated AI Doc Manager requirements with existing project requirements (Requirements 36-44)
  - Enhanced requirements traceability linking AI Doc Manager specification to project documentation standards
- **Router Enhancement**: Enhanced router error logging and debugging capabilities
  - Added comprehensive message format validation with detailed error reporting
  - Implemented enhanced malformed message logging with expected vs actual format comparison
  - Added detailed frame error logging for ZMQ communication issues
  - Implemented JSON parsing error logging with debugging hints
  - Added unknown sender type error logging with valid options display
  - Provided comprehensive debugging information for all error types
  - Enhanced router troubleshooting capabilities for message format issues
- **Documentation**: Router message protocol fix documentation
  - Created comprehensive router message protocol fix design document (docs/_source/architecture/router_message_protocol_fix.rst)
  - Added detailed problem statement, solution overview, and architecture documentation
  - Documented message format mapping between ZMQMessage and RouterConstants formats
  - Included format conversion process, validation, error handling, and testing strategy
  - Added migration and deployment guidance with backward compatibility considerations
- **Documentation Updates**: Comprehensive documentation status tracking and implementation progress
  - Updated architecture documentation with implementation status indicators
  - Added "Current Implementation Status" section to router message protocol fix documentation
  - Updated heartbeat message format section with "IN PROGRESS" status tracking
  - Enhanced MQClient integration documentation with current implementation details
  - Added implementation notes showing actual RouterConstants format usage
  - Updated architecture.rst with MQClient and Message Format Adapter status indicators
  - Enhanced ZMQ protocol documentation with implementation status and completion checkmarks
  - Updated specification documents with comprehensive implementation status tracking
  - Added status indicators to requirements documentation (Requirements 31-35)
  - Implemented consistent status tracking across all documentation files
  - Added progress visibility with ✅ **IMPLEMENTED**, 🔄 **IN PROGRESS**, and ⏳ **PENDING** indicators
  - Enhanced requirement tracking with completion status for all acceptance criteria
  - Updated CHANGELOG.md with router message protocol fix implementation status
  - Provided clear implementation progress visibility across all documentation

### Changed
- **AI Documentation Manager Requirements**: Refined and consolidated requirements specification for improved clarity and focus
  - Clarified user story language for better specificity and actionable outcomes
  - Consolidated requirements by removing redundant Documentation Runbook System requirement
  - Removed Multi-Audience Support requirement to focus on core functionality
  - Renumbered requirements 2-7 for consistency after consolidation
  - Updated acceptance criteria to be more focused and implementable
  - Simplified integration requirements to essential functionality
  - Enhanced requirement 1 to explicitly include ai_hydra folder artifact integration
- **Documentation**: Updated ZeroMQ protocol documentation for message format standardization
  - Enhanced message structure documentation to include both RouterConstants and ZMQMessage formats
  - Added message format conversion section with automatic conversion details
  - Updated heartbeat message documentation to specify RouterConstants format requirement
  - Added Message Format Adapter section with conversion process and error handling details
- **Documentation**: Updated requirements specification with router message protocol fix requirements
  - Added Requirements 31-35 covering message format standardization, heartbeat protocol compliance
  - Defined bidirectional message format conversion and error handling requirements
  - Established message format migration and backward compatibility requirements
- **Documentation**: Updated architecture documentation with message format adapter information
- **Specifications**: Updated router message protocol fix task status to reflect active implementation progress
  - Marked task 2 (heartbeat message format fix) as in-progress in router-message-protocol-fix spec
  - Indicates ongoing work on MQClient _send_heartbeat() method RouterConstants format implementation
  - Enhanced MQClient description to include built-in message format conversion capabilities
  - Added Message Format Adapter component description with transparent operation details
  - Updated RouterConstants description to clarify router message format expectations
  - **Implementation Status**: Task 2 heartbeat message format fix is currently being implemented in MQClient
- **Specifications**: Added comprehensive requirements documentation for router message protocol fix
  - Created detailed requirements document with 5 main requirements covering message format standardization
  - Defined acceptance criteria for heartbeat protocol compliance and bidirectional message conversion
  - Established error handling, validation, and backward compatibility requirements
  - Added glossary of key terms and components for router-MQClient communication protocol
- **Specifications**: Updated task 3.1 status in router message protocol fix specification
  - Changed task 3.1 from completed to in-progress status to reflect current implementation state
  - Updated format validation property test status tracking for accurate project management
  - Task 3.1 (Property test for format validation) status updated to reflect ongoing implementation work
  - Property 2: RouterConstants Format Compliance validation requires additional implementation

## [Release 0.8.0] - 2025-12-31 17:08


### Added
- **Project Organization**: Comprehensive directory structure reorganization and standardization
  - Created standardized directory structure with `scripts/`, `tools/`, and organized `tests/` directories
  - Established `tools/` directory with categorized development utilities:
    - `debug/` - Debugging utilities (debug_file_patterns.py)
    - `testing/` - Testing utilities (run_tests.py)
    - `documentation/` - Documentation tools (test_documentation.py, validate_docs.py)
    - `analysis/` - Analysis utilities (ready for future use)
  - Reorganized test suite by type for better maintainability:
    - `unit/` - Unit tests (6 existing + 6 moved files)
    - `property/` - Property-based tests (10 moved files)
    - `integration/` - Integration tests (4 existing + 5 moved files)
    - `e2e/` - End-to-end tests (ready for future use)
  - Created `scripts/` directory for build and maintenance scripts (update_version.sh)
  - Added comprehensive file organization utility script (`scripts/organize_files.sh`)
  - Updated file permissions for all executable scripts and tools
  - Moved documentation files to appropriate locations (DOCUMENTATION_UPDATE_SUMMARY.md → docs/)
  - Removed empty/unused files (debug_property_test.py)
- **Directory Layout Standards**: Comprehensive project organization standards and guidelines
  - Created directory-layout-standards.md in .kiro/steering/ with complete project structure definitions
  - Established standard directory structure for ai_hydra/ main package and ai_snake_lab/ legacy code
  - Defined testing organization with unit/, property/, integration/, and e2e/ test categories
  - Documented file naming conventions for Python files, tests, documentation, and configuration
  - Provided directory creation guidelines and migration procedures for reorganization
  - Included automation tools for structure validation and import path checking
- **Development Standards Documentation**: Comprehensive development standards and guidelines documentation
  - Created docs/_source/runbook/development_standards.rst with complete development standards
  - Integrated directory layout standards with existing SDLC procedures documentation
  - Documented code organization standards, import organization, and module structure guidelines
  - Provided configuration management standards and file migration procedures
  - Included automation and tooling documentation for structure validation and maintenance
  - Updated main documentation index to include development standards in operations runbook
- **Token Tracker System**: Complete implementation of comprehensive token usage tracking system
  - Core data models (TokenTransaction, TrackerConfig) with validation and CSV serialization
  - Thread-safe CSV operations with file locking and concurrent access protection
  - Token tracker service with validation, error handling, and transaction history
  - Special character and Unicode handling for international text support
  - Comprehensive property-based testing with universal correctness properties
  - Metadata collection system for workspace and execution context gathering
  - Agent hook integration with automatic token tracking on AI interactions
  - Configuration management with runtime updates and state consistency
  - Debug utilities for file patterns preservation testing
  - **Maintenance Utilities**: Comprehensive maintenance system for token tracking
    - MaintenanceManager class for CSV file rotation and cleanup operations
    - Automatic file rotation based on configurable size and row count limits (50MB or 100k rows)
    - Data compression support for archived files with gzip compression
    - Cleanup functionality for old data based on retention policies
    - Archive statistics and maintenance recommendations system
    - Thread-safe operations with proper error handling and recovery
    - Comprehensive maintenance API with rotation, cleanup, and monitoring capabilities
  - **System Monitoring**: Complete monitoring and health checking system
    - SystemMonitor class for performance metrics collection and health checks
    - Real-time monitoring with configurable intervals and alert callbacks
    - Performance metrics tracking including memory usage, CPU utilization, and transaction rates
    - Health check utilities with comprehensive system validation
    - Export capabilities for monitoring data in multiple formats
  - **Documentation Restructuring**: Complete documentation system reorganization
    - Restructured docs/_source/ with architecture/, end_user/, and runbook/ directories
    - Comprehensive token tracking documentation with usage guides and troubleshooting
    - Updated main documentation index with clear navigation structure
    - Property-based testing documentation with requirements traceability
- **Integration Testing**: Complete integration test suite for token tracking system
  - Comprehensive system workflow tests covering end-to-end token tracking operations
  - Token tracker integration tests with hook validation and metadata accuracy
  - Error scenario testing and recovery mechanism validation
  - Hook integration testing with real Kiro IDE event simulation

### Changed
- **Documentation Dependencies**: Added myst-parser>=2.0.0 for Markdown support in Sphinx documentation
- **Project Structure**: Complete reorganization following directory layout standards
  - Moved development tools from root directory to categorized `tools/` subdirectories
  - Reorganized 21 test files by type (unit, property-based, integration) for better test management
  - Relocated build and maintenance scripts to dedicated `scripts/` directory
  - Enhanced development workflow with standardized tool locations and clear categorization
  - Improved discoverability and maintainability through consistent directory structure
  - Updated all file permissions for executable scripts and development tools
- **Documentation**: Comprehensive updates for token tracking system
  - Updated token tracking documentation with configuration management features
  - Enhanced testing documentation with new property-based tests
  - Updated requirements documentation with extended agent hook capabilities
  - Added API reference documentation for TokenTrackingHook class
  - Complete comprehensive token tracking documentation runbook with usage examples, troubleshooting guides, FAQ, and diagnostic tools
  - Added deployment and maintenance procedures documentation
  - Created SDLC procedures documentation with operational guidelines
  - **Documentation Restructuring**: Complete reorganization of documentation system
    - Restructured docs/_source/ directory with architecture/, end_user/, and runbook/ categories
    - Updated main index.rst with clear navigation and project overview
    - Migrated existing documentation to appropriate categories without content loss
    - Enhanced documentation build process and cross-reference validation
- **Project Management**: Updated token tracker implementation task status
  - Tasks 1-14 (Core implementation through final integration) marked as completed
  - Task 15 (Final system validation) currently in progress
  - Comprehensive documentation implementation finished
  - All major system components implemented and tested
  - Final validation phase initiated for production readiness assessment

### Fixed
- **Token Tracker**: Enhanced error recovery and resilience mechanisms
  - Graceful handling of file system errors and permission issues
  - Robust CSV integrity validation and corruption prevention
  - Improved metadata collection with fallback for missing context

## [Release 0.7.1] - 2025-12-30 23:37


### Changed
- **Project Structure**: Complete cleanup of root directory removing legacy files
  - Removed `setup.py` (legacy packaging file, redundant with `pyproject.toml`)
  - Removed `qasync.sh`, `manual_test_server.py`, `test.log`, `accounting.txt`
  - Moved test files from root to `tests/` directory for better organization
- **Documentation**: Updated version update procedure to remove setup.py references
  - Updated `docs/_source/version_update_procedure.rst` to reflect current packaging approach
  - Removed all legacy setup file handling from documentation
- **Version Scripts**: Enhanced `update_version.sh` to remove setup.py handling
  - Added deprecation warnings if setup.py is found
  - Focused script on modern `pyproject.toml` packaging approach
- **Development Environment**: Enhanced `.gitignore` with comprehensive exclusions
  - Added coverage reports, testing artifacts, build artifacts
  - Added IDE files and OS generated files exclusions

## [Release 0.7.0] - 2025-12-30 23:04


### Added

#### Router Architecture System
- **Complete Router Implementation**: Comprehensive client/server router system based on ai_snake_lab SimRouter pattern
  - `HydraRouter` class (`ai_hydra/router.py`) with ZeroMQ ROUTER socket for message routing
  - `MQClient` generic class (`ai_hydra/mq_client.py`) for router communication with both client and server support
  - `RouterConstants` (`ai_hydra/router_constants.py`) for centralized message types and network configuration
  - Support for multiple clients connecting to single server through centralized router
  - Heartbeat-based client registration and automatic management (5-second intervals)
  - Automatic inactive client detection and removal (15-second timeout)
  - `ai-hydra-router` CLI entry point for standalone router operation with configurable address/port
  - Network architecture: `[TUI Client] ←→ [Router:5556] ←→ [Headless Server]`

#### Router Features
- **Client Registration**: Automatic client registration via heartbeat messages
- **Message Routing**: Intelligent routing based on sender type (HydraClient/HydraServer)
- **Heartbeat Management**: Background task for inactive client detection and removal
- **Error Handling**: Graceful error handling with informative error messages
- **Scalability**: Support for multiple clients per server with distributed deployment
- **Background Tasks**: Proper async task management with cleanup

#### MQClient Features
- **Connection Management**: Automatic connection and heartbeat with reconnection support
- **Message Types**: Commands, responses, and broadcasts with structured protocol
- **Timeout Handling**: Configurable operation timeouts with fallback behavior
- **Context Management**: Python context manager support for resource cleanup
- **Error Recovery**: Graceful error handling and automatic cleanup

#### Message Protocol
- **Standardized Format**: JSON message structure with sender, client_id, message_type, timestamp, request_id, data
- **Control Commands**: start_simulation, stop_simulation, pause_simulation, resume_simulation, reset_simulation
- **Status Messages**: get_status, status_update, game_state_update, performance_update
- **System Messages**: heartbeat, error, ok acknowledgments

#### TUI Epoch Display Feature
- **Epoch Counter Display**: Added "Epoch: N" display in TUI status widget between Snake Length and Runtime
- **Reactive Updates**: Real-time epoch updates using Textual's reactive variable system
- **Status Processing**: Enhanced status update processing to extract epoch from game_state data
- **Reset Integration**: Epoch resets to 0 when simulation is reset
- **Demo Enhancement**: Updated `demo_tui.py` with epoch progression (increments every 20 moves)
- **Production Integration**: Works with production `ai-hydra-tui` command via `pip install ai-hydra[tui]`

#### Comprehensive Test Suite
- **Unit Tests**: Individual component functionality testing
  - `tests/unit/test_mq_client.py`: MQClient functionality validation
  - `tests/unit/test_router.py`: HydraRouter functionality validation
  - `tests/unit/test_router_constants.py`: Constants and configuration validation
  - `tests/unit/test_tui_status_display.py`: TUI status display functionality
- **Property-Based Tests**: Universal behavior validation with Hypothesis
  - `tests/property/test_router_properties.py`: Router behavior properties
  - `tests/property/test_tui_epoch_display.py`: Epoch display properties (100+ test cases)
- **Integration Tests**: Component interaction validation
  - `tests/integration/test_router_integration.py`: Router component integration
  - `tests/integration/test_tui_epoch_integration.py`: End-to-end TUI epoch workflow
- **End-to-End Tests**: Complete workflow validation
  - `tests/e2e/test_router_system.py`: Complete router system workflows

#### CLI Commands
- **Router Command**: `ai-hydra-router --address 0.0.0.0 --port 5556 --log-level INFO`
- **Updated Server Command**: `ai-hydra-server --router tcp://localhost:5556` (connects to router)
- **Updated TUI Command**: `ai-hydra-tui --router tcp://localhost:5556` (connects to router)
- **Remote Deployment Support**: Full support for distributed deployment across multiple machines

#### Enhanced Error Handling
- **TUI Dependencies**: Better error messages for missing TUI dependencies
  - Graceful fallback when `textual` package not installed
  - Helpful installation instructions for optional dependencies (`pip install ai-hydra[tui]`)
- **Router Error Handling**: Connection failures, malformed messages, resource cleanup
- **Client Error Handling**: Connection loss recovery, timeout handling, message validation

### Changed

#### Network Architecture Transformation
- **Headless Server**: Updated `ai_hydra/headless_server.py` to use MQClient and connect to router instead of direct binding
- **TUI Client**: Updated `ai_hydra/tui/client.py` to use MQClient for router communication instead of direct server connection
- **Connection Model**: Transformed from direct client-server to router-based messaging system
  - Clients now connect to router at port 5556 instead of directly to server
  - Router handles intelligent message routing between clients and servers
  - Supports distributed deployment across multiple machines with remote router access

#### Status Display Enhancement
- **Status Widget Layout**: Added epoch display between Snake Length and Runtime in TUI status panel
- **Data Flow**: Enhanced `process_status_update()` to extract and process epoch information
- **UI Architecture**: Integrated epoch display with Textual's reactive variable system

#### CLI Integration
- **Entry Points**: Added `ai-hydra-router` to `pyproject.toml` CLI entry points
- **Command Arguments**: Enhanced server and TUI commands with `--router` parameter for router address
- **Deployment Flexibility**: Support for local and remote router deployments

### Fixed

#### Dependency Management
- **TUI Dependencies**: Fixed ModuleNotFoundError when textual package not installed
- **Import Handling**: Added graceful error handling for optional TUI dependencies
- **Installation Instructions**: Clear guidance for installing TUI support via `pip install ai-hydra[tui]`

#### Communication Protocol
- **Message Protocol**: Standardized message format across all components with proper JSON structure
- **Connection Management**: Improved connection handling and cleanup with proper resource management
- **Error Propagation**: Enhanced error handling and informative error messages throughout the system

#### Testing and Quality
- **Test Coverage**: 13/13 router constants tests passing with comprehensive edge case coverage
- **Property Testing**: Universal behavior validation with Hypothesis for robust edge case discovery
- **Integration Validation**: Complete component interaction testing with mock-based isolation
- **Requirements Compliance**: Full validation of Requirements 3.5 and 3.6 for epoch display feature

### Technical Details

#### Performance and Scalability
- **Message Throughput**: Efficient message routing with minimal latency
- **Resource Management**: Automatic cleanup of inactive clients and proper memory management
- **Concurrent Clients**: Support for multiple concurrent clients with heartbeat-based tracking
- **Async Operations**: Full async/await support for non-blocking operations

#### Security and Reliability
- **Network Security**: Configurable bind addresses and controlled error information disclosure
- **Input Validation**: Message validation to prevent malformed messages and resource exhaustion
- **Connection Security**: Framework for future authentication and encryption features
- **Monitoring**: Comprehensive logging for security monitoring and debugging

#### Code Quality Standards
- **Type Hints**: Comprehensive type annotations throughout all new components
- **Documentation**: Complete docstrings with Google style formatting
- **Error Handling**: Robust error handling with graceful degradation
- **Testing Standards**: Property-based testing with requirements traceability
- **Async Patterns**: Proper async programming patterns with resource cleanup

---

## [0.1.0] - Initial Release

### Added
- Initial AI Hydra implementation
- Neural network + tree search hybrid system
- Game board and logic implementation
- Configuration management system
- Basic TUI client interface
- Headless server implementation
- Comprehensive testing framework
- Documentation and code style standards

---
