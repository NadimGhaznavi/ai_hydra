# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The the project's [CHANGELOG](https://snakelab.osoyalce.com/CHANGELOG.html) for more information.

---

## [0.12.0] - 2025-10-31

### Fixed
- Set default router address to 0.0.0.0

---

## [0.12.0] - 2025-10-24

### Added
- A link to the [Alberta Plan](https://arxiv.org/pdf/2208.11173) on the Richard Sutton page.

### Changed
- Tuned the `SimServer` *INFO* log messages:
  - Print the epoch every 50 epochs.
  - Print a new highscore when achieved.
- Tuned the `SimRouter` *INFO* log messages:
  - Print the client and server counts when they change.

### Fixed
- Fixed *highscores plot*.
- Fixed `-v` switch so it doesn't require an argument.
---

## [0.11.3] - 2025-10-23

### Fixed
- Git tag issue

---

## [0.11.2] - 2025-10-22

### Fixed
- PIP packages calls a synchronous main(). Add an main_async() method that main() calls.

---

## [0.11.1] - 2025-10-22

### Fixed
- Git tag issue

---

## [0.11.0] - 2025-10-22

### Added
- Complete redesign:
  - Client/Server architecture to make the simulations fully deterministic. The Textual asyncio loop was polluting the simualtion, yielding different results with the same simulation starting parameters.
    - Split the `AISim` into `SimServer` and `SimClient`.
    - Added `SimRouter` and implemented ZeroMQ based messaging.
    - Supports running the multiple `SimClients` connected to the same `SimServer`.
    - Supports running the `SimClient`, `SimRouter` and `SimServer` on different machines.
    - Added a new `DMQ` constants file to support *ZeroMQ* messaging.
  - Took the SQLite (mostly) out of `ReplayMemory` and into a new `DbMgr` class.
    - Added `highscore_events`, `games_score` and `avg_loss` tables.
  - Updated website content:
    - Added additional *architecture* and *schema* pages to the website.
      - Added additional images to the architecture page.
    - Updated the main page.
  - Implemented targeted messaging:
    - When a `SimClient` connects to a running simulation, it receives old *highscore*, *game score* and *average loss* data so the TUI plots are up to date.
  - Command line switches:
    - To increase verbosity of console output for the `SimServer` and `SimRouter` and for the `SimClient's` log file.
    - To specify the hostname of the `SimRouter` if it's not running locally.
  - Added a *logger* based `LabLogger` module for cleaner console and log file output.
    - Added a `DLabLogger` constants file.
  - Implemented *full throttle* mode for the `SimServer`.
    - Automatically enabled when there are no `SimClients` connected to the `SimRouter`.
    - Automatically disabled when a `SimClient` connects.
    - Increases simulation speed by a factor of about three!!
  - Made the README a pointer to the project website.

### Fixed
  - All TUI elements are now properly updated from the running simulation
  - Header on all project files.
    - Added a line pointing at the project website: https://snakelab.osoyalce.com/

---

## [0.10.1] - 2025-10-19

### Fixed
- Updated `ReplayMemory:append()` to **not** store games if the *memory type* is set to *None*.
- `EpsilonAlgo` to make it compatable with `EpsilonAlgoN`. This way the two algorithms can be changed without any effect on `AiSim`.
    - Fixed `played_game()` to accept an unused `score` parameter.
    - Fixed dummy `new_highcore()` to accept a `score` paramenter.
    - Fixed `epsilon()` to accept an unused `score` parameter.

---

## [0.10.0] - 2025-10-16

### Added
- A page outlining the notation used in Sutton's RL Book.
- Added an *dynamic training* feature, where the `AIAgent:train_long_memory()` code is run an increasing number of times at the end of each epoch as the total number of epochs increases.
  - Modified the `train_long_memory()` code.
  - Added a new `DAIAgent` constants file with `MAX_DYNAMIC_TRAINING_LOOPS`
  - Added a `Dynamic Training` checkbox to the TUI settings
  - Added `dynamic_training()` and `epoch()` methods to the `AIAgent` class.
  - Added a `loops` TUI element to show the current number of loops used during the *long training* phase.
  - Added a *Dynamic Training* section to the project's [website](https://snakelab.osoyalce.com/).
- Added a *Epsilon N* feature in a new `EpsilonAlgoN` class.
  - Added a drop-down menu in the TUI to let the user choose between *Epsilon* and *Epsilon N*.
  - Added an *Epsilon N* section to the project's [website](https://snakelab.osoyalce.com/).

### Changed
- The learning rate default value for the RNN.
- Changed the default *minimum Epsilon* from 0.0 to 0.1.

### Fixed
- Passed the TUI *minimum*, *initial* and *decay* *Epsilon* values from the *settings* to the actual `EpsilonAlgo` and `EpsilonAlgoN` classes `AISim:update_settings()`.

---

## [0.9.0] - 2025-10-13

### Changed
- Added formatting for large numbers to the *stored games* and *highscores game number*.
- Added a *Tabbed Plots* widget.
  - Moved the *Game Score* plot widget into the new *Tabbed Plots* widget.
  - Added a *Highscores* plot.
  - Added a *Loss* plot.
- Based on the shiny new *Loss* plot, tuned the *learning rate* for the linear and RNN models.
- Added a *Learning Rate* input to the *configuration settings*.
  - Wired the sane default learning rate values (based on the model selection) so that the correct one is loaded when the *Defaults* button is pressed.
  - Modified the `AITrainer` to capture the loss at the end of each *train step* and return an average of these values at the end of an epoch with a new `get_epoch_loss()` function.
- Added additional constants to the `DLabels` and `Dlayouts` files to support the new features.

### Removed
- Removed the `LabPlot` widget. It has been replaced by the new `TabbedPlots` widget.

---

## [0.8.1] - 2025-10-13

### Added
- Show the number of frames for the *long training phase*, when `mem_type` is *random frames*.

### Fixed
- Increased width so *Random Frames* isn't truncated.
- Fixed SQL bug in `ReplayMemory:get_random_frames()`.

### Changed
- Changed the *highscores* layout so there's a space between the scrollbar and the seconds.
- Set the *stored games* back to zero when the *restart* button is pressed
- Clear the data from the *games* and *frames* SQLite3 tables when the *restart* button is pressed.
- Format the number of stored games (add commas) when they are large.

---

## [0.8.0] - 2025-10-13

### Changed
- Linear Model, `ModelL`, changes:
  - Formatting improvement of the highscores.
  - Increased the linear model's (`ModelL`) learning rate.
  - Added an additional hidden layer to the linear model.
  - Decreased the dropout value from 0.2 to 0.1.

---

## [0.7.0] - 2025-10-13 

### Added
- Added a *Time* column to the *Highscores*
- A screenshot of the TUI to the website.

### Changed
- Plotting Widget:
  - Renamed the plotting widget from `Db4EPlot` to `LabPlot`
  - Removed the code that averaged results since I'm using a sliding method instead.
- Added a *target model* that keeps a frozen copy of the main network. This prevents *chasing moving targets* instability issues.
- Added a *soft update* (τ=0.01) to blend weights in slowly. This smooths the learning curve.
- Replaced MSE with Huber loss. It's less sensitive to large TD errors (outliers).
- Added *gradient clipping* to prevents exploding gradients.
- Added an update frequency (target_update_freq=100) to sync target every ~100 frames
- Vectorize the data coming out of the ReplayMemory leading to a **HUGE** improvement in speed.
- Adjust gradients on the batch, not single frames for *better* learning and smoother gradients
- Return the loss for future plotting.


### Fixed
- Cleared the *Plot Widget* when the *Restart* button is pressed
- Reset the neural network model's learned weights when the *Restart* button is pressed.

---

## [0.6.0] - 2025-10-12

### Added
- Added a `DColors` constants file.
- Updated website content.
  - Added *Schema*, *Constants*, *Project Layout*, *Git Branching* and *Git Commit* technical documents.
  - Added a *Richard Sutton* page.
- ReplayMemory exposes the current training game ID:
  - Modified the `ReplayMemory` to return the stored game and frame IDs.
  - Modified the `AIAgent` to first load the training data, get the training game ID, then train.
  - Modified `AISim` to get the training game ID from teh `AIAgent` and display it in the TUI.

### Changed
- Set the *learning rate* in the `AITrainer`, based on the selected *model*.

### Refactor
- Made handling and definition of constants consistent.
- Renamed the `DDb4ePlot` constants file to `DPlot`

---

## [0.5.0] - 2025-10-11

### Added
- Added a highscores widget showing the game number and the high score.
- Reworked the state map in `GameBoard:get_state()`
- Moved the `ModelRNN` model parameters into the `DModelRNN` constants file.
- Updated the *SQLite* schema used by the `ReplayMemory` class.
  - See https://snakelab.osoyalce.com/db_schema.html
- Added drop down menu to select the `ReplayMemory` type i.e. *Random Game*, *Random Frames* or *None*.
    - Added a runtime widget to display the current memory type
- Added sections to the website / README file.
- Added a [Richard Sutton](https://snakelab.osoyalce.com//richart_sutton.html) page.

---

## [0.4.8] - 2025-10-11

### Fixed
- Fix path to `AISim.tcss`.

---

## [0.4.7] - 2025-10-11

### Changed
- Use `tempfile` module to get a temporary file location for *SQLite3* database.
- Use safer cleanup for *SQLite3* database.

---

## [0.4.6] - 2025-10-11


### Fixed
- Fix path issue with `ReplayMemory`'s use of SQLite.

---

## [0.4.6] - 2025-10-11

### Fixed
- Add main() function to `AISim.py` so the CLI entry point `ai-snake-lab` works after a `pip install ai-snake-lab`.

---

## [0.4.5] - 2025-10-11

### Fixed
- Fix import statements in `ai`/* and `game`/* files.

---

## [0.4.4] - 2025-10-11

### Fixed
- Fix import statements in `Constants` files.

---

## [0.4.3] - 2025-10-11

### Added
- Initial PyPI release.

### Fixed

### Changed

---
