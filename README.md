![HydraClient TUI](https://aihydra.osoyalce.com/images/hydra-client.png)

# AI Hydra - Reinforcement Learning Platform

## Overview

AI Hydra is a distributed application that includes a Textual TUI client, a 
simple router, and a headless server. Communication is over ZeroMQ. The basic
functionality is that the Client sends control messages (e.g., handshake, start)
to the router, which forwards them to the server. When the server is running,
it publishes board and other telemetry information on a ZeroMQ PUB/SUB socket.
The client subscribes to the server's PUB socket and displays the game state.

The server splits the simulation telemetry into two ZeroMQ PUB topics. One 
topic provides *per-step* updates that include the actual board state (snake
position, food position, and current score). The other topic publishes per 
episode information such as high score, NN loss, current epsilon, and more.

The client can select *Turbo* mode to disable per-step updates. The board and
current score are no longer updated, but the server runs more than 15x faster.
This is useful for running experiments quickly.

The headless server uses a neural network and a policy stack to play the game.
The policy stack includes an implementation of the traditional Epsilon-Greedy
algorithm to encourage exploration at the beginning of a simulation run. 

## Installation

```
   $ python3 -m venv hydra-venv
   $ . hydra-venv/bin/activate
   hydra-venv> pip install ai-hydra
```

## Distributed Architecture

![Architecture](https://aihydra.osoyalce.com/images/architecture.png)

The *HydraClient*, *HydraRouter*, and *HydraMgr* are run in three different
terminals. The project supports running the client, router, and server on
different machines, but at this early stage in the project, it's recommended
that all three be run on the same machine.

## Startup

Start the *HydraClient* in the first terminal:

```
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-client
```

Start the *HydraRouter* in a second terminal:

```
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-router
```

Finally, start the *HydraMgr* in a third terminal:

```
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-mgr
```

Click the *Start* button in the *HydraRouter* to start the routing functions.
Click the *Handshake* button in the *HydraClient*. This causes a `HANDSHAKE` 
ZeroMQ message to be sent through the *HydraRouter* to the *HydraMgr*. 

- If the server cannot be reached, a console message appears.
- If the server can be reached
  - And no simulation is running, then the *Handshake* is replaced with a *Start* button, and settings appear which can be configured.
  - If a simulation is running, then the *Handshake* button is replaced with a *Update Config* button, and only runtime values are configurable. The settings from the running simulation are displayed in the *HydraClient*.

## Shutdown

In this early release, the server can only be stopped by hitting `Control-C` in 
the terminal where it's running. The client is stopped by hitting the *Quit* key, 
but the shutdown is currently not 100% clean, so an additional `Control-C` may be 
required to fully terminate the *HydraClient*. The *HydraRouter* can be shut down 
by clicking the *Quit* button.

A clean shutdown will be implemented for the client and server in a future release.

## Fully Deterministic

Simulations are fully deterministic with a few caveats:

- The user must click the *Turbo* button before starting the simulation
- The user cannot switch back to *Normal* mode during the simulation
- The HydraMgr must be restarted after a simulation run

This means that the **exact** same scores will be achieved at the exact same
game number during a simulation run. This makes true comparisons between runs
possible.

## Supported Models: Linear and RNN

This project includes a *Linear* and an *RNN* for the backend neural network.
The choice of model is made available in the TUI with a simple drop down 
menu.

## Visualizations

The TUI includes real-time visualzations. The screenshot at the top of this
document shows the actual Snake Game, a *high scores* widget, and two plots
at the bottom; a *high score* plot and a *current score* plot.

Other visualizations are shown below:

### Loss

The loss plotted over the duration of the running simulation and a short, 
75 episode, sliding window showing recent loss.

![Loss Plot](https://aihydra.osoyalce.com/images/loss-plot.png)

### Scores Histogram

This plot shows the score distribution over the simulation run and a second plot 
with a sliding window showing a histogram of the scores over the previous 500 games.

![Score Distribution](https://aihydra.osoyalce.com/images/scores-histogram.png)

## Snapshot Report

The *HydraClient TUI* includes a *Snapshot* button that creates a simple
text file that captures the simulation settings. A sample is shows below.

```
📸 AI Hydra - Snapshot
══════════════════════
Timestamp: 2026-03-19 06:28:53
Simulation Run Time:  3m  8s
Current Episode Number: 1076
AI Hydra Version: v0.17.0
Random Seed: 1970

🎯 Epsilon Greedy
═════════════════
Initial Epsilon: 0.999
Minimum Epsilon: 0.005
Epsilon Decay Rate: 0.993

🧠 RNN Model
════════════
Input Size: 16
Hidden Size: 256
Dropout Layer P-Value: 0.05
RNN Layers: 4
RNN Tau: 0.01

🧙 Training
═══════════
Learning Rate: 0.0008
Discount/Gamma: 0.96
Batch Size: 64
Sequence Length: 4
RNN Tau: 0.01

📚 Event Log Messages
═════════════════════
   Type                Time  Event
💻 Hydra Client          0s  Initialized...
⚡ Simulation            0s  Connected to simulation server
⚡ Simulation            0s  Turbo mode enabled: Game rendering disabled, move delay set to 0.0
⚡ Simulation            0s  Simulation started
💾 Replay Memory         0s  Epoch 0: Setting max_frames to 100000
💾 Replay Memory         0s  Epoch 0: Setting max_buckets to 20
💾 Replay Memory        38s  Epoch 459: Shifting into higher gear: 2
💾 Replay Memory     1m 16s  Epoch 651: Shifting into higher gear: 3
💾 Replay Memory     1m 28s  Epoch 704: Memory is full, pruning initiated
🎲 Epsilon           1m 41s  Epoch 754: Exploration complete: ε = 0.005

🏆 Highscore Events
═══════════════════
Episode Highscore        Time Epsilon
═══════ ═════════ ═══════════ ═══════
      2         1          0s   0.999
     82         2          4s  0.5576
     82         3          4s  0.5576
    170         4          8s  0.3135
    185         5          9s  0.2762
    279         6         17s  0.1407
    310         7         20s  0.1148
    340         8         22s   0.095
    353         9         24s  0.0837
    462        10         38s  0.0397
    498        11         45s  0.0309
    498        12         45s  0.0309
    498        13         45s  0.0309
    548        14         54s  0.0213
    548        15         54s  0.0213
    587        16      1m  1s  0.0162
    587        17      1m  1s  0.0162
    643        18      1m 13s  0.0109
    643        19      1m 13s  0.0109

```

## ATH Replay Memory

The ATH (Adaptive Temporal Horizon) Replay Memory is not a static buffer. It’s a system that evolves with the model.

Rather than locking in a single sequence length and batch size, ATH introduces a gearbox that shifts as the agent improves. Early on, it prioritizes speed and simplicity with short sequences and large batches. As the agent becomes more capable, it transitions into longer sequences and smaller batches, allowing the model to reason over deeper time horizons.

But temporal adaptation is only part of the story.

ATH Replay Memory also uses a bucketed sampling strategy to address a common issue in reinforcement learning: training data naturally follows the agent’s score distribution. Without intervention, the model would mostly train on “typical” games and rarely revisit edge cases such as high-scoring runs or rare failure modes.

To counter this, transitions are grouped into buckets based on outcome characteristics (e.g., score ranges). During sampling, these buckets are used to ensure a more balanced and representative training set, preventing the model from overfitting to the most common experiences. The *memory* widget at the bottom shows the buckets and how many sequences each bucket contains. The shading shows how full the buckets are relative to each other.

![ATH Memory](https://aihydra.osoyalce.com/images/ath-memory.png)

In practice, this means:

- Early training is fast and efficient
- Later training becomes more context-aware and strategic
- Rare but important experiences remain visible to the model
- The system adapts both what it learns from and how it learns

Replay memory itself remains clean and decoupled from global state. It emits lifecycle events (warm-up, capacity, gear shifts), which are enriched and tracked by the client. This makes the training process observable in terms of state transitions, not just outcomes.
phase transitions.

## Blazing Speed

Simulations run **BLAZINGLY** fast on a consumer grade laptop without a GPU.
This is due to careful architectural design decisions. For the *RNN*, the
`ReplayMemory`, `RNNModel`, and `RNNTrainer` work in a pipeline, minimizing
data transformations. The *Linear* model uses a similar strategy.
