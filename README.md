![HydraClient TUI](https://github.com/NadimGhaznavi/ai_hydra/blob/main/images/hydra-client.png)

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

![Architecture](https://github.com/NadimGhaznavi/ai_hydra/blob/main/images/architecture.png)

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

![Loss Plot](https://github.com/NadimGhaznavi/ai_hydra/blob/main/images/loss-plot.png)

### Scores Histogram

This plot shows the score distribution over the simulation run and a second plot 
with a sliding window showing a histogram of the scores over the previous 500 games.

![Score Distribution](https://github.com/NadimGhaznavi/ai_hydra/blob/main/images/scores-histogram.png)

## Snapshot Report

The *HydraClient TUI* includes a *Snapshot* button that creates a simple
text file that captures the simulation settings. A sample is shows below.

```
📸 AI Hydra - Snapshot
══════════════════════
Timestamp: 2026-03-14 13:00:38
Simulation Run Time: 1h  0m
Episode Number: 7585
AI Hydra Version: v0.14.1
Random Seed: 129

🎯 Epsilon Greedy
═════════════════
Initial Epsilon: 0.999
Minimum Epsilon: 0.005
Epsilon Decay Rate: 0.993

🧠 RNN Model
════════════
Input Size: 16
Hidden Size: 320
RNN Layers: 5
Dropout Layer P-Value: 0.1
Sequence Length: 32
Batch Size: 64

🏆 Highscore Events
═══════════════════
Episode Highscore        Time Epsilon
═══════ ═════════ ═══════════ ═══════
      2         1          0s   0.999
     91         2         10s  0.5235
    118         3         15s  0.4423
    189         4         33s  0.2705
    189         5         33s   0.263
    215         6         40s  0.2222
    215         7         40s  0.2222
    295         8      1m  1s  0.1267
    324         9      1m  9s  0.1026
    415        10      1m 39s  0.0545
    428        11      1m 43s  0.0501
    428        12      1m 43s  0.0501
    428        13      1m 43s  0.0501
    484        14      2m  1s  0.0336
```

## Blazing Speed

Simulations run **BLAZINGLY** fast on a consumer grade laptop without a GPU.
This is due to careful architectural design decisions. For the *RNN*, the
`ReplayMemory`, `RNNModel`, and `RNNTrainer` work in a pipeline, minimizing
data transformations. The *Linear* model uses a similar strategy.
