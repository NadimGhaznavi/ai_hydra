# AI Hydra: Hybrid Neural Network + Tree Search System

## Overview

AI Hydra is a distributed application that includes a Textual TUI client, a 
simple router, and a headless server. Communication is over ZeroMQ. The basic
functionality is that the Client sends control messages (e.g. handshake, start)
to the router which forwards them to the server. When the server is running
it publishes board and other telemetry information on a ZeroMQ PUB/SUB socket.
The client subscribes to the server's PUB socket and displays the game state.

The server splits the simulation telemetric into two ZeroMQ PUB topics. One 
topic provides *per-step* updates that include the actual board state (snake
position, food position and current score). The other topic publishes per 
episode information such as highscore, NN loss, current epsilon and more.

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
  - And no simulation is running then the *Handshake* is replaced with a *Start* button and settings appear which can be configured.
  - If a simulation is running, then the *Handshake* button is replaced with a *Update Config* button and only runtime values are configurable. The settings from the running simulation are displayed in the *HydraClient*.

## Shutdown

In this early release the server can only be stopped by hitting `Control-C` in 
the terminal where it's running. The client is stopped by hitting the *Quit* key, 
but the shutdown is currently not clean, so an additional `Control-C` may be 
required to fully terminate the *HydraClient*. The *HydraRouter* can be shutdown 
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

## Look Ahead Feature

The server includes a *lookahead* policy where the server checks to see if
the move suggested by the neural network will result in a collision. If so,
then it "looks ahead" at alternative moves. If a *look-ahead* move results
in finding food, then that move is selected. If a move does **not** result in
a collision, then that move is selected.

The *look-ahead* policy is only enabled for a configurable probability of the
time. The training of the neural network includes the moves and game state
information that was executed. So when the *look-ahead* policy is used the
training data is enhanced. This leads to better neural network performance.

## Visualizations

The TUI includes real-time visualzations that show the loss and a histogram
of the scores. The score histgrams also show the *mean* and *median* values.