AI Hydra: Hybrid Neural Network + Tree Search System
#####################################################

.. image:: https://aihydra.osoyalce.com/images/hydra-client.png
   :alt: HydraClient TUI

Overview
********

AI Hydra is a distributed reinforcement learning platform composed of three core components:

- **HydraClient** — a Textual TUI for control and visualization  
- **HydraRouter** — a lightweight message router  
- **HydraMgr** — a headless simulation and training engine  

All components communicate over **ZeroMQ**, forming a clean separation between control, execution, and visualization.

The client sends control messages (e.g., *handshake*, *start*) to the router, which forwards them to the server. The server runs the simulation and publishes telemetry via PUB/SUB sockets. The client subscribes to these streams and renders the system state in real time.

Telemetry is split into two channels:

- **Per-step updates** — board state (snake position, food, score)  
- **Per-episode updates** — metrics (high score, loss, epsilon, etc.)  

This separation allows fine-grained control over performance and observability.

----

Turbo Mode
**********

The client supports a **Turbo mode** that disables per-step updates. This removes rendering overhead and allows the simulation to run **15×+ faster**, making it ideal for rapid experimentation.

----

Installation
************

.. code-block:: shell

   $ python3 -m venv hydra-venv
   $ . hydra-venv/bin/activate
   hydra-venv> pip install ai-hydra

----

Distributed Architecture
************************

.. image:: https://aihydra.osoyalce.com/images/architecture.png
   :alt: Architecture

Each component runs independently:

- HydraClient
- HydraRouter
- HydraMgr

----

Startup
*******

Start each component in its own terminal:

1. Client
=========

.. code-block:: shell

    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-client

2. Router
=========

.. code-block:: shell

    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-router

3. Server
=========

.. code-block:: shell
    
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-mgr

Workflow
--------

After starting the three processes:

1. Click **Start** in the *HydraRouter*
2. Click **Handshake** in the *HydraClient*

This sends a `HANDSHAKE` message through the router to the server.

If the client cannot connect to the server, then nothing will happen.

If the server is reachable, then the **Start** button appears, and the 
simulation settings will also appear and be editable.

To launch the simulation, click on the **Start** button. Upon starting, the *
*Pause** and **Reset** buttons will appear. These allow the user to pause a 
running simulation, or reset the simulation, which takes you back to the 
*post-handshake* display.

----

Shutdown
********

In this early release, the server can only be stopped by hitting `Control-C` 
in the terminal where it's running. The client is stopped by hitting the 
**Quit** key. The *HydraRouter* can be shut down by clicking the *Quit* button.

The shutdown process is not 100% clean, so you may be required to also hit 
`Control-C` on the client to fully stop it.

----

Deterministic Simulations
*************************

Simulations are **fully deterministic.**

Given the same configuration and random seed:

- Episode progression is identical
- Scores occur at the same episodes
- Pausing/resuming does not affect outcomes

This enables reliable experimentation and reproducibility.

You can also vary only the random seed to validate that results are not artifacts of a *“lucky run.”*

----

Supported Models: Linear, RNN, and GRU
**************************************

- Linear Model
- Recurrent Neural Network (RNN)
- Gated Recurrent Network (GRU)

Model selection is available directly in the TUI.

----

Visualizations
**************

The TUI provides real-time insight into training:

- Live Snake board
- High score tracking
- Score plots

Views include:

High Scores, Current and Average Current Scores
===============================================

.. image:: https://aihydra.osoyalce.com/images/hydra-client.png
   :alt: Scores

Loss
====

- Full training loss over time
- Sliding window (recent 75 episodes)

.. image:: https://aihydra.osoyalce.com/images/loss-plot.png
   :alt: Loss Plot

Scores Distribution
===================

- Global score histogram
- Recent performance window (last 500 episodes)

.. image:: https://aihydra.osoyalce.com/images/scores-histogram.png
   :alt: Score Distribution

Event Log
=========

A structured event stream capturing system behavior:

- Simulation lifecycle
- Replay memory transitions
- Gear shifts and thresholds

.. image:: https://aihydra.osoyalce.com/images/event-log.png
   :alt: Event Log


----

ATH Replay Memory
*****************

The **ATH (Adaptive Temporal Horizon) Replay Memory** is a dynamic replay 
system that evolves with the agent.

Instead of storing isolated transitions for uniform replay, ATH stores 
**complete episodes** and generates **gear-specific metadata** describing how 
each episode can be sampled at different temporal horizons.

The ATH gearbox adjusts:

- **Sequence length**
- **Batch size**

Adaptive Training Dynamics
==========================

- Early training → short sequences, large batches
- Later training → long sequences, smaller batches

The system shifts gears based on observed performance rather than a fixed 
schedule.

Temporal Bucket Indexing
************************

ATH does not bucket experiences by score or outcome.

Instead, completed episodes are analyzed against the active gear, and metadata 
is used to determine which **end-aligned temporal chunks** are available for 
sampling.

This produces a bucketed index based on **temporal chunk position and usability**, 
not reward classification.

Sampling then draws across warmed buckets to preserve temporal variety in the 
training batch.

This gives ATH a few useful properties:

- Full episodes remain intact in storage
- Sampling adapts automatically as sequence length changes
- Training batches retain coverage across different temporal windows
- Replay stays observable and decoupled from the trainer

The TUI memory widget visualizes the current bucket distribution and how balanced 
the replay space is under the active gear.

.. image:: https://aihydra.osoyalce.com/images/ath-memory.png
   :alt: ATH Memory

System Properties
=================

- Training adapts **how** it learns through the gearbox
- Replay adapts **how episodes are sampled** through gear-aware temporal metadata
- Memory remains **decoupled, inspectable, and observable**

Lifecycle events such as warm-up, pruning, and gear shifts are emitted and 
tracked in real time.

----

Train Manager
*************

`TrainMgr` is the central coordinator for AI Hydra’s learning loop.

It is responsible for constructing and wiring together:

- the model and trainer
- replay memory
- the base action policy
- the Epsilon Nice policy wrapper
- stagnation detection and recovery behavior

`TrainMgr` does more than orchestrate training. It also acts as the system’s 
recovery controller.

When progress stalls, `TrainMgr` emits stagnation alerts to adaptive subsystems 
so they can respond in different ways:

- **ATH Gearbox** adjusts sequence length and batch size by shifting gears
- **Epsilon Nice** can be enabled as a bounded corrective policy layer

This makes stagnation a first-class runtime signal rather than a passive metric.

The result is a coordinated feedback system where memory dynamics and policy 
behavior can both adapt when learning plateaus.

---

Epsilon Nice - Bounded Safe-Detour Policy
*****************************************

Overview
========

**Epsilon Nice** is a conditionally enabled policy wrapper that can temporarily 
redirect action selection into a short safe-detour mode.

It is not a one-step correction and it is not always active.

Instead, once normal epsilon exploration has effectively ended, Epsilon Nice 
may be enabled by the training system. When enabled, it can probabilistically 
arm a temporary intervention window. During that window, action selection is 
redirected toward safe alternatives for a fixed number of steps.

This allows the agent to break out of locally bad or stagnant behavior without 
permanently replacing the underlying learned policy.

How It Works
============

At each step:

1. The base policy selects an action.
2. If normal epsilon exploration is still active, that action is returned unchanged.
3. If Nice is not enabled, the action is returned unchanged.
4. If Nice is enabled:
   - with probability `p_value`, Nice arms a detour
   - once armed, Nice remains active for `steps` turns
   - during those turns, it attempts to replace the suggested action with a safe alternative

So:

- `p_value` controls how often a Nice detour begins
- `steps` controls how long the detour lasts

Override Behavior
=================

During an active Nice detour, the policy looks at the other available actions 
and gathers the non-colliding alternatives.

- If safe alternatives exist, one is selected at random
- If no safe alternative exists, the original action is preserved

This means Nice does not simply “block fatal moves.”  
It temporarily moves the policy into a safe-alternative action mode for a 
bounded number of steps.

Key Properties
==============

- **Post-epsilon gated** - Nice does nothing while normal epsilon exploration is still active
- **Externally controlled** - Enabled and disabled by the training system
- **Probabilistic arming** - `p_value` controls entry into Nice mode
- **Multi-step intervention** - `steps` defines the length of the detour
- **Safe-alternative sampling** - Overrides choose from non-colliding alternatives when available
- **Deterministic-friendly** - Uses the shared RNG
- **Observable** - Reports rolling stats such as calls, triggers, overrides, fatal suggestions, and no-safe-alternative cases

Telemetry
=========

Epsilon Nice tracks and reports:

- calls
- triggers
- overrides
- fatal suggested actions
- no-safe-alternative cases
- trigger rate
- override rate

These statistics are emitted on rolling windows, allowing the TUI to show how 
often Nice is being consulted and how often it actually changes behavior.

Performance
***********

AI Hydra is designed for **high-throughput training on consumer hardware**, 
without relying on a GPU.

Its performance comes from two complementary design pillars:

1. Aligned Training Pipeline
============================

The training path is tightly aligned across memory, batching, and model execution.

- Replay memory stores full episodes and exposes **variable-length sequences via metadata**
- Sequences are constructed **without expensive transformation or reshaping**
- Batches are fed directly into the model’s **sequence-forward path**

This creates a continuous pipeline:

**memory → sequence assembly → batched training → model forward**

Because each stage is designed to match the next, the system avoids unnecessary 
copying, slicing, or recomputation.

2. Lean Simulation Loop
=======================

The core run loop is intentionally minimal and stays focused on simulation and training only.

- No rendering or plotting occurs in the training process
- Telemetry is published asynchronously via a **ZeroMQ PUB socket**
- Visualization, plotting, and event correlation are handled by a separate client (TUI)

This keeps the hot path free of UI and analysis overhead, allowing the simulation to run at full speed.

Result
======

- High episode throughput on CPU-only systems  
- Stable real-time telemetry without blocking the simulation  
- Fast iteration cycles with immediate observability  

AI Hydra achieves speed not through hardware, but through **alignment and separation of concerns**.
The system is optimized so that the cost of learning is dominated by model 
computation, not data movement or orchestration.

----

Snapshot Report
***************

The TUI includes a **Snapshot** feature that captures:

- Configuration
- Model parameters
- Replay memory state
- Event log
- Performance metrics

Snapshots are saved to a `AI-Hydra` directory that the system creates in the 
user's home directory.


Example Report
==============

.. code-block:: shell

    📸 AI Hydra - Snapshot Report
    ════════════════════════════
    Timestamp              : 2026-03-25 06:33:58
    Simulation Run Time    : 4h 50m
    Current Episode Number : 54655
    AI Hydra Version       : v0.19.2
    Random Seed            : 1970

    🎯 Epsilon Greedy
    ════════════════
    Initial Epsilon    : 0.999
    Minimum Epsilon    : 0.0
    Epsilon Decay Rate : 0.985

    🧠 GRU Model
    ═══════════
    Input Size            : 18
    Hidden Size           : 192
    Dropout Layer P-Value : 0.1
    Layers                : 3
    Learning Rate         : 0.001
    Discount/Gamma        : 0.97
    Batch Size            : 64
    Sequence Length       : 4
    Tau                   : 0.001

    💰 Rewards
    ═════════
    MAX_MOVES_MULTIPLIER : 100
    FOOD_REWARD          : 10
    COLLISION_PENALTY    : -10
    EMPTY_MOVE_REWARD    : 0.1
    CLOSER_TO_FOOD       : 0.1
    FURTHER_FROM_FOOD    : -0.1

    💾 Replay Memory
    ═══════════════
    MAX_TRAINING_FRAMES       : 512
    MAX_FRAMES                : 125000
    MAX_BUCKETS               : 20
    NUM_COOLDOWN_EPISODES     : 100
    THRESHOLD_BUCKETS         : [17, 18, 19]
    UPSHIFT_COUNT_THRESHOLD   : 150
    DOWNSHIFT_COUNT_THRESHOLD : 50
    MAX_STAGNANT_EPISODES     : 300
    MAX_HARD_RESET_EPISODES   : 2000

    📚 Event Log Messages
    ════════════════════
        -           0s    💻 Hydra Client    🔵 Initialized...
        -           0s    ⚡ Simulation      🔵 Connected to simulation server
        -           0s    ⚡ Simulation      🔵 Simulation started
        111           6s    🏁 ATH Gearbox     🟢 Shifting UP: 1 > 2 - 6/85
        212          19s    🏁 ATH Gearbox     🟢 Shifting UP: 2 > 3 - 8/64
        313          45s    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
        414       1m 18s    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
        455       1m 33s    💾 ATH Data Mgr    🔵 Memory is full (125000), pruning initiated
        457       1m 34s    🧭 Epsilon         🔵 Exploration complete: ε = 0.0
        515       1m 54s    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
        616       2m 31s    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
        717       3m  7s    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
        818       3m 53s    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
        919       4m 37s    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    1020       5m 14s    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    1121       6m  5s    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    1157       6m 19s    💪 Train Manager   🟡 Stagnation alert raised
    1157       6m 19s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    1222       6m 50s    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    1323       7m 31s    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    1424       8m  8s    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    1525       8m 48s    🏁 ATH Gearbox     🟢 Shifting UP: 15 > 16 - 34/15
    1626       9m 24s    🏁 ATH Gearbox     🟢 Shifting UP: 16 > 17 - 36/14
    1837      10m 40s    💪 Train Manager   🟢 Stagnation alert cleared
    1837      10m 40s    🙂 Nice Policy     🔵 Disabling EpsilonNice
    2137      12m 23s    💪 Train Manager   🟡 Stagnation alert raised
    2137      12m 23s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    2137      12m 23s    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 17 > 16 - 34/15
    2405      13m 56s    🏁 ATH Gearbox     🔵 Resetting stagnation count to 0
    2405      13m 56s    💪 Train Manager   🟢 Stagnation alert cleared
    2405      13m 56s    🙂 Nice Policy     🔵 Disabling EpsilonNice
    2705      15m 32s    💪 Train Manager   🟡 Stagnation alert raised
    2705      15m 32s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    2705      15m 32s    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 16 > 15 - 32/16
    2803      16m  3s    🏁 ATH Gearbox     🔵 Resetting stagnation count to 0
    2803      16m  3s    💪 Train Manager   🟢 Stagnation alert cleared
    2803      16m  3s    🙂 Nice Policy     🔵 Disabling EpsilonNice
    3125      17m 38s    💪 Train Manager   🟡 Stagnation alert raised
    3125      17m 38s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    3125      17m 38s    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 15 > 14 - 30/17
    3426      19m 15s    🏁 ATH Gearbox     🔵 Resetting stagnation count to 0
    3426      19m 15s    💪 Train Manager   🟢 Stagnation alert cleared
    3426      19m 15s    🙂 Nice Policy     🔵 Disabling EpsilonNice
    3834      21m 19s    💪 Train Manager   🟡 Stagnation alert raised
    3834      21m 19s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    3834      21m 19s    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 14 > 13 - 28/18
    3935      21m 51s    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    4673      25m 48s    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    4890      27m  3s    💪 Train Manager   🟢 Stagnation alert cleared
    4890      27m  3s    🙂 Nice Policy     🔵 Disabling EpsilonNice
    5190      28m 48s    💪 Train Manager   🟡 Stagnation alert raised
    5190      28m 48s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    5190      28m 48s    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 15 > 14 - 30/17
    5291      29m 21s    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    6890      38m 19s    💪 Train Manager   🔴 Critical stagnation event(1) detected
    6890      38m 19s    🏁 ATH Gearbox     🔴 Critical Stagnation alert(1): Radical DOWN shift: 15 > 3 - 8/64
    6990      38m 50s    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
    7091      39m 24s    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
    7191      39m 57s    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
    7293      40m 28s    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
    7394      40m 58s    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
    7495      41m 30s    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
    7596      42m  1s    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    7697      42m 35s    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    7798      43m  9s    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    7899      43m 42s    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    8000      44m 14s    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    8294      45m 54s    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    8890      49m 16s    💪 Train Manager   🔴 Critical stagnation event(2) detected
    8890      49m 16s    🏁 ATH Gearbox     🔴 Critical Stagnation alert(2): Radical DOWN shift: 15 > 3 - 8/64
    8990      49m 47s    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
    9091      50m 18s    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
    9192      50m 52s    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
    9293      51m 25s    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
    9394      51m 59s    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
    9494      52m 33s    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
    9596      53m  5s    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    9697      53m 40s    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    9798      54m 15s    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    9899      54m 50s    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    10000      55m 26s    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    10101      56m  1s    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    12648       1h 10m    🏁 ATH Gearbox     🟢 Shifting UP: 15 > 16 - 34/15
    12890       1h 12m    💪 Train Manager   🔴 Critical stagnation event(3) detected
    12890       1h 12m    🏁 ATH Gearbox     🔴 Critical Stagnation alert(3): Radical DOWN shift: 16 > 3 - 8/64
    12990       1h 12m    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
    13091       1h 13m    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
    13192       1h 13m    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
    13293       1h 14m    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
    13394       1h 14m    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
    13495       1h 15m    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
    13596       1h 16m    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    13697       1h 16m    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    13797       1h 17m    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    13899       1h 17m    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    14000       1h 18m    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    14101       1h 19m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    14184       1h 19m    💪 Train Manager   🟢 Stagnation alert cleared
    14184       1h 19m    🙂 Nice Policy     🔵 Disabling EpsilonNice
    14484       1h 21m    💪 Train Manager   🟡 Stagnation alert raised
    14484       1h 21m    🙂 Nice Policy     🔵 Enabling EpsilonNice
    14484       1h 21m    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 15 > 14 - 30/17
    14602       1h 21m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    17602       1h 39m    💪 Train Manager   🟢 Stagnation alert cleared
    17602       1h 39m    🙂 Nice Policy     🔵 Disabling EpsilonNice
    17902       1h 41m    💪 Train Manager   🟡 Stagnation alert raised
    17902       1h 41m    🙂 Nice Policy     🔵 Enabling EpsilonNice
    17902       1h 41m    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 15 > 14 - 30/17
    18003       1h 41m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    18828       1h 46m    🏁 ATH Gearbox     🟡 Shifting DOWN: 14 - 30/17
    19183       1h 48m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    19353       1h 49m    💪 Train Manager   🟢 Stagnation alert cleared
    19353       1h 49m    🙂 Nice Policy     🔵 Disabling EpsilonNice
    19653       1h 51m    💪 Train Manager   🟡 Stagnation alert raised
    19653       1h 51m    🙂 Nice Policy     🔵 Enabling EpsilonNice
    19653       1h 51m    🏁 ATH Gearbox     🟡 Stagnation alert(1) - Shifting DOWN: 15 > 14 - 30/17
    20040       1h 53m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    23243       2h 11m    🏁 ATH Gearbox     🟡 Shifting DOWN: 14 - 30/17
    23577       2h 13m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    25353       2h 23m    💪 Train Manager   🔴 Critical stagnation event(4) detected
    25353       2h 23m    🏁 ATH Gearbox     🔴 Critical Stagnation alert(4): Radical DOWN shift: 15 > 3 - 8/64
    25453       2h 24m    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
    25554       2h 24m    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
    25655       2h 25m    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
    25756       2h 25m    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
    25857       2h 26m    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
    25958       2h 27m    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
    26059       2h 27m    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    26160       2h 28m    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    26261       2h 28m    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    26362       2h 29m    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    26463       2h 29m    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    26571       2h 30m    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    29289       2h 45m    🏁 ATH Gearbox     🟢 Shifting UP: 15 > 16 - 34/15
    30277       2h 51m    🏁 ATH Gearbox     🟡 Shifting DOWN: 15 - 32/16
    30746       2h 53m    🏁 ATH Gearbox     🟡 Shifting DOWN: 14 - 30/17
    32364       3h  2m    🏁 ATH Gearbox     🟡 Shifting DOWN: 13 - 28/18
    33353       3h  6m    💪 Train Manager   🔴 Critical stagnation event(5) detected
    33353       3h  6m    🏁 ATH Gearbox     🔴 Critical Stagnation alert(5): Radical DOWN shift: 13 > 3 - 8/64
    33453       3h  7m    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
    33554       3h  7m    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
    33655       3h  8m    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
    33756       3h  8m    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
    33857       3h  9m    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
    33958       3h  9m    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
    34059       3h 10m    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    34160       3h 10m    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    34261       3h 11m    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    34362       3h 11m    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    43353       3h 55m    💪 Train Manager   🔴 Critical stagnation event(6) detected
    43353       3h 55m    🏁 ATH Gearbox     🔴 Critical Stagnation alert(6): Radical DOWN shift: 13 > 3 - 8/64
    43453       3h 56m    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
    43554       3h 56m    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
    43655       3h 57m    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
    43756       3h 57m    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
    43856       3h 58m    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
    43958       3h 58m    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
    44059       3h 59m    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    44160       3h 59m    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    44261       4h  0m    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    44519       4h  1m    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    50268       4h 29m    🏁 ATH Gearbox     🟡 Shifting DOWN: 12 - 26/19
    51065       4h 33m    🏁 ATH Gearbox     🟡 Shifting DOWN: 11 - 24/21
    52007       4h 37m    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    52860       4h 41m    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18

    🏆 Highscore Events
    ══════════════════
    Epoch  Highscore  Time     Epsilon
    ═════  ═════════  ═══════  ═══════
    0      0          0s       0.9900 
    18     1          0s       0.7726 
    124    2          8s       0.1629 
    137    3          9s       0.1338 
    137    4          9s       0.1338 
    192    5          15s      0.0566 
    192    6          15s      0.0566 
    192    7          15s      0.0566 
    192    8          15s      0.0566 
    291    9          37s      0.0125 
    291    10         37s      0.0125 
    294    11         39s      0.0119 
    294    12         39s      0.0119 
    294    13         39s      0.0119 
    327    14         49s      0.0071 
    381    15          1m  7s  0.0032 
    381    16          1m  7s  0.0032 
    455    17          1m 33s  0.0010 
    541    18          2m  4s  0.0000 
    541    19          2m  4s  0.0000 
    650    20          2m 41s  0.0000 
    679    21          2m 53s  0.0000 
    679    22          2m 53s  0.0000 
    679    23          2m 53s  0.0000 
    679    24          2m 53s  0.0000 
    679    25          2m 53s  0.0000 
    679    26          2m 53s  0.0000 
    800    27          3m 43s  0.0000 
    800    28          3m 43s  0.0000 
    800    29          3m 43s  0.0000 
    800    30          3m 43s  0.0000 
    800    31          3m 43s  0.0000 
    800    32          3m 43s  0.0000 
    857    33          4m 12s  0.0000 
    857    34          4m 12s  0.0000 
    857    35          4m 12s  0.0000 
    1837   36         10m 40s  0.0000 
    1837   37         10m 40s  0.0000 
    1837   38         10m 40s  0.0000 
    1837   39         10m 40s  0.0000 
    1837   40         10m 40s  0.0000 
    2405   41         13m 56s  0.0000 
    2803   42         16m  3s  0.0000 
    2819   43         16m  8s  0.0000 
    2825   44         16m 11s  0.0000 
    2825   45         16m 11s  0.0000 
    2825   46         16m 11s  0.0000 
    2825   47         16m 11s  0.0000 
    2825   48         16m 11s  0.0000 
    3426   49         19m 15s  0.0000 
    3534   50         19m 47s  0.0000 
    3534   51         19m 47s  0.0000 
    3534   52         19m 47s  0.0000 
    3534   53         19m 47s  0.0000 
    3534   54         19m 47s  0.0000 
    3534   55         19m 47s  0.0000 
    3534   56         19m 47s  0.0000 
    4890   57         27m  3s  0.0000 
    4890   58         27m  3s  0.0000 
    4890   59         27m  3s  0.0000 
    4890   60         27m  3s  0.0000 
    4890   61         27m  3s  0.0000 
    4890   62         27m  3s  0.0000 
    4890   63         27m  3s  0.0000 
    4890   64         27m  3s  0.0000 


    🙂 Epsilon Nice
    ══════════════
    P-Value    : 0.001
    Nice Steps : 20

    🙂 Epsilon Nice Events
    ═════════════════════
    Window       Epoch  Calls   Triggered  Fatal Suggested  Overrides  No Safe Alt  Trigger Rate  Override Rate
    ═══════════  ═════  ══════  ═════════  ═══════════════  ═════════  ═══════════  ════════════  ═════════════
    1-500        500    0       0          0                0          0            0.0000        0.0000       
    501-1000     1000   0       0          0                0          0            0.0000        0.0000       
    1001-1500    1500   191421  3780       0                3563       217          0.0197        0.0186       
    1501-2000    2000   153238  2880       1                2728       152          0.0188        0.0178       
    2001-2500    2500   115393  2460       1                2339       121          0.0213        0.0203       
    2501-3000    3000   35131   500        1                464        36           0.0142        0.0132       
    3001-3500    3500   115424  2540       2                2358       182          0.0220        0.0204       
    3501-4000    4000   65779   1180       0                1094       86           0.0179        0.0166       
    4001-4500    4500   186080  3620       0                3333       287          0.0195        0.0179       
    4501-5000    5000   158790  3380       1                3142       238          0.0213        0.0198       
    5001-5500    5500   122657  2420       0                2227       193          0.0197        0.0182       
    5501-6000    6000   200011  4080       2                3748       332          0.0204        0.0187       
    6001-6500    6500   201276  4360       1                4040       320          0.0217        0.0201       
    6501-7000    7000   200522  3760       5                3451       309          0.0188        0.0172       
    7001-7500    7500   199207  4020       1                3747       273          0.0202        0.0188       
    7501-8000    8000   200504  4120       3                3807       313          0.0205        0.0190       
    8001-8500    8500   199125  4020       0                3701       319          0.0202        0.0186       
    8501-9000    9000   195740  4100       2                3799       301          0.0209        0.0194       
    9001-9500    9500   210212  4240       1                3927       313          0.0202        0.0187       
    9501-10000   10000  209103  4120       1                3818       302          0.0197        0.0183       
    

    ⚙️  ATH Shift / Mean / Median
    ═════════════════════════════
    Epoch  Gear  Seq Length  Batch Size  Mean   Median
    ═════  ════  ══════════  ══════════  ═════  ══════
    500    5     12          42          4.88   3.00  
    1000   10    22          23          8.75   9.00  
    1500   15    32          16          10.22  10.00 
    2000   17    36          14          10.64  11.00 
    2500   16    34          15          11.76  12.00 
    3000   15    32          16          12.88  13.00 
    3500   14    30          17          13.93  13.00 
    4000   14    30          17          15.01  14.00 
    4500   14    30          17          15.88  15.00 
    5000   15    32          16          16.87  16.00 
    5500   15    32          16          17.73  17.00 
    6000   15    32          16          18.40  17.00 
    6500   15    32          16          18.99  18.00 
    7000   4     10          51          19.49  19.00 
    7500   9     20          25          19.92  19.00 
    8000   14    30          17          20.30  19.00 
    8500   15    32          16          20.63  20.00 
    9000   4     10          51          20.91  20.00 
    9500   9     20          25          21.26  20.00 
    10000  14    30          17          21.56  21.00 


    🪣 ATH Memory Bucket Usage
    ═════════════════════════
    Epoch  b1   b2   b3   b4   b5   b6   b7   b8   b9   b10  b11  b12  b13  b14  b15  b16  b17  b18  b19  b20
    ═════  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══
    500    288  286  285  281  281  280  276  272  266  265  258  253  248  243  235  232  226  219  209  204
    1000   220  220  219  219  217  214  212  210  206  201  192  184  178  166  162  155  145  137  129  125
    1500   242  241  240  239  235  228  211  203  197  176  168  159  142  128  114  101  90   82   69   62 
    2000   293  292  290  289  277  260  245  219  177  159  131  109  90   79   70   63   55   45   34   28 
    2500   285  284  279  276  270  263  247  235  217  191  173  150  132  101  85   68   57   43   38   32 
    3000   355  353  346  336  315  295  272  247  213  171  138  119  100  89   66   55   45   37   31   21 
    3500   328  327  326  322  313  291  274  252  230  207  182  165  140  119  102  82   62   52   38   33 
    4000   326  326  325  318  314  300  287  265  239  211  191  167  148  126  106  83   65   54   44   30 
    4500   320  319  318  312  306  295  275  254  234  219  201  178  156  133  104  78   68   53   42   36 
    5000   300  299  294  291  286  276  262  247  229  210  188  173  148  124  101  81   65   46   37   31 
    5500   314  314  311  305  298  284  267  247  228  209  182  157  140  116  94   69   54   41   31   22 
    6000   311  310  307  302  295  278  261  244  231  205  183  160  132  109  87   77   62   50   43   32 
    6500   314  311  308  304  296  290  277  259  241  213  185  165  140  111  96   76   51   38   24   18 
    7000   305  305  305  304  303  303  302  301  301  299  299  299  298  296  292  290  287  287  284  283
    7500   324  324  323  322  320  312  306  299  292  278  269  258  246  236  221  204  182  170  149  137
    8000   313  312  307  303  296  285  275  253  233  212  198  180  158  134  112  95   82   62   52   38 
    8500   313  311  309  301  289  282  274  254  230  205  183  166  144  116  93   82   69   45   34   22 
    9000   324  324  324  324  323  323  323  322  319  319  316  314  312  311  305  304  299  296  294  287
    9500   294  294  293  292  290  288  287  282  276  269  262  256  246  238  224  217  209  187  172  157
    10000  294  294  292  289  282  271  260  246  232  214  194  178  168  144  123  108  94   81   67   49 

----

Closing Thoughts
****************

AI Hydra is not just a simulation engine — it is a **learning ecosystem**.

Its components — adaptive replay memory, dynamic policy layers, and real-time 
orchestration — do not operate in isolation. They interact.

- The gearbox reshapes temporal learning
- The policy adapts under changing constraints
- Recovery mechanisms respond to stagnation signals

Together, they form a system where behavior is not explicitly programmed, but 
**emerges** from the interaction of these parts.

The result is a tightly coupled environment where:

- learning dynamics are visible
- iteration is fast
- and meaningful behavior can emerge from system-level feedback

AI Hydra is not just running games.

It is **evolving behavior.**