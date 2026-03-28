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

Distributed Architecture
************************

.. image:: https://aihydra.osoyalce.com/images/architecture.png
   :alt: Architecture

**AI Hydra** components run independently and can be deployed across different machines.

### Example Deployment

This configuration demonstrates a fully distributed setup where simulation, routing, and visualization are physically separated across hosts. This example also uses custom ports. The *Hydra Client* is running locally, the *Hydra Router* is running on **bingo**, and the *Hydra Manager* is running on **islands**.

**1. Client**

.. code-block:: shell

    # Client (local machine)
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-client --router-port 6757 --router-hb-port 6758 --server-pub-port 6760 --router-address bingo --server-address islands


**2. Router**

.. code-block:: shell

    # Router (bingo)
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-router --port 6757 --hb-port 6758

**3. Server**

.. code-block:: shell

    # Manager / Simulation (islands)
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-mgr --port 5759 --router-port 6757 --router-hb-port 6758 --router-address bingo

----

Settings
********

Random Seed
===========

.. image:: https://aihydra.osoyalce.com/images/random-seed.png
   :alt: Random Seed


**AI Hydra** is fully deterministic and the random seed for a simulation run is visible and can be set in the TUI.

Move Delay
==========

.. image:: https://aihydra.osoyalce.com/images/move-delay.png
   :alt: Move Delay

When the a simulation is not running in *turbo mode*, the movement of the snake is too fast to see clearly. The **move delay** introduces a delay between steps that slows the simulation down.

Config Settings
===============

.. image:: https://aihydra.osoyalce.com/images/settings-config.png
   :alt: Config Settings

The **Config** settings tab allows the user to set:

- **Epsilon** - Initial, minimum, and the epsilon decay rate. The *current epsilon* value is also shown here
- **Model** - The **model type** (*Linear*, *RNN*, or *GRU*), the number of nodes in the hidden layers, the *p-value* of the model's dropout layer, and the number of layers is configurable here.
- **Training** - The *learning rate*, *discount/gamma* and *tau* settings can be set here. The current *batch size* and *sequence length* (as set by the **ATH Memory**) are displayed here.

Memory Settings
===============

.. image:: https://aihydra.osoyalce.com/images/settings-memory.png
   :alt: Memory Settings

The **Memory** tab allows the user to configure the **ATH Memory**.

- **Memory Sizing**
  - The **Max Frames** sets the maximum number of stored frames. The **ATH Memory** stores complete games. When the totaly number of frames exceeds this value, the oldest game is deleted from memory.
  - The **Memory Buckets** is a read-only setting that shows how many memory buckets are used.
  - The **Max Training Frames** sets the maximum number of frames (**sequnce length** * **batch size**) used during training.
- **Gearbox Settings**
  - The **Highest Gear** can be set here to limit the **sequence length** and **batch size**.
  - The **Cooldown Threshold** determines the minimum number of episodes that must be executed before a gear shift can occur.
  - The **Upshift Threshold** and **DownShift Threshold** determine when the **ATH Memory** shifts up or down, respectively. This number is the total number of frames stored in the last three **memory buckets**.

Rewards Settings
================

.. image:: https://aihydra.osoyalce.com/images/settings-rewards.png
   :alt: Rewards Settings

- **Reward Structure** - This section contains the reward that is allocated to the AI when the snake finds food, hits the wall or itself, or exceeds the maximum number of moves.
- **Movement Incentives** - This section contains rewards that can be assigned for moving into an empty square, moving towards the food, and moving away from the food.
- **Max Moves Multiplier** - This setting is used to configure the maximum number of moves the AI can make before the game ends. This is to avoid games where the AI circles endlessly. The maximum number of allowed moved is calculated by multiplying the length of the snake by this **max moves multiplier**.

Epsilon Nice Settings
=====================

.. image:: https://aihydra.osoyalce.com/images/settings-nice.png
   :alt: EpsilonNice Settings


The **EpsilonNice** policy execute a configurable number of steps in a random direction such that the move does not result in a collision (if possible).

- **Nice P-Value** - The probability that the **EpsilonNice** policy will be activated.
- **Nice Steps** - The number of consecutive game steps that will be executed using the **EpsilonNice** algorithm.

Network Settings
================

.. image:: https://aihydra.osoyalce.com/images/settings-network.png
   :alt: Network Settings

This section shows the hostnames and port numbers that **AI Hydra** is using. These can be configured during startup using command line switches.

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

.. image:: https://aihydra.osoyalce.com/images/scores.png
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
    Timestamp              : 2026-03-28 11:46:12
    Simulation Run Time    :  9m 53s
    Current Episode Number : 1832
    AI Hydra Version       : v0.23.0
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
    Tau                   : 0.001

    💰 Rewards
    ═════════
    Food Reward          : 10.0
    Collision Penalty    : -10.0
    Max Moves Penalty    : 0.0
    Empty Move Reward    : 0.1
    Closer to Food       : 0.1
    Further from Food    : -0.1
    Max Moves Multiplier : 100

    💾 Replay Memory
    ═══════════════
    Max Frames                  : 125000
    Memory Buckets              : 20
    Max Training Frames         : 512
    Highest Gear                : 26
    Cooldown Threshold          : 100
    Upshift Threshold           : 150
    Downshift Threshold         : 50
    Stagnant Threshold          : 300
    Critical Stagnant Threshold : 2000

    📚 Event Log Messages
    ════════════════════
        -           0s    ⚡ Simulation      🔵 Simulation started
        111           9s    🏁 ATH Gearbox     🟢 Shifting UP: 1 > 2 - 6/85
        211          25s    🏁 ATH Gearbox     🟢 Shifting UP: 2 > 3 - 8/64
        313          55s    🏁 ATH Gearbox     🟢 Shifting UP: 3 > 4 - 10/51
        414       1m 32s    🏁 ATH Gearbox     🟢 Shifting UP: 4 > 5 - 12/42
        455       1m 47s    💾 ATH Data Mgr    🔵 Memory is full (125000), pruning initiated
        456       1m 49s    🧭 Epsilon         🔵 Exploration complete: ε = 0.0
        515       2m 15s    🏁 ATH Gearbox     🟢 Shifting UP: 5 > 6 - 14/36
        616       3m  4s    🏁 ATH Gearbox     🟢 Shifting UP: 6 > 7 - 16/32
        717       3m 52s    🏁 ATH Gearbox     🟢 Shifting UP: 7 > 8 - 18/28
        818       4m 46s    🏁 ATH Gearbox     🟢 Shifting UP: 8 > 9 - 20/25
        919       5m 34s    🏁 ATH Gearbox     🟢 Shifting UP: 9 > 10 - 22/23
    1020       6m 21s    🏁 ATH Gearbox     🟢 Shifting UP: 10 > 11 - 24/21
    1121       7m  4s    🏁 ATH Gearbox     🟢 Shifting UP: 11 > 12 - 26/19
    1222       7m 48s    🏁 ATH Gearbox     🟢 Shifting UP: 12 > 13 - 28/18
    1297       8m 21s    💪 Train Manager   🟡 Stagnation alert raised
    1297       8m 21s    🙂 Nice Policy     🔵 Enabling EpsilonNice
    1322       8m 25s    🏁 ATH Gearbox     🟢 Shifting UP: 13 > 14 - 30/17
    1423       8m 41s    🏁 ATH Gearbox     🟢 Shifting UP: 14 > 15 - 32/16
    1523       8m 58s    🏁 ATH Gearbox     🟢 Shifting UP: 15 > 16 - 34/15

    🏆 Highscore Events
    ══════════════════
    Epoch  Highscore  Time     Epsilon
    ═════  ═════════  ═══════  ═══════
    18     1          0s       0.7726 
    124    2          11s      0.1510 
    137    3          13s      0.1299 
    137    4          13s      0.1299 
    192    5          20s      0.0566 
    192    6          21s      0.0540 
    192    7          21s      0.0540 
    192    8          21s      0.0540 
    291    9          47s      0.0123 
    291    10         47s      0.0123 
    294    11         48s      0.0119 
    294    12         48s      0.0119 
    294    13         48s      0.0119 
    327    14          1m  0s  0.0071 
    381    15          1m 20s  0.0032 
    381    16          1m 20s  0.0032 
    455    17          1m 47s  0.0010 
    541    18          2m 28s  0.0000 
    541    19          2m 28s  0.0000 
    667    20          3m 26s  0.0000 
    721    21          3m 54s  0.0000 
    724    22          3m 57s  0.0000 
    724    23          3m 57s  0.0000 
    724    24          3m 57s  0.0000 
    744    25          4m  8s  0.0000 
    744    26          4m  8s  0.0000 
    744    27          4m  8s  0.0000 
    744    28          4m  8s  0.0000 
    777    29          4m 24s  0.0000 
    787    30          4m 31s  0.0000 
    997    31          6m 11s  0.0000 
    997    32          6m 11s  0.0000 
    997    33          6m 11s  0.0000 
    997    34          6m 11s  0.0000 
    997    35          6m 11s  0.0000 
    997    36          6m 11s  0.0000 

    🙂 Epsilon Nice
    ══════════════
    P-Value    : 0.001
    Nice Steps : 20

    🙂 Epsilon Nice Events
    ═════════════════════
    Window     Epoch  Calls  Triggered  Fatal Suggested  Overrides  No Safe Alt  Trigger Rate  Override Rate
    ═════════  ═════  ═════  ═════════  ═══════════════  ═════════  ═══════════  ════════════  ═════════════
    1-500      500    0      0          0                0          0            0.0000        0.0000       
    501-1000   1000   0      0          0                0          0            0.0000        0.0000       
    1001-1500  1500   17469  3513       0                3513       0            0.2011        0.2011       

    ⚙️  ATH Shift / Mean / Median
    ═════════════════════════════
    Epoch  Gear  Seq Length  Batch Size  Mean  Median
    ═════  ════  ══════════  ══════════  ════  ══════
    500    5     12          42          4.88  3.00  
    1000   10    22          23          9.23  9.00  
    1500   15    32          16          8.55  8.00  

    🪣 ATH Memory Bucket Usage
    ═════════════════════════
    Epoch  b1   b2   b3   b4   b5   b6   b7   b8   b9   b10  b11  b12  b13  b14  b15  b16  b17  b18  b19  b20
    ═════  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══
    500    288  286  285  281  281  280  276  272  266  265  258  253  248  243  235  232  226  219  209  204
    1000   219  219  218  216  214  213  208  204  196  191  183  176  168  155  145  137  132  121  112  102
    1500   383  308  264  241  218  201  192  180  173  140  137  130  112  105  96   74   68   63   53   49

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