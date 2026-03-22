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

Supported Models: Linear and RNN
********************************

- Linear Model
- Recurrent Neural Network (RNN)

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

The **ATH (Adaptive Temporal Horizon) Replay Memory** is a dynamic system that evolves with the agent.

Instead of using fixed training parameters, ATH introduces a gearbox that adjusts:

- **Sequence length**
- **Batch size**

Adaptive Training Dynamics
==========================

- Early training → short sequences, large batches (fast learning)
- Later training → long sequences, small batches (deep temporal reasoning)

The system shifts gears based on observed performance, not fixed schedules.

Bucketed Sampling
=================

Training data is not uniformly useful.

Without intervention, the model overfits to typical gameplay and neglects:

- Rare high-scoring runs
- Edge-case failures

ATH addresses this by grouping transitions into buckets based on outcomes.

Sampling pulls from across buckets, ensuring:

- Diverse training signals
- Visibility into rare but critical experiences
- Reduced bias toward median performance

The TUI memory widget visualizes:

- Bucket distribution
- Relative capacity
- System balance over time

.. image:: https://aihydra.osoyalce.com/images/ath-memory.png
   :alt: ATH Memory

System Properties
=================

- Training adapts **how** it learns (gearbox)
- Training adapts **what** it learns (buckets)
- Replay memory remains **decoupled and observable**

Lifecycle events (warm-up, capacity, gear shifts) are emitted and tracked in real time.

----

Epsilon Nice - Post-Exploration Safety Layer
********************************************

Overview
========

**Epsilon Nice** is a lightweight, post-exploration policy layer that improves agent stability by preventing immediate fatal actions.

It operates after epsilon-greedy exploration has completed and applies a minimal, probabilistic correction to the selected action when necessary.

The goal is not to guide the agent, but to preserve *late-stage* learning by avoiding obviously bad decisions.

### Behavior

At each step (after epsilon has decayed to zero):

1. The base policy selects an action.
2. With probability `p_value`, **Epsilon Nice** is activated.
3. If the selected action would result in an immediate collision:
  - A non-colliding alternative is selected (if available).
4. Otherwise, the original action is preserved.

If no safe alternative exists, the action is left unchanged.

Key Properties
==============

- **Minimal intervention** - Only applies to immediate, one-step collisions.
- **Probabilistic** - Controlled by `p_value`` (e.g., `0.005`).
- **Post-epsilon only** - Disabled during exploration to avoid interfering with learning.
- **Model-agnostic** - Works with both Linear and RNN policies.
- **Deterministic-friendly** - Uses the same RNG as the rest of the system.

----

Performance
***********

AI Hydra is designed for **high-throughput training on consumer hardware.**

Key design choices:

- Pipeline architecture (memory → model → trainer)
- Minimal data transformation overhead
- Asynchronous telemetry via ZeroMQ

The result: **extremely fast simulations without requiring a GPU**

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
    Timestamp              : 2026-03-20 07:48:40
    Simulation Run Time    : 48m 10s
    Current Episode Number : 11705
    AI Hydra Version       : v0.17.0
    Random Seed            : 1970

    🎯 Epsilon Greedy
    ════════════════
    Initial Epsilon    : 0.999
    Minimum Epsilon    : 0.0
    Epsilon Decay Rate : 0.993

    🧠 RNN Model
    ═══════════
    Input Size            : 16
    Hidden Size           : 256
    Dropout Layer P-Value : 0.1
    RNN Layers            : 4
    Learning Rate         : 0.001
    Discount/Gamma        : 0.96
    Batch Size            : 64
    Sequence Length       : 4
    RNN Tau               : 0.001

    💾 Replay Memory
    ═══════════════
    Gearbox : 
    {1: (4, 64),
    2: (8, 32),
    3: (16, 16),
    4: (24, 10),
    5: (32, 8),
    6: (40, 6),
    7: (48, 5),
    8: (56, 4),
    9: (64, 3)}

    📚 Event Log Messages
    ════════════════════
    💻 Hydra Client          0s  Initialized...
    ⚡ Simulation            0s  Connected to simulation server
    ⚡ Simulation            0s  Turbo mode enabled: Game rendering disabled, move delay set to 0.0
    ⚡ Simulation            0s  Simulation started
    💾 ATH Data Mgr          0s  Epoch 0: Initialized
    ⚙️  ATH Gearbox           6s  Epoch 225: Shifting UP: 2
    ⚙️  ATH Gearbox          23s  Epoch 482: Shifting UP: 3
    💾 ATH Data Mgr      1m 21s  Epoch 962: Memory is full, pruning initiated
    ⚙️  ATH Gearbox       1m 29s  Epoch 1013: Shifting UP: 4
    ⚙️  ATH Gearbox       1m 52s  Epoch 1214: Shifting DOWN: 3
    ⚙️  ATH Gearbox       2m 24s  Epoch 1415: Shifting UP: 4
    ⚙️  ATH Gearbox       4m 19s  Epoch 1976: Shifting UP: 5
    ⚙️  ATH Gearbox       5m  1s  Epoch 2177: Shifting DOWN: 4
    ⚙️  ATH Gearbox       5m 51s  Epoch 2379: Shifting UP: 5
    ⚙️  ATH Gearbox       6m 42s  Epoch 2580: Shifting DOWN: 4
    ⚙️  ATH Gearbox       7m 35s  Epoch 2781: Shifting UP: 5
    ⚙️  ATH Gearbox      10m 22s  Epoch 3397: Shifting DOWN: 4
    ⚙️  ATH Gearbox      11m 11s  Epoch 3598: Shifting UP: 5
    ⚙️  ATH Gearbox      11m 54s  Epoch 3799: Shifting DOWN: 4
    ⚙️  ATH Gearbox      12m 46s  Epoch 4000: Shifting UP: 5
    ⚙️  ATH Gearbox      13m 46s  Epoch 4201: Shifting DOWN: 4
    ⚙️  ATH Gearbox      14m 36s  Epoch 4402: Shifting UP: 5
    ⚙️  ATH Gearbox      15m 56s  Epoch 4724: Shifting DOWN: 4
    ⚙️  ATH Gearbox      16m 49s  Epoch 4925: Shifting UP: 5
    ⚙️  ATH Gearbox      21m 50s  Epoch 6106: Shifting DOWN: 4
    ⚙️  ATH Gearbox      22m 45s  Epoch 6307: Shifting UP: 5
    ⚙️  ATH Gearbox      23m 33s  Epoch 6508: Shifting DOWN: 4
    ⚙️  ATH Gearbox      24m 25s  Epoch 6719: Shifting UP: 5
    ⚙️  ATH Gearbox      25m 14s  Epoch 6920: Shifting DOWN: 4
    ⚙️  ATH Gearbox      26m 11s  Epoch 7126: Shifting UP: 5
    ⚙️  ATH Gearbox      26m 59s  Epoch 7327: Shifting DOWN: 4
    ⚙️  ATH Gearbox      27m 55s  Epoch 7527: Shifting UP: 5
    ⚙️  ATH Gearbox      28m 58s  Epoch 7729: Shifting DOWN: 4
    ⚙️  ATH Gearbox      29m 50s  Epoch 7930: Shifting UP: 5
    ⚙️  ATH Gearbox      30m 54s  Epoch 8182: Shifting DOWN: 4
    ⚙️  ATH Gearbox      31m 56s  Epoch 8383: Shifting UP: 5
    ⚙️  ATH Gearbox      35m 24s  Epoch 9185: Shifting DOWN: 4
    ⚙️  ATH Gearbox      36m 33s  Epoch 9386: Shifting UP: 5

    🏆 Highscore Events
    ══════════════════
    Epoch  Highscore  Time     Epsilon
    ═════  ═════════  ═══════  ═══════
    2      1          0s       0.9990 
    90     2          2s       0.5655 
    102    3          2s       0.4914 
    198    4          5s       0.2648 
    224    5          6s       0.2115 
    247    6          7s       0.1812 
    263    7          8s       0.1654 
    297    8          10s      0.1276 
    322    9          11s      0.1093 
    322    10         11s      0.1093 
    322    11         11s      0.1093 
    380    12         15s      0.0707 
    380    13         15s      0.0707 
    380    14         15s      0.0707 
    380    15         15s      0.0707 
    476    16         22s      0.0360 
    501    17         25s      0.0304 
    501    18         25s      0.0304 
    558    19         31s      0.0200 
    558    20         31s      0.0200 
    558    21         31s      0.0200 
    558    22         31s      0.0200 
    558    23         31s      0.0200 
    605    24         37s      0.0143 
    605    25         37s      0.0143 
    605    26         37s      0.0143 
    605    27         37s      0.0143 
    1003   28          1m 28s  0.0009 
    1127   29          1m 42s  0.0004 
    1265   30          1m 59s  0.0001 
    1323   31          2m  8s  0.0001 
    1628   32          2m 58s  0.0000 
    1628   33          2m 58s  0.0000 
    1672   34          3m  7s  0.0000 
    1672   35          3m  7s  0.0000 
    1672   36          3m  7s  0.0000 
    1672   37          3m  7s  0.0000 
    1672   38          3m  7s  0.0000 
    1672   39          3m  7s  0.0000 
    1720   40          3m 18s  0.0000 
    1720   41          3m 18s  0.0000 
    1866   42          3m 48s  0.0000 
    1866   43          3m 48s  0.0000 
    1866   44          3m 48s  0.0000 
    1866   45          3m 48s  0.0000 
    1866   46          3m 49s  0.0000 
    1866   47          3m 49s  0.0000 
    1866   48          3m 49s  0.0000 
    1866   49          3m 49s  0.0000 
    1866   50          3m 49s  0.0000 
    1866   51          3m 49s  0.0000 
    1866   52          3m 49s  0.0000 
    4465   53         14m 52s  0.0000 
    4465   54         14m 52s  0.0000 
    8324   55         31m 37s  0.0000 
    8324   56         31m 38s  0.0000 
    8327   57         31m 38s  0.0000 
    8327   58         31m 38s  0.0000 
    9289   59         36m  1s  0.0000 
    9289   60         36m  1s  0.0000 
    9352   61         36m 22s  0.0000 
    9352   62         36m 22s  0.0000 

    ⚙️  ATH Shift / Mean / Median
    ═════════════════════════════
    Epoch  Gear  Seq Length  Batch Size  Mean   Median
    ═════  ════  ══════════  ══════════  ═════  ══════
    500    3     16          16          3.07   2.00  
    1000   3     16          16          5.36   3.00  
    1500   4     24          10          7.13   5.00  
    2000   5     32          8           9.51   7.00  
    2500   5     32          8           11.32  9.00  
    3000   5     32          8           13.19  11.00 
    3500   4     24          10          14.06  12.00 
    4000   5     32          8           14.92  13.00 
    4500   5     32          8           15.75  14.00 
    5000   5     32          8           16.36  15.00 
    5500   5     32          8           16.97  16.00 
    6000   5     32          8           17.42  17.00 
    6500   5     32          8           17.80  17.00 
    7000   4     24          10          18.08  18.00 
    7500   4     24          10          18.39  18.00 
    8000   5     32          8           18.78  19.00 
    8500   5     32          8           19.04  19.00 
    9000   5     32          8           19.30  19.00 
    9500   5     32          8           19.60  19.00 
    10000  5     32          8           19.81  20.00 
    10500  5     32          8           20.04  20.00 
    11000  5     32          8           20.25  20.00 
    11500  5     32          8           20.43  20.00 

    🪣 ATH Memory Bucket Usage
    ═════════════════════════
    Epoch  b1   b2   b3   b4   b5   b6   b7   b8   b9   b10  b11  b12  b13  b14  b15  b16  b17  b18  b19  b20
    ═════  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══  ═══
    500    476  373  294  232  190  152  120  92   71   54   44   40   27   21   13   11   7    5    4    4  
    1000   808  668  580  504  451  398  342  299  257  213  188  173  146  123  100  83   65   56   46   40 
    1500   505  464  421  372  332  288  254  221  188  163  136  113  90   73   62   53   42   31   22   16 
    2000   354  340  311  283  259  238  208  184  157  131  104  91   74   56   41   34   22   16   15   12 
    2500   314  302  291  279  261  241  215  191  168  139  116  98   85   72   56   42   33   23   16   11 
    3000   282  279  274  257  248  228  207  187  173  156  137  121  103  84   65   53   41   31   23   14 
    3500   337  329  321  308  295  274  256  229  210  192  176  163  141  125  105  92   81   67   56   43 
    4000   313  305  291  279  257  228  210  191  171  152  128  110  85   73   54   40   30   19   12   8  
    4500   294  285  278  257  241  225  206  192  172  151  130  120  99   81   64   53   42   31   24   14 
    5000   304  303  290  272  259  238  211  197  177  145  124  106  86   64   50   39   28   23   19   13 
    5500   297  290  283  273  255  232  215  194  181  151  123  102  86   66   57   43   35   26   19   16 
    6000   292  289  284  267  254  233  214  191  175  150  131  112  83   68   56   51   40   31   23   14 
    6500   296  296  291  279  266  247  223  201  180  157  128  103  83   67   53   39   27   20   14   10 
    7000   304  301  297  296  288  283  275  263  252  227  200  178  152  130  109  95   78   62   45   36 
    7500   291  289  287  284  280  270  256  242  221  206  193  178  159  141  125  111  94   86   72   54 
    8000   265  263  259  248  241  229  217  203  185  165  137  117  91   73   62   49   40   33   27   20 
    8500   280  278  271  261  248  229  219  201  183  157  132  117  91   71   62   48   37   25   20   14 
    9000   282  272  266  260  249  227  214  197  176  159  137  116  96   78   63   49   32   24   21   14 
    9500   248  243  238  233  227  214  205  192  184  166  153  132  112  103  84   67   52   42   32   22 
    10000  259  250  241  235  227  214  207  195  183  160  142  124  108  96   80   68   58   42   33   23 
    10500  278  267  256  251  243  233  216  203  183  153  138  117  102  90   76   62   43   28   20   13 
    11000  260  256  247  243  236  226  212  187  174  154  142  129  116  99   82   65   50   38   28   18 
    11500  287  280  267  257  240  227  207  187  169  152  133  117  100  80   62   52   38   29   23   19

----

Closing Thoughts
****************

AI Hydra isn’t just running a simulation.

It exposes the **mechanics of learning itself** — in real time, with structure, 
visibility, and control. Its high-performance architecture enables users to do 
**extremely fast iterations** and the visualizations enable **real-time analysis** 
that is augmented by the **Snapshot Reports.**