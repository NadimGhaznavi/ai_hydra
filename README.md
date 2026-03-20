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
This is due to careful architectural design decisions. The *Replay Memory*,
*Model*, and *Trainer* work in a pipline, minimizing data transformations.

