---
title: "ATH Replay Memory"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

The *ATH Replay Memory* is a core component of *AI Hydra*.

It sits between the simulation and the *Trainer*, and controls how
experience is stored, organized, and sampled for learning. It is a key
driver of emergent behavior.

A key takeaway is that the AI can learn new policies even when it is
trained on the same underlying data, simply by changing how that data
is sampled.

In particular, reducing the sequence length can expose patterns that
are not learned when using longer sequences. Evidence suggests that some
behaviors are easier to learn from shorter temporal windows, and are
diluted or obscured when the sequence length is too long.

In effect, changing the sequence length changes the temporal resolution
at which the agent learns.

## ATH Overview

- Games are stored as ordered sequences of frames.
- At the end of each episode, the  *Trainer* asks the *ATH Replay Memory* for training data.
- The *ATH Memory* returns a batch of ordered frames.
  - This *ordered set of frames* is referred to as a *sequence*.
  - The *number of ordered frames* is referred to as the *sequence length*.
  - The number of *sequences* that are returned is the *batch size*

## ATH Gears

At the core of *ATH Memory* is the concept of a *gear*.

- When the simulation starts, the initial gear is one.
  - When the *ATH Memory* is in *first gear* (gear one), the *sequence length* is 4.
  - Gear 2 has a sequence length of 6.
  - Gear 3 has a sequence length of 8.
  - Every shift to a higher gear, increases the sequence length by two.

## Batch Size

One of the ATH Memory settings is the *Max Training Frames* parameter.

The batch size is determined by this simple calculation:

  `batch_size = max_training_frames // sequence_length`

The minimum size of a batch is one.

## Buckets

- Assume that *ATH Memory* is in first gear
  - This means the sequence length is 4
- If the game has 17 frames, then:
  - It will contain 4 sequences, each containing 4 frames
  - The first frame will be discarded
- The *ATH Memory* will put this game into bucket 1, 2, 3, and 4

When the system shifts to the second gear, the data is reorganized using
the new sequence length of 6:

- Now the 17 frames will be split into 2 sequences
  - Each sequence contains 6 frames
  - The first 5 frames will be discarded
- The *ATH Memory* will now put the same game into bucket 1 and 2

**Warmed Buckets**

A bucket is defined as being *warm* when it contains at least one game.

## Selecting Training Data

When the *Trainer* asks the *ATH Memory* for training data, the goal is to
construct a batch of sequences that all have the same length.

- The *ATH Memory* creates a batch using games from **all active buckets**
- A batch will contain sequences from:
  - early-stage gameplay
  - mid-game gameplay
  - late-game gameplay

**Key Concept:** *Sampling occurs across time.*

In a traditional replay memory, sequences from the ends of long-running
games are underrepresented.

## Gearbox Shifting

The *ATH Replay Memory* contains an *ATH Gearbox* module that is responsible
for changing gears. Its behavior depends on the following settings:

- Upshift threshold
- Downshift threshold
- Cooldown threshold

- If the **last three buckets combined** contain at least *upshift threshold*
  games, the *ATH Gearbox* will shift up

- If the **last three buckets combined** contain fewer than *downshift threshold*
  games, the *ATH Gearbox* will shift down

- The gearbox will not perform any shifting until *cooldown threshold*
  episodes have elapsed

## Stagnation Shifting

The *Train Manager* also plays a role in gear shifting. Its behavior is
determined by the following *stagnation* settings:

- Stagnation threshold
- Critical stagnation threshold

- If *stagnation threshold* episodes elapse without a new high score, the
  *Train Manager* sends a *stagnation alert* to the *ATH Gearbox*

  - The gearbox will shift down one gear, unless it has shifted up within
    the last *stagnation threshold* episodes

- If *critical stagnation threshold* episodes elapse without a new high score,
  the *Train Manager* sends a *critical stagnation alert* to the *ATH Gearbox*

  - The gearbox will shift directly to gear 1 (sequence length 4)

## ATH Memory Telemetry

The TUI memory widget shows the number of games each bucket contains. Note
that this number changes when the gear changes.

![ATH Memory]({{ '/images/ath-memory.png' | relative_url }})

Lifecycle events such as warm-up, pruning, and gear shifts are emitted and
tracked in real time and can be seen in the *Event Log* tab of the TUI.
