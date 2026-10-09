---
title: "Visualizations"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

The TUI provides real-time insight into training:

- Live Snake board
- High score tracking
- Score plots

Views include:

## High Scores, Current and Average Current Scores

![Scores]({{ '/images/scores.png' | relative_url }})

## Loss

- Full training loss over time
- Sliding window (recent 75 episodes)

![Loss Plot]({{ '/images/loss-plot.png' | relative_url }})

## Scores Distribution

- Global score histogram
- Recent performance window (last 500 episodes)

![Score Distribution]({{ '/images/scores-histogram.png' | relative_url }})

## Event Log

A structured event stream capturing system behavior:

- Simulation lifecycle
- Replay memory transitions
- Gear shifts and thresholds

![Event Log]({{ '/images/event-log.png' | relative_url }})
