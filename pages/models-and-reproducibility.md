---
title: "Models and Reproducibility"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

## Deterministic Simulations

Simulations are **fully deterministic.**

Given the same configuration and random seed:

- Episode progression is identical
- Scores occur at the same episodes
- Pausing/resuming does not affect outcomes

This enables reliable experimentation and reproducibility.

You can also vary only the random seed to validate that results are not artifacts of a *“lucky run.”*

---

## Supported Models: Linear, RNN, and GRU

- Linear Model
- Recurrent Neural Network (RNN)
- Gated Recurrent Network (GRU)

Model selection is available directly in the TUI.
