---
title: "Performance"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

AI Hydra is designed for **high-throughput training on consumer hardware**,
without relying on a GPU.

Its performance comes from two complementary design pillars:

## 1. Aligned Training Pipeline

The training path is tightly aligned across memory, batching, and model execution.

- Replay memory stores full episodes and exposes **variable-length sequences via metadata**
- Sequences are constructed **without expensive transformation or reshaping**
- Batches are fed directly into the model’s **sequence-forward path**

This creates a continuous pipeline:

**memory → sequence assembly → batched training → model forward**

Because each stage is designed to match the next, the system avoids unnecessary
copying, slicing, or recomputation.

## 2. Lean Simulation Loop

The core run loop is intentionally minimal and stays focused on simulation and training only.

- No rendering or plotting occurs in the training process
- Telemetry is published asynchronously via a **ZeroMQ PUB socket**
- Visualization, plotting, and event correlation are handled by a separate client (TUI)

This keeps the hot path free of UI and analysis overhead, allowing the simulation to run at full speed.

## Result

- High episode throughput on CPU-only systems
- Stable real-time telemetry without blocking the simulation
- Fast iteration cycles with immediate observability

AI Hydra achieves speed not through hardware, but through **alignment and separation of concerns**.
The system is optimized so that the cost of learning is dominated by model
computation, not data movement or orchestration.
