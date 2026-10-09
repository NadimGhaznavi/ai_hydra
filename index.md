---
title: "AI Hydra: Hybrid Neural Network + Tree Search System"
author_profile: true
layout: single
---

![HydraClient TUI]({{ '/images/hydra-client.png' | relative_url }})

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

---

## Project Documents

- [Getting Started]({{ site.baseurl }}{% link pages/getting-started.md %})
- [Distributed Architecture]({{ site.baseurl }}{% link pages/architecture.md %})
- [Settings]({{ site.baseurl }}{% link pages/settings.md %})
- [Models and Reproducibility]({{ site.baseurl }}{% link pages/models-and-reproducibility.md %})
- [Visualizations]({{ site.baseurl }}{% link pages/visualizations.md %})
- [ATH Replay Memory]({{ site.baseurl }}{% link pages/ath-replay-memory.md %})
- [Train Manager]({{ site.baseurl }}{% link pages/train-manager.md %})
- [Epsilon Nice — Bounded Safe-Detour Policy]({{ site.baseurl }}{% link pages/epsilon-nice.md %})
- [Performance]({{ site.baseurl }}{% link pages/performance.md %})
- [Snapshot Report]({{ site.baseurl }}{% link pages/snapshot-report.md %})

---

## Closing Thoughts

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
