---
title: "Train Manager"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

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

Additionally, when a new high score is achieved the *Train Manager* calls
`reset_cooldown()` on the *ATH Gearbox* thereby delaying an upshift to allow the AI to fully exploit the opportunities at the current sequence length and batch size.

This makes stagnation a first-class runtime signal rather than a passive metric.

The result is a coordinated feedback system where memory dynamics and policy
behavior can both adapt when learning plateaus.
