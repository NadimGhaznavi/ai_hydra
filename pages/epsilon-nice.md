---
title: "Epsilon Nice — Bounded Safe-Detour Policy"
author_profile: true
layout: single
---

[Documentation index]({{ site.baseurl }}{% link index.md %})

## Overview

**Epsilon Nice** is a conditionally enabled policy wrapper that can temporarily
redirect action selection into a short safe-detour mode.

It is not a one-step correction and it is not always active.

Instead, once normal epsilon exploration has effectively ended, Epsilon Nice
may be enabled by the training system. When enabled, it can probabilistically
arm a temporary intervention window. During that window, action selection is
redirected toward safe alternatives for a fixed number of steps.

This allows the agent to break out of locally bad or stagnant behavior without
permanently replacing the underlying learned policy.

## How It Works

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

## Override Behavior

During an active Nice detour, the policy looks at the other available actions
and gathers the non-colliding alternatives.

- If safe alternatives exist, one is selected at random
- If no safe alternative exists, the original action is preserved

This means Nice does not simply “block fatal moves.”
It temporarily moves the policy into a safe-alternative action mode for a
bounded number of steps.

## Key Properties

- **Post-epsilon gated** - Nice does nothing while normal epsilon exploration is still active
- **Externally controlled** - Enabled and disabled by the training system
- **Probabilistic arming** - `p_value` controls entry into Nice mode
- **Multi-step intervention** - `steps` defines the length of the detour
- **Safe-alternative sampling** - Overrides choose from non-colliding alternatives when available
- **Deterministic-friendly** - Uses the shared RNG
- **Observable** - Reports rolling stats such as calls, triggers, overrides, fatal suggestions, and no-safe-alternative cases

## Telemetry

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
