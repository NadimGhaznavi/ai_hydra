---
Title: Brainstorming on Architecture
---

# 🧩 The MBRL Agent Paradigm (Conceptually)

Most MBRL agents — even fancy ones — decompose into the same four blocks:

## World Model

A learned dynamics model:

![Formula for learned dynamics model](/images/learned-dynamics-model-formula.png)

This can be:

- deterministic (simple NN)
- stochastic (RSSM, latent state models)
- or completely handcrafted for classical control

## Planner / Controller

Uses the world model to search, optimize, or roll out imagined trajectories:

- CEM
- MPPI
- Random Shooting
- Tree search
- Value expansion

## Policy / Actor (optional)

The agent may also distill planning results into a reactive policy.

## Memory / Replay

The reservoir of experience for continuously training the world model and the policy.

Everything else is glue.

---

# 🧠 How This Fits into the Hydra-Zen Skeleton

Hydra-Zen lets you express these 4 MBRL components as *modular config groups* — essentially giving you a parametric agent factory.

The skeleton already uses `just()` + `store()`.

Your high-level Hydra groups might evolve into:

```shell
mb_agent/
    model/          # world model configs
    planner/        # CEM, RS, MPPI, etc
    policy/         # optional
    buffer/         # replay setup
    env/            # snake, gym envs
    train/          # trainer loop configs
```

Then your entrypoint might select:

```python
hydra_defaults = [
    "mb_agent/model:tiny_mlp",
    "mb_agent/planner:cem",
    "mb_agent/env:snake",
    "mb_agent/buffer:basic",
]
```

The key realization:

**The Hydra skeleton becomes a declarative agent factory.**

Every MBRL experiment is just a config combination.

---

# 🧱 Porting the AI Snake Lab — Safest Order

The Sim-Server (environment-side) is exactly the right first migration target.

Porting sequence:

## 1. Environment → Sim-Server layer

Hydra group: `env/snake.py`

Encapsulate:

- observation shape
- action space
- reset/step
- render/speed options
- reward shaping knobs

This yields a clean foundation.

## 2. Memory / Replay

Hydra group: `buffer/basic.py`

Start minimal:

- a list of (s, a, r, s') tuples
- maybe a ring buffer

Keep it stupid-simple early.

## 3. World Model

Hydra group: `model/*`

Start with:

- deterministic forward model
- small MLP

Later add latent states / RSSM-like structure.

## 4. Planner

Hydra group: `planner/*`

Begin with:

- random-shooting or CEM
- extremely small horizon (5–15 steps)

This lets you “close the loop” and get the agent acting.

## 5. Policy (optional)

Once you have:

- world model training
- planning working end-to-end

Can distill the planner into a policy if desired.

---

# 🔄 How to Structure the Codeflow in Your Skeleton

```
Agent:
    world_model.step(...)
    planner.plan(...)
    policy.act(...)

Trainer:
    collect_rollout(...)
    update_world_model(...)
    optional: distill_planner(...)
```

And each component’s implementation is determined by a Hydra config.
This gives you a powerful but simple way to swap algorithms.

---

I am also keeping this R. Sutton paper in mind: [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://arxiv.org/pdf/2202.03466), since I believe that is the key to creating a truly novel RL implementation. So that's another piece of the puzzle that needs to be fit into my Hydra-Zen project.

# 🧠 What Sutton’s RRS Paper Is About (in practical terms)

The key idea is:

**Break the main task into subtasks whose reward structures are simpler, reusable, and compositional.**

But crucially:

- Subtasks share transition dynamics
- Subtasks differ only in the reward signal
- Subtasks define abstract transitions that compress the environment’s dynamics
- These abstractions can be used with MB planning, search, or Bellman updates

From Sutton’s viewpoint:

If two states differ only in irrelevant detail w*ith respect to a subtask reward*, treat them as equivalent.

This enables:

- Generalizing faster
- Planning over reduced models
- Composing solutions across tasks

And importantly for your Snake project:

**RRS essentially defines a secondary modeling layer on top of your learned dynamics model.**

---

# 🧩 Where RRS fits in the MBRL architecture

Let’s integrate it into the standard MBRL blocks:

```
World Model (transition dynamics)
Reward Model(s)
Subtask Structure (RRS)
Planner / Controller
Policy (optional)
Memory / Replay
```

The new element is subtask reward models and subtask abstractions.

**🟦 The world model stays the same**

The transition model predicts:

- next state
- reward (task reward or subtask reward)
- maybe termination

**🟨 RRS adds a new module:**

A **subtask** abstraction module that:

- defines a set of subtasks
- defines reward-respecting equivalences
- transforms environment states into abstracted states for each subtask
- provides subtask-specific reward functions
- can generate “macro-actions” or “abstract transitions” for your planner

RRS can be thought of as:

**A transformation layer between raw environment state ↔ the planner.**

---

# 🏗️ Mapping RRS into the Hydra-Zen skeleton

This part is beautifully clean.

Create a new Hydra config group:

```shell
mb_agent/
    rrs/
        none.yaml
        snake_food.yaml
        snake_hazard.yaml
        general_hierarchy.yaml
        combined.yaml
```

Your agent’s defaults might now include:

```python
hydra_defaults = [
    "mb_agent/env:snake",
    "mb_agent/model:deterministic",
    "mb_agent/planner:cem",
    "mb_agent/rrs:snake_food",
]
```

Then in Python:

**A possible RRS module API:**

```python
class RRSModule:
    def __init__(self, subtask_defs):
        ...

    def abstract_state(self, s):
        # Return abstract features or equivalence class
        ...

    def reward(self, s, a, s_next):
        # Subtask-specific reward
        ...

    def options(self):
        # (Optional) Generate subtask-level options / macro-actions
        return []
```

The planner modifies:

```python
if rrs is not None:
    s_planner = rrs.abstract_state(s)
else:
    s_planner = s
```

The world model may receive:

- task reward loss
- OR subtask reward losses
- OR both

All clean, all modular, all driven by Hydra configs.

---

🪜 How RRS integrates into MBRL rollouts in practice

A standard MBRL imagined rollout:

```
s0 → a0 → s1 → a1 → ...  
```

Becomes:

```
z0 = RRS.abstract(s0)
z1 = RRS.abstract(s1)
reward = RRS.reward(s0, a0, s1)
```

The planner plans in z-space, not raw state space.

In Snake, this can let you define subtasks like:

- subtask: go toward food
- subtask: avoid hazard
- subtask: maintain safe corridor
- subtask: escape self-trap
- subtask: reach center
- subtask: extend tail loop

Each with its own reward structure and equivalence classes.

---
# 🎯 RRS is a great fit for a “novel RL implementation”

**RRS is exactly this kind of novelty:**

- It’s philosophically deep (coming from Sutton).
- Almost no practical implementations exist.
- It synergizes perfectly with planning-based MBRL.
- It enables hierarchical, compositional, interpretable planning.
- It fits your Snake domain extremely well (Snake is highly subtaskable).
- And Hydra-Zen makes it easy to experiment with different subtask definitions.

Different RRS decompositions strategies can be implemented by simply swapping configs — this is exactly what Hydra is for.

---

# 🌌 1. What’s the actual structure of the problem?

Not the implementation — *the problem.*

For Snake specifically, you can see:

- stable loops
- local optima strategies
- survival bias
- boundary-frequency patterns
- how the agent implicitly negotiates space
- switching dynamics between exploration modes

Even visually, you can almost see the agent’s internal model reaching diminishing returns.

Those patterns are the seeds of your RRS ideas later.

---

# 🧩 2. What parts of the behaviour feel like subtasks?

This is the high-level creative part that algorithms can’t automate.

When you watch the simulation long enough, you start to see:

- phases
- modes
- “moves”
- macro-structures

It’s incredibly aligned with a “reward-respecting subtask” framework.

At this altitude, you can freely think:

- “Is the environment actually composed of a few invariant survival motifs?”
- “What abstractions would collapse these states into equivalence classes?”

This is exactly the right thinking mode for RRS.