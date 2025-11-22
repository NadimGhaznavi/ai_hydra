# hydra_server.py


# NOTE:
#  - store(group="X") registers config objects accessible as `X/<name>`
#  - hydra_defaults should select names from groups you registered
#  - use `just()` to store callables (activation fns, planners, etc.)
#  - zen(fn).hydra_main(config_name="NAME") chooses which stored entrypoint NAME to run

from hydra_zen import just, store

from env.Env import Env
from policy.Policy import Policy
from model.Model import Model
from misc.DataLoader import DataLoader


# Minimal placeholder activation functions.
# They are intentionally trivial here — in real code they would be actual functions.
def relu(x): ...
def sigmoid(x): ...


def simulation_loop(env: Env, policy: Policy, num_steps: int):
    # The function we intend to run for simulation experiments.
    # Note how the arguments are typed with classes defined above — hydra-zen
    # will construct instances of Env/Policy when this function is run via the CLI.
    print("=== Simulation Loop ===")
    print(env.__dict__)
    print(policy.__dict__)
    print(num_steps)


def train_fn(model: Model, dataloader: DataLoader, num_epochs: int = -1):
    # The training function used as an alternate entrypoint to the simulation_loop.
    # hydra-zen will construct `Model` and `DataLoader` instances and pass them in.
    print("=== Training Loop ===")
    print(f"Training with {num_epochs=}\n")
    print(model.summary)
    print(dataloader.summary)


# -------------------------
# Hydra-Zen CONFIG STORES
# -------------------------

# model_store: register model configurations under the `model` group.
# Each call to model_store(...) registers a different named config that can be selected
# via the CLI (e.g., model=big).
model_store = store(group="model")
model_store(Model, name="generic")
model_store(Model, nlayers=100, name="big")
model_store(Model, nlayers=2, name="tiny")

# activation_store: store activation functions under the nested group model/activation.
# Using `just(relu)` means "store a reference to the relu function" rather than
# instantiating it.
activation_store = store(group="model/activation")
activation_store(just(relu), name="relu")
activation_store(just(sigmoid), name="sigmoid")

# dataloader configs: stored under group `dataloader`
data_store = store(group="dataloader")
data_store(DataLoader, name="train")
data_store(DataLoader, shuffle_batch=False, name="test")

# -------------------------
# Entrypoint: train_fn with defaults
# -------------------------

# This `store(...)` call registers an entrypoint named "train".
# hydra_defaults is the list of default selections added to the final composed config
# when the "train" entry is used.
store(
    train_fn,
    hydra_defaults=[
        "_self_",
        {"model": "big"},
        {"model/activation": "relu"},
        {"dataloader": "train"},
    ],
    name="train",
)

# -------------------------
# Simulation configs (env & policy)
# -------------------------

# env_store: register environment variants under group `sim/env`
env_store = store(group="env")
env_store(Env, size=20, light_decay=0.95, name="default")
env_store(Env, size=40, light_decay=0.90, name="large")

# policy_store: register policy variants under group `sim/policy`
policy_store = store(group="policy")
policy_store(Policy, hidden=128, name="mlp")
policy_store(Policy, hidden=128, reactive=just("wall_avoidance"), name="mlp_reactive")
# note: reactive is stored as a raw value here (a string) for demonstration;
# in production, reactive might be just(some_callable) or a dataclass ref.

# -------------------------
# Entrypoint: simulation_loop with defaults
# -------------------------

# Register the simulation_loop function as a separate entrypoint named "sim".
# Its hydra_defaults pick sim/env and sim/policy defaults *without* a "sim/" prefix,
# because the function parameters are `env` and `policy`. Hydra will map the
# `env` parameter to the group named `env` unless you stored it under `sim/env`,
# so note the group mismatch here (see quick reference below).
store(
    simulation_loop,
    hydra_defaults=[
        "_self_",
        {"env": "default"},
        {"policy": "mlp"},
    ],
    name="sim",
)

# -------------------------
# Script Main: choose which entrypoint to expose
# -------------------------
if __name__ == "__main__":
    from hydra_zen import zen

    # push everything registered into the Hydra store so hydra can find it.
    store.add_to_hydra_store()

    # Generate the CLI wrapper for train_fn and run it.
    # This line chooses the `train` entrypoint stored earlier (config_name="train").
    # To run the sim entrypoint instead, you would call:
    #   zen(simulation_loop).hydra_main(config_name="sim", ...)
    zen(train_fn).hydra_main(
        config_name="train",
        config_path=None,
        version_base="1.3",
    )
    # When executed, hydra_zen will:
    # 1. compose the final config from the selected defaults and any CLI overrides
    # 2. instantiate objects defined by the chosen configs (Model, DataLoader, Env, Policy, etc.)
    # 3. call train_fn(model, dataloader, ...) with the instantiated objects
