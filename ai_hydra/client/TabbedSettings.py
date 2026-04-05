# ai_hydra/client/TabbedSettings.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations
import time

from textual import on
from textual.app import ComposeResult, Widget
from textual.widgets import TabbedContent, Label, Input, Select
from textual.containers import Horizontal, Vertical
from textual.validation import Number

from ai_hydra.constants.DNNet import DLinear, DRNN, MODEL_TYPES
from ai_hydra.constants.DHydraTui import DLabel, DField, DStatus
from ai_hydra.constants.DReplayMemory import DMemDef
from ai_hydra.constants.DGame import DGameDef


class TabbedSettings(Widget):
    """A TabbedContent widget to house simulation settings"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Network configuration
        self.router_address = Label(id=DField.ROUTER_ADDR)
        self.router_port = Label(id=DField.ROUTER_PORT)
        self.router_hb_port = Label(id=DField.ROUTER_HB_PORT)
        self.router_hb_status = Label(id=DField.ROUTER_HB_STATUS)
        self.server_address = Label(id=DField.SERVER_ADDR)
        self.server_pub_port = Label(id=DField.SERVER_PUB_PORT)
        self.server_status = Label(
            f"{DStatus.UNKNOWN}", id=DField.SERVER_STATUS
        )

        # ----- ATH Replay memory ---
        # MAX_FRAMES
        self.max_frames_label = Label(id=DField.MAX_FRAMES_LABEL)
        self.max_frames_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MAX_FRAMES),
            id=DField.MAX_FRAMES_INPUT,
        )
        # MAX_TRAINING_FRAMES
        self.max_training_frames_label = Label(
            id=DField.MAX_TRAINING_FRAMES_LABEL,
        )
        self.max_training_frames_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MAX_TRAINING_FRAMES),
            id=DField.MAX_TRAINING_FRAMES_INPUT,
        )
        # MAX_BUCKETS
        self.max_buckets_label = Label(id=DField.MAX_BUCKETS_LABEL)
        self.max_buckets_input = Input(
            type=DField.INTEGER,
            validators=[Number(minimum=3)],
            compact=True,
            value=str(DMemDef.MAX_BUCKETS),
            id=DField.MAX_BUCKETS_INPUT,
        )
        # MAX_GEAR
        self.max_gear_label = Label(id=DField.MAX_GEAR_LABEL)
        self.max_gear_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MAX_GEAR),
            id=DField.MAX_GEAR_INPUT,
        )
        # NUM_COOLDOWN_EPISODES
        self.num_cooldown_eps_label = Label(
            id=DField.NUM_COOLDOWN_EPISODES_LABEL
        )
        self.num_cooldown_eps_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.NUM_COOLDOWN_EPISODES),
            id=DField.NUM_COOLDOWN_EPISODES_INPUT,
        )
        # UPSHIFT_COUNT_THRESHOLD
        self.upshift_count_threshold_label = Label(
            id=DField.UPSHIFT_COUNT_THRESHOLD_LABEL,
        )
        self.upshift_count_threshold_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.UPSHIFT_COUNT_THRESHOLD),
            id=DField.UPSHIFT_COUNT_THRESHOLD_INPUT,
        )
        # DOWNSHIFT_COUNT_THRESHOLD
        self.downshift_count_threshold_label = Label(
            id=DField.DOWNSHIFT_COUNT_THRESHOLD_LABEL,
        )
        self.downshift_count_threshold_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.DOWNSHIFT_COUNT_THRESHOLD),
            id=DField.DOWNSHIFT_COUNT_THRESHOLD_INPUT,
        )
        # MAX_STAGNANT_EPISODES
        self.max_stag_eps_label = Label(
            id=DField.MAX_STAGNANT_EPISODES_LABEL,
        )
        self.max_stag_eps_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MAX_STAGNANT_EPISODES),
            id=DField.MAX_STAGNANT_EPISODES_INPUT,
        )
        # MAX_HARD_RESET_EPISODES
        self.max_crit_stag_eps_label = Label(
            id=DField.MAX_CRIT_STAGNANT_EPISODES_LABEL
        )
        self.max_crit_stag_eps_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MAX_HARD_RESET_EPISODES),
            id=DField.MAX_CRIT_STAGNANT_EPISODES_INPUT,
        )

        # NICE_P_VALUE
        self.nice_p_value_label = Label(id=DField.NICE_P_VALUE_LABEL)
        self.nice_p_value_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DLinear.NICE_P_VALUE),
            id=DField.NICE_P_VALUE_INPUT,
        )
        # NICE_STEPS
        self.nice_steps_label = Label(id=DField.NICE_STEPS_LABEL)
        self.nice_steps_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.NICE_STEPS),
            id=DField.NICE_STEPS_INPUT,
        )
        # MCTS_DEPTH
        self.mcts_depth_label = Label(id=DField.MCTS_DEPTH_LABEL)
        self.mcts_depth_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MCTS_DEPTH),
            id=DField.MCTS_DEPTH_INPUT,
        )
        # MCTS_SEARCH_ITERATIONS
        self.mcts_iter_label = Label(id=DField.MCTS_ITER_LABEL)
        self.mcts_iter_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MCTS_ITER),
            id=DField.MCTS_ITER_INPUT,
        )
        # MCTS_EXPLORE_P_VALUE
        self.mcts_explore_p_value_label = Label(
            id=DField.MCTS_EXPLORE_P_VALUE_LABEL
        )
        self.mcts_explore_p_value_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DLinear.MCTS_EXPLORE_P_VALUE),
            id=DField.MCTS_EXPLORE_P_VALUE_INPUT,
        )
        # MCTS_GATE_P_VALUE
        self.mcts_gate_p_value_label = Label(id=DField.MCTS_GATE_P_VALUE_LABEL)
        self.mcts_gate_p_value_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DLinear.MCTS_GATE_P_VALUE),
            id=DField.MCTS_GATE_P_VALUE_INPUT,
        )
        # MCTS_STEPS
        self.mcts_steps_label = Label(id=DField.MCTS_STEPS_LABEL)
        self.mcts_steps_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MCTS_STEPS),
            id=DField.MCTS_STEPS_INPUT,
        )
        # MCTS_SCORE_THRESH
        self.mcts_score_thresh_label = Label(id=DField.MCTS_SCORE_THRESH_LABEL)
        self.mcts_score_thresh_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DLinear.MCTS_SCORE_THRESH),
            id=DField.MCTS_SCORE_THRESH_INPUT,
        )

        # FOOD_REWARD
        self.food_rewards_label = Label(id=DField.FOOD_REWARD_LABEL)
        self.food_rewards_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DGameDef.FOOD_REWARD),
            id=DField.FOOD_REWARD_INPUT,
        )
        # COLLISION_PENALTY
        self.collision_penalty_label = Label(id=DField.COLLISION_PENALTY_LABEL)
        self.collision_penalty_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DGameDef.COLLISION_PENALTY),
            id=DField.COLLISION_PENALTY_INPUT,
        )
        # MAX_MOVES_PENALTY
        self.max_moves_penalty_label = Label(id=DField.MAX_MOVES_PENALTY_LABEL)
        self.max_moves_penalty_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DGameDef.MAX_MOVES_PENALTY),
            id=DField.MAX_MOVES_PENALTY_INPUT,
        )
        # EMPTY_MOVE_REWARD
        self.empty_move_reward_label = Label(id=DField.EMPTY_MOVE_REWARD_LABEL)
        self.empty_move_reward_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DGameDef.EMPTY_MOVE_REWARD),
            id=DField.EMPTY_MOVE_REWARD_INPUT,
        )
        # CLOSER_TO_FOOD
        self.closer_to_food_label = Label(id=DField.CLOSER_TO_FOOD_LABEL)
        self.closer_to_food_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DGameDef.CLOSER_TO_FOOD),
            id=DField.CLOSER_TO_FOOD_INPUT,
        )
        # FURTHER_FROM_FOOD
        self.further_from_food_label = Label(id=DField.FURTHER_FROM_FOOD_LABEL)
        self.further_from_food_input = Input(
            type=DField.NUMBER,
            compact=True,
            value=str(DGameDef.FURTHER_FROM_FOOD),
            id=DField.FURTHER_FROM_FOOD_INPUT,
        )
        # MAX_MOVES_MULTIPLIER
        self.max_moves_multiplier_label = Label(
            id=DField.MAX_MOVES_MULTIPLIER_LABEL
        )
        self.max_moves_multiplier_input = Input(
            type=DField.INTEGER,
            compact=True,
            value=str(DGameDef.MAX_MOVES_MULTIPLIER),
            id=DField.MAX_MOVES_MULTIPLIER_INPUT,
        )

        # Current epsilon
        self.cur_epsilon = Label(id=DField.CUR_EPSILON)

    def compose(self) -> ComposeResult:
        with TabbedContent(
            DLabel.CONFIG,
            DLabel.MEMORY,
            DLabel.REWARDS,
            DLabel.POLICY,
            DLabel.NETWORK,
        ):

            # ----- Config Tab ---
            yield Vertical(
                # ----- Epsilon ---
                Vertical(
                    # Initial epsilon
                    Horizontal(
                        Label(f"{DLabel.INITIAL_EPSILON:>15s} : "),
                        Label(
                            str(DLinear.INITIAL_EPSILON),
                            id=DField.INITIAL_EPSILON_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=str(DLinear.INITIAL_EPSILON),
                            id=DField.INITIAL_EPSILON_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Minimum epsilon
                    Horizontal(
                        Label(f"{DLabel.MIN_EPSILON:>15s} : "),
                        Label(
                            str(DLinear.MINIMUM_EPSILON),
                            id=DField.MIN_EPSILON_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=str(DLinear.MINIMUM_EPSILON),
                            id=DField.MIN_EPSILON_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Epsilon decay
                    Horizontal(
                        Label(f"{DLabel.EPSILON_DECAY:>15s} : "),
                        Label(
                            str(DLinear.EPSILON_DECAY_RATE),
                            id=DField.EPSILON_DECAY_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=str(DLinear.EPSILON_DECAY_RATE),
                            id=DField.EPSILON_DECAY_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Current epsilon
                    Horizontal(
                        Label(f"{DLabel.CUR_EPSILON:>15s} : "),
                        self.cur_epsilon,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.EPSILON_BOX,
                ),
                # ----- Model ---
                Vertical(
                    # Model type
                    Horizontal(
                        Label(f"{DLabel.NN_MODEL:>15s} : "),
                        Label(DLabel.LINEAR, id=DField.MODEL_TYPE_LABEL),
                        Select(
                            MODEL_TYPES,
                            compact=True,
                            id=DField.MODEL_TYPE_SELECT,
                            allow_blank=False,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Hidden Size
                    Horizontal(
                        Label(f"{DLabel.HIDDEN_SIZE:>15s} : "),
                        Label(
                            f"{DLinear.HIDDEN_SIZE:.7f}",
                            id=DField.HIDDEN_SIZE_LABEL,
                        ),
                        Input(
                            type=DField.INTEGER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DLinear.HIDDEN_SIZE}",
                            id=DField.HIDDEN_SIZE_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Dropout p-value
                    Horizontal(
                        Label(f"{DLabel.DROPOUT_P_VAL:>15s} : "),
                        Label(
                            f"{DLinear.DROPOUT_P:.2f}",
                            id=DField.DROPOUT_P_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DLinear.DROPOUT_P:.2f}",
                            id=DField.DROPOUT_P_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # RNN Layers
                    Horizontal(
                        Label(
                            f"{DLabel.RNN_LAYERS:>15s} : ",
                            id=DField.RNN_LAYERS_OPT,
                        ),
                        Label(
                            f"{DLinear.LINEAR_LAYERS}",
                            id=DField.RNN_LAYERS_LABEL,
                        ),
                        Input(
                            type=DField.INTEGER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DLinear.LINEAR_LAYERS}",
                            id=DField.RNN_LAYERS_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.MODEL_BOX,
                ),
                # ----- Training ---
                Vertical(
                    # Learning rate
                    Horizontal(
                        Label(f"{DLabel.LEARNING_RATE:>15s} : "),
                        Label(
                            f"{DLinear.LEARNING_RATE:.7f}",
                            id=DField.LEARNING_RATE_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DLinear.LEARNING_RATE:.7f}",
                            id=DField.LEARNING_RATE_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Discount/Gamma
                    Horizontal(
                        Label(f"{DLabel.GAMMA:>15s} : "),
                        Label(
                            f"{DLinear.GAMMA:.2f}",
                            id=DField.GAMMA_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DLinear.GAMMA:.2f}",
                            id=DField.GAMMA_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # Batch size
                    Horizontal(
                        Label(f"{DLabel.BATCH_SIZE:>15s} : "),
                        Label(id=DField.RNN_BATCH_SIZE_LABEL),
                        classes=DField.INPUT_FIELD,
                    ),
                    # RNN Sequence Length
                    Horizontal(
                        Label(f"{DLabel.SEQUENCE_LENGTH:>15s} : "),
                        Label(id=DField.SEQ_LENGTH_LABEL),
                        classes=DField.INPUT_FIELD,
                    ),
                    # RNN Trainer Tau
                    Horizontal(
                        Label(
                            f"{DLabel.RNN_TAU:>15s} : ", id=DField.RNN_TAU_OPT
                        ),
                        Label(
                            f"{DRNN.TAU}",
                            id=DField.RNN_TAU_LABEL,
                        ),
                        Input(
                            type=DField.NUMBER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DRNN.TAU}",
                            id=DField.RNN_TAU_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.TRAINING_BOX,
                ),
            )

            # ----- Memory tab ---
            yield Vertical(
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.MAX_FRAMES:>19s} : "),
                        self.max_frames_label,
                        self.max_frames_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MAX_BUCKETS:>19s} : "),
                        self.max_buckets_label,
                        self.max_buckets_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MAX_TRAINING_FRAMES:>19s} : "),
                        self.max_training_frames_label,
                        self.max_training_frames_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.MEM_SIZING_BOX,
                ),
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.MAX_GEAR:>19s} : "),
                        self.max_gear_label,
                        self.max_gear_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.NUM_COOLDOWN_EPISODES:>19s} : "),
                        self.num_cooldown_eps_label,
                        self.num_cooldown_eps_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.UPSHIFT_COUNT_THRESHOLD:>19s} : "),
                        self.upshift_count_threshold_label,
                        self.upshift_count_threshold_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.DOWNSHIFT_COUNT_THRESHOLD:>19s} : "),
                        self.downshift_count_threshold_label,
                        self.downshift_count_threshold_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.SHIFTING_BOX,
                ),
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.MAX_STAGNANT_EPISODES:>19s} : "),
                        self.max_stag_eps_label,
                        self.max_stag_eps_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MAX_HARD_RESET_EPISODES:>19s} : "),
                        self.max_crit_stag_eps_label,
                        self.max_crit_stag_eps_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.STAGNATION_BOX,
                ),
            )

            # ----- Rewards ---
            yield Vertical(
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.FOOD_REWARD:>20s} : "),
                        self.food_rewards_label,
                        self.food_rewards_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.COLLISION_PENALTY:>20s} : "),
                        self.collision_penalty_label,
                        self.collision_penalty_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MAX_MOVES_PENALTY:>20s} : "),
                        self.max_moves_penalty_label,
                        self.max_moves_penalty_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.REWARD_STRUCTURE_BOX,
                ),
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.EMPTY_MOVE_REWARD:>20s} : "),
                        self.empty_move_reward_label,
                        self.empty_move_reward_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.CLOSER_TO_FOOD:>20s} : "),
                        self.closer_to_food_label,
                        self.closer_to_food_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.FURTHER_FROM_FOOD:>20s} : "),
                        self.further_from_food_label,
                        self.further_from_food_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.MOVEMENT_INCENTIVES_BOX,
                ),
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.MAX_MOVES_MULTIPLIER:>20s} : "),
                        self.max_moves_multiplier_label,
                        self.max_moves_multiplier_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.EPISODE_CONSTRAINTS_BOX,
                ),
            )

            # ----- Policy ---
            yield Vertical(
                # --- Nice ---
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.NICE_P_VALUE:>19s} : "),
                        self.nice_p_value_label,
                        self.nice_p_value_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.NICE_STEPS:>19s} : "),
                        self.nice_steps_label,
                        self.nice_steps_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.NICE_BOX,
                ),
                # --- Monte Carlo Tree Search ---
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.MCTS_GATE_P_VALUE:>19s} : "),
                        self.mcts_gate_p_value_label,
                        self.mcts_gate_p_value_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.SEARCH_DEPTH:>19s} : "),
                        self.mcts_depth_label,
                        self.mcts_depth_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MCTS_ITER:>19s} : "),
                        self.mcts_iter_label,
                        self.mcts_iter_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MCTS_EXPLORE_P_VALUE:>19s} : "),
                        self.mcts_explore_p_value_label,
                        self.mcts_explore_p_value_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.STEPS:>19s} : "),
                        self.mcts_steps_label,
                        self.mcts_steps_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.MCTS_SCORE_THRESH:>19s} : "),
                        self.mcts_score_thresh_label,
                        self.mcts_score_thresh_input,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.MCTS_BOX,
                ),
            )

            # ------ Network tab ---
            yield Vertical(
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.ROUTER_ADDR:>14s} : "),
                        self.router_address,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.ROUTER_PORT:>14s} : "),
                        self.router_port,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.ROUTER_HB_PORT:>14s} : "),
                        self.router_hb_port,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.ROUTER_STATUS:>14s} : "),
                        self.router_hb_status,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.ROUTER_BOX,
                ),
                Vertical(
                    Horizontal(
                        Label(f"{DLabel.SERVER_ADDR:>14s} : "),
                        self.server_address,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.SERVER_PUB_PORT:>14s} : "),
                        self.server_pub_port,
                        classes=DField.INPUT_FIELD,
                    ),
                    Horizontal(
                        Label(f"{DLabel.SERVER_STATUS:>14s} : "),
                        self.server_status,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.SERVER_BOX,
                ),
                id=DField.NETWORK,
            )

    def on_mount(self) -> None:
        self.query_one(f"#{DField.MODEL_BOX}", Vertical).border_subtitle = (
            DLabel.MODEL
        )
        self.query_one(f"#{DField.EPSILON_BOX}", Vertical).border_subtitle = (
            DLabel.EPSILON
        )
        self.query_one(f"#{DField.TRAINING_BOX}", Vertical).border_subtitle = (
            DLabel.TRAINING
        )
        self.query_one(
            f"#{DField.MEM_SIZING_BOX}", Vertical
        ).border_subtitle = DLabel.SIZING
        self.query_one(f"#{DField.SHIFTING_BOX}", Vertical).border_subtitle = (
            DLabel.SHIFTING
        )
        self.query_one(
            f"#{DField.STAGNATION_BOX}", Vertical
        ).border_subtitle = DLabel.STAGNATION
        self.query_one(f"#{DField.ROUTER_BOX}", Vertical).border_subtitle = (
            DLabel.ROUTER
        )
        self.query_one(f"#{DField.SERVER_BOX}", Vertical).border_subtitle = (
            DLabel.SERVER
        )
        self.query_one(
            f"#{DField.REWARD_STRUCTURE_BOX}", Vertical
        ).border_subtitle = DLabel.REWARD_STRUCTURE
        self.query_one(
            f"#{DField.MOVEMENT_INCENTIVES_BOX}", Vertical
        ).border_subtitle = DLabel.MOVEMENT_INCENTIVES
        self.query_one(
            f"#{DField.EPISODE_CONSTRAINTS_BOX}", Vertical
        ).border_subtitle = DLabel.EPISODE_CONSTRAINTS
        self.query_one(f"#{DField.NICE_BOX}", Vertical).border_subtitle = (
            DLabel.EPSILON_NICE
        )
        self.query_one(f"#{DField.MCTS_BOX}", Vertical).border_subtitle = (
            DLabel.MCTS
        )
