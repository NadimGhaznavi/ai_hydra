# ai_hydra/client/TabbedSettings.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025-2026 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai_hydra
#    Website: https://ai-hydra.readthedocs.io/en/latest
#    License: GPL 3.0

from __future__ import annotations

from textual import on
from textual.app import ComposeResult, Widget
from textual.widgets import TabbedContent, Label, Input, Select
from textual.containers import Horizontal, Vertical

from ai_hydra.constants.DNNet import DLinear, DRNN, MODEL_TYPES
from ai_hydra.constants.DHydraTui import DLabel, DField, DStatus


class TabbedSettings(Widget):
    """A TabbedContent widget to house simulation settings"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.router_address = Label(id=DField.ROUTER_ADDR)
        self.router_port = Label(id=DField.ROUTER_PORT)
        self.router_hb_port = Label(id=DField.ROUTER_HB_PORT)
        self.router_hb_status = Label(id=DField.ROUTER_HB_STATUS)
        self.server_address = Label(id=DField.SERVER_ADDR)
        self.server_pub_port = Label(id=DField.SERVER_PUB_PORT)
        self.server_status = Label(id=DField.SERVER_STATUS)
        self.cur_epsilon = Label(id=DField.CUR_EPSILON)

    def compose(self) -> ComposeResult:
        with TabbedContent(DLabel.SETTINGS, DLabel.NETWORK):

            # ----- Settings Tab ---
            yield Vertical(
                # ----- Epsilon ---
                Vertical(
                    # Initial epsilon
                    Horizontal(
                        Label(f"{DLabel.INITIAL_EPSILON:>15s}: "),
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
                        Label(f"{DLabel.MIN_EPSILON:>15s}: "),
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
                        Label(f"{DLabel.EPSILON_DECAY:>15s}: "),
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
                        Label(f"{DLabel.CUR_EPSILON:>15s}: "),
                        self.cur_epsilon,
                        classes=DField.INPUT_FIELD,
                    ),
                    id=DField.EPSILON_BOX,
                ),
                # ----- Model ---
                Vertical(
                    # Model type
                    Horizontal(
                        Label(f"{DLabel.NN_MODEL:>15s}: "),
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
                        Label(f"{DLabel.HIDDEN_SIZE:>15s}: "),
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
                        Label(f"{DLabel.DROPOUT_P_VAL:>15s}: "),
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
                            f"{DLabel.RNN_LAYERS:>15s}: ",
                            id=DField.RNN_LAYERS_OPT,
                        ),
                        Label(
                            f"{DRNN.RNN_LAYERS}",
                            id=DField.RNN_LAYERS_LABEL,
                        ),
                        Input(
                            type=DField.INTEGER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DRNN.RNN_LAYERS}",
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
                        Label(f"{DLabel.LEARNING_RATE:>15s}: "),
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
                        Label(f"{DLabel.GAMMA:>15s}: "),
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
                        Label(f"{DLabel.BATCH_SIZE:>15s}: "),
                        Label(
                            f"{DLinear.BATCH_SIZE}",
                            id=DField.BATCH_SIZE_LABEL,
                        ),
                        Label(
                            f"{DRNN.BATCH_SIZE}",
                            id=DField.RNN_BATCH_SIZE_LABEL,
                        ),
                        Input(
                            type=DField.INTEGER,
                            compact=True,
                            valid_empty=False,
                            value=f"{DLinear.BATCH_SIZE}",
                            id=DField.BATCH_SIZE_INPUT,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # RNN Sequence Length
                    Horizontal(
                        Label(
                            f"{DLabel.SEQUENCE_LENGTH:>15s}: ",
                            id=DField.SEQ_LENGTH_OPT,
                        ),
                        Label(
                            f"{DRNN.SEQ_LENGTH}",
                            id=DField.SEQ_LENGTH_LABEL,
                        ),
                        classes=DField.INPUT_FIELD,
                    ),
                    # RNN Trainer Tau
                    Horizontal(
                        Label(
                            f"{DLabel.RNN_TAU:>15s}: ", id=DField.RNN_TAU_OPT
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
                    ),
                    id=DField.TRAINING_BOX,
                ),
            )

            # ------ Network tab ---
            yield Vertical(
                Horizontal(
                    Label(f"{DLabel.ROUTER_ADDR} : "),
                    self.router_address,
                    classes=DField.INPUT_FIELD,
                ),
                Horizontal(
                    Label(f"{DLabel.ROUTER_PORT}    : "),
                    self.router_port,
                    classes=DField.INPUT_FIELD,
                ),
                Horizontal(
                    Label(f"{DLabel.ROUTER_HB_PORT} : "),
                    self.router_hb_port,
                    classes=DField.INPUT_FIELD,
                ),
                Horizontal(
                    Label(f"{DLabel.ROUTER_STATUS}  : "),
                    self.router_hb_status,
                    classes=DField.INPUT_FIELD,
                ),
                Label(),
                Horizontal(
                    Label(f"{DLabel.SERVER_ADDR} : "),
                    self.server_address,
                    classes=DField.INPUT_FIELD,
                ),
                Horizontal(
                    Label(f"{DLabel.SERVER_PUB_PORT}: "),
                    self.server_pub_port,
                    classes=DField.INPUT_FIELD,
                ),
                Horizontal(
                    Label(f"{DLabel.SERVER_STATUS}  : "),
                    self.server_status,
                    classes=DField.INPUT_FIELD,
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
