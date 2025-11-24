# ai_hydra/model/Model.py
#
#    AI Hydra
#    Author: Nadim-Daniel Ghaznavi
#    Copyright: (c) 2025 Nadim-Daniel Ghaznavi
#    GitHub: https://github.com/NadimGhaznavi/ai-hydra
#    Website: https://aihydra.osoyalce.com
#    License: GPL 3.0

from ai_hydra.model.BaseModel import BaseModel


class Model(BaseModel):
    def __init__(self, activation, nlayers, logits=False) -> None:
        super().__init__(activation, nlayers, logits)
