from model.BaseModel import BaseModel


class Model(BaseModel):
    def __init__(self, activation, nlayers, logits=False) -> None:
        super().__init__(activation, nlayers, logits)
