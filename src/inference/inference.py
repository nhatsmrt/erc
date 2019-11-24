from torch import nn


class EmoRec:
    def __init__(self, model: nn.Module):
        self.model = model

    # def predict(self, ):