from torch import nn
from nntoolbox.components import AveragePool

__all__ = ['CNNModel']


class CNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(40, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            AveragePool(dim=2),
            nn.Linear(128, 6)
        )
