from torch import nn
from nntoolbox.components import AveragePool

__all__ = ['CNNModel']


class CNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(40, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=128),
            AveragePool(dim=2),
            nn.Linear(128, 6)
        )
