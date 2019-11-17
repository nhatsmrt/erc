from torch import nn
from nntoolbox.vision.components import GlobalAveragePool

__all__ = ['CNNModel']


class CNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            GlobalAveragePool(),
            nn.Linear(128, 6)
        )
