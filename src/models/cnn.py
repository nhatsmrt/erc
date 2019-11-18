from torch import nn
from nntoolbox.vision.components import GlobalAveragePool, ConvolutionalLayer, ResidualBlockPreActivation, Flatten

__all__ = ['CNNModel']


class CNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 16, 5),
            ResidualBlockPreActivation(16),
            # ConvolutionalLayer(16, 64, 3, stride=2),
            # ResidualBlockPreActivation(64),
            # GlobalAveragePool(),
            # nn.Linear(64, 6),
            nn.AdaptiveAvgPool2d(4),
            Flatten(),
            nn.Linear(256, 6)
        )
