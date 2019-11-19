from torch import nn
from nntoolbox.vision.components import *

__all__ = ['CNNModel', 'CNNModelV2', 'DeepCNNModel']


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


class CNNModelV2(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 16, 5),
            ResidualBlockPreActivation(16),
            nn.AdaptiveAvgPool2d((4, 16)),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(4 * 16 * 16, 6),
        )


class DeepCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 16, 5),
            ResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 64, 3, stride=2),
            ResidualBlockPreActivation(64),
            ConvolutionalLayer(64, 128, 3, stride=2),
            ResidualBlockPreActivation(128),
            nn.AdaptiveAvgPool2d((2, 4)),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * 4 * 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 6),
            nn.ReLU(inplace=True)
        )
