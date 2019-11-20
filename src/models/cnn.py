from torch import nn
from nntoolbox.vision.components import *

__all__ = ['CNNModel', 'MediumCNNModel', 'DeepCNNModel']


class CNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 16, 5),
            ResidualBlockPreActivation(16),
            nn.AdaptiveAvgPool2d(4),
            Flatten(),
            nn.Linear(256, 6)
        )


class MediumCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            nn.AdaptiveAvgPool2d(4),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(16 * 4 * 4, 6),
        )


class DeepCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 32, 3, stride=2),
            ResidualBlockPreActivation(32),

            # ConvolutionalLayer(32, 64, 3, stride=2),
            # ResidualBlockPreActivation(64),
            # FeedforwardBlock(in_channels=64, out_features=6, pool_output_size=4, hidden_layer_sizes=(128,), drop_p=0.5)

            # nn.AdaptiveAvgPool2d(4),
            # Flatten(),
            # nn.Dropout(p=0.5),
            # nn.Linear(4 * 4 * 32, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(128, 6),


            FeedforwardBlock(in_channels=32, out_features=6, pool_output_size=4, hidden_layer_sizes=(128,), drop_p=0.5)
        )
