from torch import nn
from nntoolbox.vision.components import *
from torchvision.models import resnet18

__all__ = ['CNNModel', 'MediumCNNModel', 'DeepCNNModel', 'ResNet18']


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
            ConvolutionalLayer(3, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 32, 3, stride=2),
            ResidualBlockPreActivation(32),
            FeedforwardBlock(in_channels=32, out_features=6, pool_output_size=4, hidden_layer_sizes=(128,), drop_p=0.5)
        )


class ResNet18(nn.Sequential):
    def __init__(self):
        base = resnet18()
        layers = [
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            FeedforwardBlock(
                in_channels=512, out_features=6, pool_output_size=1,
                hidden_layer_sizes=(128,), drop_p=0.5
            )
        ]
        super().__init__(*layers)
