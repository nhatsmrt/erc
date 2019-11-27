from torch import nn
from nntoolbox.vision.components import *
from torchvision.models import resnet18

__all__ = [
    'CNNModel', 'CNNAoTModel', 'MediumCNNModel',
    'DeepCNNModel', 'ResNet18', 'DeeperCNNModel', 'DeepestCNNModel'
]


class CNNFeatureExtractor(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            nn.AdaptiveAvgPool2d(4),
            Flatten()
        )


class CNNAoTModel(nn.Module):
    def __init__(self, pretrained_fe=None):
        super().__init__()
        self.extractor = CNNFeatureExtractor()
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        x = self.extractor(x)
        x = self.head(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, pretrained_fe=None):
        super().__init__()
        self.extractor = CNNFeatureExtractor()
        if pretrained_fe is not None:
            state_dict = torch.load(pretrained_fe)
            state_dict = { k.replace('extractor.', ''): v for k, v in state_dict.items() if 'extractor.' in k }
            self.extractor.load_state_dict(state_dict)
            for param in self.extractor.parameters():
                param.requires_grad = False
        self.head = nn.Linear(256, 6)

    def forward(self, x):
        x = self.extractor(x)
        x = self.head(x)
        return x


class MediumCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            nn.AdaptiveAvgPool2d(4),
            Flatten(),
            nn.Dropout(),
            nn.Linear(256, 6)
        )


class DeepCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 32, 3, stride=2),
            SEResidualBlockPreActivation(32),
            nn.AdaptiveAvgPool2d(4),
            Flatten(),
            nn.Dropout(),
            nn.Linear(512, 6)
        )


class DeeperCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 32, 3, stride=2),
            SEResidualBlockPreActivation(32),
            FeedforwardBlock(32, 6, 4, (128,))
        )


class DeepestCNNModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 32, 3, stride=2),
            SEResidualBlockPreActivation(32),
            ConvolutionalLayer(32, 64, 3, stride=2),
            SEResidualBlockPreActivation(64),
            FeedforwardBlock(64, 6, 4, (128,))
        )


class ResNet18(nn.Sequential):
    def __init__(self):
        base = resnet18()
        layers = [
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 6)
        ]
        super().__init__(*layers)
