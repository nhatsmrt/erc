from torch import nn
from nntoolbox.vision.components import *
from .layers import *
from torchvision.models import resnet18

__all__ = [
    'CNNModel', 'CNNAoTModel', 'MediumCNNModel',
    'DeepCNNModel', 'ResNet18', 'DeeperCNNModel',
    'DeeperCNNModelV2', 'ModifiedNahModel','ICModel'
]


class CNNFeatureExtractor2(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, (4, 10), padding=(2,5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 32, (4, 10), padding=(2,5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 32, (4, 10), padding=(2,5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 32, (4, 10), padding=(2,5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            Flatten(),
            nn.Linear(896, 256),

            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

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
        self.extractor = CNNFeatureExtractor2()
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
            ConvolutionalLayer(3, 8, 5),
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


class DeeperCNNModelV2(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            ResidualBlockPreActivation(8),
            nn.Dropout2d(0.2, inplace=True),
            ConvolutionalLayer(8, 16, 3, stride=2),
            ResidualBlockPreActivation(16),
            nn.Dropout2d(0.2, inplace=True),
            ConvolutionalLayer(16, 32, 3, stride=2),
            SEResidualBlockPreActivation(32),
            nn.Dropout2d(0.2, inplace=True),
            FeedforwardBlock(32, 6, 4, (128,))
        )


class ModifiedNahModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            Block(1, 32, (4, 10), (2, 5), 2, 0.2),
            Block(32, 32, (4, 10), (2, 5), 2, 0.2),
            Block(32, 32, (4, 10), (2, 5), 2, 0.2),
            Block(32, 32, (4, 10), (2, 5), 2, 0.2),

            Flatten(),
            nn.Linear(896, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )


class ICModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ICBlockWithMaxPool(1, 32, (4, 10), (2, 5), 2, 0.2),
            ICBlockWithMaxPool(32, 32, (4, 10), (2, 5), 2, 0.2),
            ICBlockWithMaxPool(32, 32, (4, 10), (2, 5), 2, 0.2),
            ICBlockWithMaxPool(32, 32, (4, 10), (2, 5), 2, 0.2),

            Flatten(),
            nn.Linear(896, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
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
