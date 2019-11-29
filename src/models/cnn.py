from torch import nn
from nntoolbox.vision.components import *
from torchvision.models import resnet18

__all__ = [
    'CNNModel', 'CNNAoTModel', 'MediumCNNModel',
    'DeepCNNModel', 'ResNet18', 'DeeperCNNModel', 'DeeperCNNModelV2', 'DeepestCNNModel'
]

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_features, kernel_size, padding, pool_size, drop_p):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_features, kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        res = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        if res.shape[1] == x.shape[1]:
            x += res
        x = self.pool(x)
        x = self.dropout(x)
        return x

class Block(nn.Sequential):
    def __init__(self, in_channels, out_features, kernel_size, padding, pool_size, drop_p):
        super().__init__(
            nn.Conv2d(in_channels, out_features, kernel_size, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Dropout(drop_p)
        )

class CNNFeatureExtractor2(nn.Sequential):
    def __init__(self, size):
        super().__init__(
            Block(1, 32, (3, 9), (1, 4), 2, 0.2),
            Block(32, 32, (3, 9), (1, 4), 2, 0.2),
            Block(32, 32, (3, 9), (1, 4), 2, 0.2),
            Block(32, 32, (3, 9), (1, 4), 2, 0.2),

            Flatten(),

            nn.Linear(416, size),
            nn.Dropout(0.2),
            nn.BatchNorm1d(size),
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
        self.extractor = CNNFeatureExtractor2(256)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        x = self.extractor(x)
        x = self.head(x)
        return x

class CNNModel(nn.Module):
    def __init__(self, pretrained_fe=None):
        super().__init__()
        self.extractor = CNNFeatureExtractor2(256)
        if pretrained_fe is not None:
            state_dict = torch.load(pretrained_fe)
            state_dict = { k.replace('extractor.', ''): v for k, v in state_dict.items() if 'extractor.' in k }
            self.extractor.load_state_dict(state_dict)
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
            nn.Dropout2d(0.25, inplace=True),
            ResidualBlockPreActivation(8),
            nn.Dropout2d(0.25, inplace=True),
            ConvolutionalLayer(8, 16, 3, stride=2),
            nn.Dropout2d(0.25, inplace=True),
            ResidualBlockPreActivation(16),
            nn.Dropout2d(0.25, inplace=True),
            ConvolutionalLayer(16, 32, 3, stride=2),
            nn.Dropout2d(0.25, inplace=True),
            SEResidualBlockPreActivation(32),
            nn.Dropout2d(0.25, inplace=True),
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
