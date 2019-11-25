import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torchaudio.transforms import MFCC, MelSpectrogram, Spectrogram
from torchvision.transforms import Compose, RandomCrop, Resize, ToPILImage, ToTensor, CenterCrop
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from nntoolbox.losses import SmoothedCrossEntropy
from torch.optim import Adam
from src.utils import *
from src.models import *
import numpy as np
from nntoolbox.vision.components import *
from nntoolbox.sequence.components import *


# class To1D:
#     def __call__(self, spectrogram: Tensor) -> Tensor:
#         """
#         :param spectrogram: (1, freq_coeff, time)
#         :return: (freq_coeff, time)
#         """
#         # print(spectrogram.squeeze(0).shape)
#         return spectrogram.squeeze(0)
#
#
# class Convolutional1DLayer(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, padding: int=1, stride: int=1):
#         super().__init__(
#             nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
#             nn.ReLU(True),
#             nn.BatchNorm1d(out_channels)
#         )
#
#
# class ResidualBlock1D(nn.Sequential):
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.conv_path = nn.Sequential(
#             Convolutional1DLayer(in_channels, in_channels, 3, 1),
#             Convolutional1DLayer(in_channels, in_channels, 3, 1)
#         )
#
#     def forward(self, input):
#         return input + super().forward(input)
#
#
# class Conv1DModel(nn.Sequential):
#     def __init__(self):
#         super().__init__(
#             Convolutional1DLayer(128, 64, 5),
#             ResidualBlock1D(64),
#             Convolutional1DLayer(64, 32, 5, stride=2),
#             ResidualBlock1D(32),
#             Convolutional1DLayer(32, 16, 5, stride=2),
#             ResidualBlock1D(16),
#             nn.AdaptiveAvgPool1d(4),
#             Flatten(),
#             nn.Linear(64, 6)
#         )
#
#     def forward(self, input):
#         return super().forward(input)



class SEModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvolutionalLayer(1, 8, 5),
            SEResidualBlockPreActivation(8),
            ConvolutionalLayer(8, 16, 3, stride=2),
            SEResidualBlockPreActivation(16),
            ConvolutionalLayer(16, 32, 3, stride=2),
            SEResidualBlockPreActivation(32),
            FeedforwardBlock(in_channels=32, out_features=6, pool_output_size=4, hidden_layer_sizes=(128,), drop_p=0.5)
        )



batch_size = 128
frequency = 16000
# lr = 3e-4
# lr = 3e-4
factor = 0.1


transform_train = Compose(
    [
        DBScaleMelSpectrogram(sample_rate=frequency),
        NormalizeAcrossTime(),
        FrequencyMasking(20),
        TimeMasking(32, p=0.20),
        TimePad(280),
        # To1D()
    ]
)

transform_val = Compose(
    [
        DBScaleMelSpectrogram(sample_rate=frequency),
        NormalizeAcrossTime(),
        TimePad(280),
        # To1D()
    ]
)

train_val_dataset = ERCDataRaw("data/", True)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split_before_transform(
    train_val_dataset, lengths=[train_size, val_size], transforms=[transform_train, transform_val]
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# for image, _ in train_loader:
#     print(image.shape)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = SEModel()
# model = ResNet18()
optimizer = Adam(model.parameters())
learner = SupervisedLearner(
    train_loader, val_loader, model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    mixup=True, mixup_alpha=0.4
)
callbacks = [
    ToDeviceCallback(),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
    ReduceLROnPlateauCB(optimizer=optimizer, patience=10, factor=factor),
    Tensorboard()
]

metrics = {
    "accuracy": Accuracy(),
    "loss": Loss()
}

final = learner.learn(
    n_epoch=500,
    callbacks=callbacks,
    metrics=metrics,
    final_metric='accuracy'
)


print(final)