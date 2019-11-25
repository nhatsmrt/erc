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


class To1D:
    def __call__(self, spectrogram: Tensor) -> Tensor:
        """
        :param spectrogram: (1, freq_coeff, time)
        :return: (freq_coeff, time)
        """
        # print(spectrogram.squeeze(0).shape)
        return spectrogram.squeeze(0)


class Conv1DModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(128, 64, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.AdaptiveAvgPool1d(4),
            Flatten(),
            nn.Linear(64, 6)
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
        To1D()
    ]
)

transform_val = Compose(
    [
        DBScaleMelSpectrogram(sample_rate=frequency),
        NormalizeAcrossTime(),
        TimePad(280),
        To1D()
    ]
)

train_val_dataset = ERCDataRaw("data/", True)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split_before_transform(
    train_val_dataset, lengths=[train_size, val_size], transforms=[transform_train, transform_val]
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = Conv1DModel()
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