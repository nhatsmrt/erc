import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torchaudio.transforms import MFCC, MelSpectrogram, Spectrogram
from torchvision.transforms import Compose, RandomCrop, Resize, ToPILImage, ToTensor
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from nntoolbox.losses import SmoothedCrossEntropy
from torch.optim import Adam
from src.utils import *
from src.models import *
import numpy as np

batch_size = 128
frequency = 16000
lr = 3e-4
factor = 0.5


transform_train = Compose(
    [
        DBScaleMelSpectrogram(sample_rate=frequency),
        NormalizeAcrossTime(),
        FrequencyMasking(20),
        TimeMasking(32, p=0.20),
        # TimePad(280),
        TimePad(128, False),
        ToPILImage(),
        RandomCrop((128, 128)),
        Resize((256, 256)),
        ToTensor(),
        AugmentDelta()
    ]
)

transform_val = Compose(
    [
        DBScaleMelSpectrogram(sample_rate=frequency),
        NormalizeAcrossTime(),
        TimePad(128, False),
        ToPILImage(),
        RandomCrop((128, 128)),
        Resize((256, 256)),
        ToTensor(),
        AugmentDelta()
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

model = DeepCNNModel()
# model = ResNet18()
optimizer = Adam(model.parameters(), lr=lr)
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