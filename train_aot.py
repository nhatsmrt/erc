from torch.utils.data import DataLoader, random_split
from torch import nn
from torchaudio.transforms import MFCC
from torchvision.transforms import Compose
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from torch.optim import Adam
from src.utils import *
from src.models import *


batch_size = 128
frequency = 16000
# lr = 5e-4


transform_train = Compose(
    [
        CropCenter(40000),
        Noise(),
        MFCC(sample_rate=frequency, n_mfcc=30),
        TimePad(216)
    ]
)

transform_val = Compose(
    [
        MFCC(sample_rate=frequency, n_mfcc=30),
        TimePad(216)
    ]
)

train_val_dataset = ERCAoTData("data/", True)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split_before_transform(
    train_val_dataset, lengths=[train_size, val_size], transforms=[transform_train, transform_val]
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)


model = CNNAoTModel()
optimizer = Adam(model.parameters())

learner = SupervisedLearner(
    train_loader, val_loader, model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer
)
callbacks = [
    ToDeviceCallback(),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model_aot.pt", monitor='accuracy', mode='max'),
    ConfusionMatrixCB(),
    ReduceLROnPlateauCB(optimizer, patience=7),
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