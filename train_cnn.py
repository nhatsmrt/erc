from torch.utils.data import DataLoader, random_split
from torch import nn
from torchaudio.transforms import MFCC
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from torch.optim import Adam
from src.utils import *
from src.models import *


batch_size = 128
frequency = 16000


transform = MFCC(sample_rate=frequency)


train_val_dataset = ERCData("data/", True, frequency=frequency, transform=transform)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split(train_val_dataset, lengths=[train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = CNNModel()
learner = SupervisedLearner(
    train_loader, val_loader, model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=Adam(model.parameters()),
)
callbacks = [
    ToDeviceCallback(),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
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