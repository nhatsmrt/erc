from torch.utils.data import DataLoader, random_split
from torch import nn
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from nntoolbox.components import AveragePool
from nntoolbox.utils import get_device
from torch.optim import Adam
from src.utils import ERCData


train_val_dataset = ERCData("data/", True)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split(train_val_dataset, lengths=[train_size, val_size])


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)


model = nn.Sequential(
    # MFCC(sample_rate=16000),
    nn.Conv1d(40, 64, 3),
    nn.ReLU(),
    nn.Conv1d(64, 128, 3),
    nn.ReLU(),
    nn.Conv1d(128, 10, 3),
    AveragePool(dim=2)
).to(get_device())

learner = SupervisedLearner(train_loader, val_loader, model=model, criterion=nn.CrossEntropyLoss(), optimizer=Adam(model.parameters()))
callbacks = [
    ToDeviceCallback(),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
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
