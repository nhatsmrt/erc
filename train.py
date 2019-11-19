from torch.utils.data import DataLoader, random_split
from torch import nn
from torchaudio.transforms import MFCC, MelSpectrogram
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from nntoolbox.vision.learner import SupervisedImageLearner
from nntoolbox.losses import SmoothedCrossEntropy
from torch.optim import Adam
from src.utils import ERCData
from src.models import *


batch_size = 128
frequency = 16000
transform = MFCC(sample_rate=frequency)
# transform = MelSpectrogram(sample_rate=frequency)
# transform = MFCC(sample_rate=frequency, log_mels=True)

train_val_dataset = ERCData("data/", True, frequency=frequency, transform=transform)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split(train_val_dataset, lengths=[train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

model = DeepCNNModel()
learner = SupervisedLearner(
    train_loader, val_loader, model=model,
    # criterion=nn.CrossEntropyLoss(),
    criterion=SmoothedCrossEntropy(),
    optimizer=Adam(model.parameters()),
    mixup=True, mixup_alpha=0.4
)
# learner = SupervisedImageLearner(
#     train_loader, val_loader, model=model,
#     criterion=nn.CrossEntropyLoss(), optimizer=Adam(model.parameters()),
#     mixup=True, mixup_alpha=0.2
# )
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


print(final)