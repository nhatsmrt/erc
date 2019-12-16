from torch.utils.data import DataLoader, random_split
from torch import nn
from torchaudio.transforms import MFCC, MelSpectrogram, Spectrogram
from torchvision.transforms import Compose
from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
from torch.optim import *
from src.utils import *
from src.models import *
import numpy as np

batch_size = 32
frequency = 16000
lr = 0.001

transform_train = Compose(
    [
        # RandomCropCenter(40000),
        # Noise(),
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

run_val_acc = []
for i in range(5):
    print('===== Run {} ===='.format(i))
    
    model = CNNModel()
    # optimizer = Adam([
    #             {'params': model.extractor.parameters(), 'lr': lr / 2},
    #             {'params': model.head.parameters(), 'lr': lr}
    #         ])
    optimizer = Adam(model.parameters(), lr=lr)

    train_val_dataset = ERCDataRaw("data/", True)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_data, val_data = stratified_random_split(
        train_val_dataset, lengths=[train_size, val_size], transforms=[transform_train, transform_val]
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    learner = SupervisedLearner(
        train_loader, val_loader, model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        mixup=True,
        mixup_alpha=0.1
    )
    callbacks = [
        ToDeviceCallback(),
        LossLogger(),
        ModelCheckpoint(learner=learner, filepath="weights/model_{}.pt".format(i), monitor='accuracy', mode='max'),
        ConfusionMatrixCB(),
        ReduceLROnPlateauCB(optimizer, patience=5, factor=0.5),
        Tensorboard()
    ]

    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss()
    }

    final = learner.learn(
        n_epoch=80,
        callbacks=callbacks,
        metrics=metrics,
        final_metric='accuracy'
    )

    run_val_acc.append(final)
np.array(run_val_acc).tofile('weights/val_acc.dat')