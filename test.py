from src.utils import *
from src.models import *
import torch
from torchaudio.transforms import *
from torchvision.transforms import Compose, RandomCrop, ToPILImage, ToTensor
from src.inference import EmoRec
import sys

_id = sys.argv[1]

frequency = 16000
transform_main = Compose(
    [
        MFCC(sample_rate=frequency, n_mfcc=30),
        TimePad(216)
    ]
)
test_dataset = ERCDataRaw("data/", False)
model = CNNModel()
model.load_state_dict(torch.load('weights/{}.pt'.format(_id), map_location=lambda storage, location: storage))
model.eval()

outputs = []
for image in test_dataset:
    output = model(transform_main(image).unsqueeze(0)).argmax(1)[0].item()
    outputs.append(output)

import pandas as pd

df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": outputs})
df_submission.to_csv("data/submission_{}.csv".format(_id), index=False)