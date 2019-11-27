from src.utils import *
from src.models import *
import torch
from torchaudio.transforms import *
from torchvision.transforms import Compose, RandomCrop, ToPILImage, ToTensor
from src.inference import EmoRec

frequency = 16000
# transform_tta = Compose(
#     [
#         RandomCropCenter(30000),
#         MFCC(sample_rate=frequency),
#         TimePad(280)
#     ]
# )
transform_main = Compose(
    [
        MFCC(sample_rate=frequency, n_mfcc=30),
        TimePad(216)
    ]
)
test_dataset = ERCDataRaw("data/", False)
model = CNNModel()
model.load_state_dict(torch.load('weights/model_PleaseWin.pt', map_location=lambda storage, location: storage))
model.eval()

outputs = []
for image in test_dataset:
    output = model(transform_main(image).unsqueeze(0)).argmax(1)[0].item()
    outputs.append(output)

import pandas as pd

df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": outputs})
df_submission.to_csv("data/submission_model_PleaseWin2.csv", index=False)


# emo = EmoRec(model, transform_main=transform_main)
# emo.export_predictions(test_dataset, "data/submission_model_PleaseWin.csv")

# emo = EmoRec(model, transform_main=transform_main, tta_transform=transform_tta, tta_beta=0.4)
# emo.export_predictions(test_dataset, "data/submission_2_tta_deeper.csv", tries=8)