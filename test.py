from src.utils import *
from src.models import *
import torch
from torchaudio.transforms import *
from torchvision.transforms import Compose, RandomCrop, ToPILImage, ToTensor
from src.inference import EmoRec

frequency = 16000
transform_tta = Compose(
    [
        RandomCropCenter(30000),
        MFCC(sample_rate=frequency),
        TimePad(280)
    ]
)
transform_main = Compose(
    [
        MFCC(sample_rate=frequency),
        TimePad(280)
    ]
)
test_dataset = ERCDataRaw("data/", False)
model = CNNModel()
model.load_state_dict(torch.load('weights/model (18).pt', map_location=lambda storage, location: storage))
model.eval()


# emo = EmoRec(model, transform_main=transform_main, tta_transform=transform_tta, tta_beta=1.0)
# emo.export_predictions(test_dataset, "data/submission_2.csv")

emo = EmoRec(model, transform_main=transform_main, tta_transform=transform_tta, tta_beta=0.6)
emo.export_predictions(test_dataset, "data/submission_2_tta.csv", tries=10)