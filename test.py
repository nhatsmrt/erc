from src.utils import *
from src.models import *
import torch
from torchaudio.transforms import *
from torchvision.transforms import Compose, RandomCrop, ToPILImage, ToTensor
from src.inference import EmoRec

frequency = 16000
transform = Compose(
    [
        RandomCropCenter(30000),
        MFCC(sample_rate=frequency),
        TimePad(280)
    ]
)
test_dataset = ERCDataRaw("data/", False)
model = CNNModel()
model.load_state_dict(torch.load('weights/model_CNN_small.pt', map_location=lambda storage, location: storage))
model.eval()


emo = EmoRec(model, tta_transform=transform)
emo.export_predictions(test_dataset, "data/submission_CNN_3.csv")