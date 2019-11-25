from src.utils import *
from src.models import *
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from typing import List
from torchaudio.transforms import *


# test_dataset = ERCData("data/", False)
#
# batch_size = 128
# frequency = 16000
# max_length = 60000
#
#
# transform = MFCC(sample_rate=frequency)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# model = CNNModel()
# model.load_state_dict(torch.load('weights/model_CNN_3.pt', map_location=lambda storage, location: storage))
# model.eval()
# outputs = []
# for i in range(len(test_dataset)):
#     input = test_dataset[i].unsqueeze(0)
#     output = torch.argmax(model(input), dim=-1).item()
#     outputs.append(output)
#     # print(output)
# # print(outputs)
# df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": outputs})
# df_submission.to_csv("data/submission_2.csv", index=False)
# df_submission.
# print(df_submission)
# print(df_submission[df_submission["File"] == "PAEP-000008"]["Label"].values)


# class EnsembleModel:
#     def __init__(self, models: List[nn.Module], paths: List[str]):
#         self._models = []
#         for model, path in zip(models, paths):
#             model.load_state_dict(torch.load(path, map_location=lambda storage, location: storage))
#             model.eval()
#             self._models.append(model)
#
#     @torch.no_grad()
#     def predict(self, test_loader: DataLoader):
#         outputs = []
#         for images in test_loader:
#             model_outputs = []
#             for model in self._models:
#                 model_outputs.append(F.softmax(model(images), dim=1))
#             outputs.append(torch.stack(model_outputs, dim=0).mean(0))
#
#         return torch.cat(outputs, dim=0).argmax(1).cpu().detach().numpy()
#
# model = CNNModel()
# models = [model]
# # model.load_state_dict(torch.load('weights/model (16).pt', map_location=lambda storage, location: storage))
# # models = [CNNModel() for _ in range(2)]
# # test_dataset = ERCDataV2("data/", False, frequency=frequency, transform=transform, max_length=max_length)
# test_loader = DataLoader(test_dataset, batch_size=128)
# ensemble = EnsembleModel(models, ['weights/model_CNN_2.pt'])
# predictions = ensemble.predict(test_loader)
#
# df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": predictions})
# df_submission.to_csv("data/submission_2.csv", index=False)

from torchvision.transforms import Compose

batch_size = 128
frequency = 16000


transform_val = Compose(
    [
        MFCC(sample_rate=frequency),
        TimePad(280)
    ]
)

test_dataset = ERCDataRaw("data/", False, transform=transform_val)
model = CNNModel()
model.load_state_dict(torch.load('weights/model.pt', map_location=lambda storage, location: storage))
model.eval()
outputs = []
for i in range(len(test_dataset)):
    input = test_dataset[i].unsqueeze(0)
    output = torch.argmax(model(input), dim=-1).item()
    outputs.append(output)
df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": outputs})
df_submission.to_csv("data/submission_2.csv", index=False)





# batch_size = 128
# frequency = 16000



# transform_val = Compose(
#     [
#         DBScaleMelSpectrogram(sample_rate=frequency),
#         NormalizeAcrossTime(),
#         TimePad(280),
#     ]
# )

# test_dataset = ERCDataRaw("data/", False, transform=transform_val)
# model = DeepCNNModel()
# model.load_state_dict(torch.load('weights/model_CNN_big.pt', map_location=lambda storage, location: storage))
# model.eval()
# outputs = []
# for i in range(len(test_dataset)):
#     input = test_dataset[i].unsqueeze(0)
#     output = torch.argmax(model(input), dim=-1).item()
#     outputs.append(output)
# df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": outputs})
# df_submission.to_csv("data/submission_CNN_big.csv", index=False)
