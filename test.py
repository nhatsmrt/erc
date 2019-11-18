from src.utils import ERCData
from src.models import *
import torch
import pandas as pd


batch_size = 128
test_dataset = ERCData("data/", False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = CNNModel()
model.load_state_dict(torch.load('weights/model.pt', map_location=lambda storage, location: storage))
model.eval()
outputs = []
for i in range(len(test_dataset)):
    input = test_dataset[i].unsqueeze(0)
    output = torch.argmax(model(input), dim=-1).item()
    outputs.append(output)
    # print(output)
# print(outputs)
df_submission = pd.DataFrame({"File": test_dataset.filenames, "Label": outputs})
df_submission.to_csv("data/submission.csv", index=False)
# df_submission.
# print(df_submission)
# print(df_submission[df_submission["File"] == "PAEP-000008"]["Label"].values)