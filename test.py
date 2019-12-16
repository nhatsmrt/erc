from src.utils import *
from src.models import *
import torch
from torchaudio.transforms import *
from torchvision.transforms import Compose, RandomCrop, ToPILImage, ToTensor
from src.inference import EmoRec
import sys
import pandas as pd
import numpy as np
from scipy import stats

def count(arr):
    cnt = dict()
    for x in arr:
        cnt.setdefault(x, 0)
        cnt[x] += 1
    return cnt

def choose(arr, best):
    cnt = count(arr)
    cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
        if cnt[0][0] == arr[best] or cnt[1][0] == arr[best]: return arr[best]
        return cnt[0][0] if np.random.rand() < 0.5 else cnt[1][0]
    return cnt[0][0]

results = dict()
for i in range(5):
    frequency = 16000
    transform_main = Compose(
        [
            MFCC(sample_rate=frequency, n_mfcc=30),
            TimePad(216)
        ]
    )
    test_dataset = ERCDataRaw("data/", False)
    model = CNNModel()
    model.load_state_dict(torch.load('weights/model_{}.pt'.format(i), map_location=lambda storage, location: storage))
    model.eval()

    outputs = []
    for image in test_dataset[:5]:
        output = model(transform_main(image).unsqueeze(0)).argmax(1)[0].item()
        outputs.append(output)

    for _id, result in zip(test_dataset.filenames[:5], outputs):
        results.setdefault(_id, [])
        results[_id].append(result)

    print('Run #{} done.'.format(i))

best = np.argmax(np.fromfile('weights/val_acc.dat'))
f = open('submission.csv', 'w')
f.write('File,Label\r\n')
for k, v in results.items():
    f.write('{},{}\r\n'.format(k, choose(v, best)))