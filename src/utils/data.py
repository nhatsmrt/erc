from torch.utils.data import Dataset
from torchaudio import load_wav
from torchaudio.transforms import MFCC
import os
import torch
import pandas as pd


__all__ = ['ERCData']


class ERCData(Dataset):
    def __init__(self, root: str, training: bool=True, max_length: int=40000):
        self.data = []
        self.transform = MFCC(sample_rate=16000)
        self.training = training
        self.filenames = []

        if training:
            df_labels = pd.read_csv(root + "train_label.csv")
            root = root + "Train/"
            self.labels = []
        else:
            root = root + "Public_Test/"

        for filename in os.listdir(root):
            if filename.endswith(".wav"):
                self.filenames.append(filename)
                input_audio, sample_rate = load_wav(root + filename)
                input_audio = self.transform(input_audio)[0, :, :max_length]
                if input_audio.shape[1] < max_length:
                    input_audio = torch.cat([input_audio, torch.zeros((40, max_length - input_audio.shape[1]))], dim=1)

                self.data.append(input_audio)
                if training:
                    self.labels.append(df_labels.loc[df_labels["File"] == filename, "Label"].values.item())

    def __getitem__(self, i: int):
        if self.training:
            return self.data[i], self.labels[i]
        else:
            return self.data[i]

    def __len__(self):
        return len(self.data)
