from torch.utils.data import Dataset
from torchaudio import load_wav
from torchaudio.transforms import MFCC, Resample
import os
import torch
import pandas as pd


__all__ = ['ERCData', 'ERCDataV2']


class ERCData(Dataset):
    def __init__(
            self, root: str, training: bool=True, frequency: int=16000,
            max_length: int=280, transform=None, return_length: bool=False
    ):
        self.data = []
        self.return_length = return_length
        if transform is None:
            self.transform = MFCC(frequency)
        else:
            self.transform = transform

        self.training = training
        self.filenames = []
        self.max_length = max_length
        if frequency != 16000:
            self.resampler = Resample(orig_freq=16000, new_freq=frequency)

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
                if frequency != 16000:
                    input_audio = self.resampler(input_audio)

                self.data.append(input_audio)
                if training:
                    self.labels.append(df_labels.loc[df_labels["File"] == filename, "Label"].values.item())

    def __getitem__(self, i: int):
        input_audio = self.transform(self.data[i])
        length = min(self.max_length, input_audio.shape[-1])
        input_audio = input_audio[:, :, :self.max_length]
        if input_audio.shape[1] < self.max_length:
            input_audio = torch.cat(
                [input_audio, torch.zeros((1, input_audio.shape[1], self.max_length - input_audio.shape[-1]))],
                dim=-1
            )

        if self.return_length:
            if self.training:
                return input_audio, length, self.labels[i]
            else:
                return input_audio, length

        if self.training:
            return input_audio, self.labels[i]
        else:
            return input_audio

    def __len__(self):
        return len(self.data)


class ERCDataV2(ERCData):
    def __getitem__(self, i: int):
        input_audio = self.data[i]
        input_audio = input_audio[:, :self.max_length]
        if input_audio.shape[1] < self.max_length:
            input_audio = torch.cat(
                [input_audio, torch.zeros((1, self.max_length - input_audio.shape[-1]))],
                dim=-1
            )
        input_audio = self.transform(input_audio)
        if self.training:
            return input_audio, self.labels[i]
        else:
            return input_audio
