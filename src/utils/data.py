from torch.utils.data import Dataset
from torchaudio import load_wav
from torchaudio.transforms import MFCC, Resample
import os
import torch
import pandas as pd
from torch import randperm
from torch._utils import _accumulate
<<<<<<< HEAD

__all__ = ['ERCData', 'ERCDataV2', 'ERCDataRaw', 'TransformableSubset', 'random_split_before_transform']
=======
import numpy as np

__all__ = ['ERCData', 'ERCDataV2', 'ERCDataRaw', 'ERCAoTData',
            'TransformableSubset', 'random_split_before_transform']
>>>>>>> master


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
        if input_audio.shape[-1] < self.max_length:
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


class ERCDataRaw(Dataset):
    def __init__(self, root: str, training: bool=True, return_length: bool=False, transform=None):
        self.data = []
        self.return_length = return_length
        self.transform = transform

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

                self.data.append(input_audio)
                if training:
                    self.labels.append(df_labels.loc[df_labels["File"] == filename, "Label"].values.item())

    def __getitem__(self, i: int):
        input_audio = self.data[i]
        if self.transform is not None:
            input_audio = self.transform(input_audio)

        if self.training:
            return input_audio, self.labels[i]
        else:
            return input_audio

    def __len__(self):
        return len(self.data)

class ERCAoTData(ERCDataRaw):
    def __init__(self, root: str, training: bool=True, return_length: bool=False, transform=None, flip_prob=0.5):
        super().__init__(root, training, return_length, transform)
        self.flip_prob = flip_prob

    def __getitem__(self, i: int):
        input_audio = self.data[i]

        if self.transform is not None:
            input_audio = self.transform(input_audio)

        if self.training:
            label = 0
            if np.random.rand() < self.flip_prob:
                input_audio = torch.flip(input_audio, dims=(-1, ))
                label = 1
            return input_audio, label
        else:
            return input_audio

class TransformableSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        audio, label = self.dataset[self.indices[idx]]
        audio = self.transform(audio)
        return audio, label

    def __len__(self):
        return len(self.indices)


def random_split_before_transform(dataset, lengths, transforms):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths before transforming the input.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        transforms: transformation to apply to data
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [
        TransformableSubset(dataset, indices[offset - length:offset], transform)
        for offset, length, transform in zip(_accumulate(lengths), lengths, transforms)
    ]

