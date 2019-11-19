import torch
from torch import nn, Tensor
from torchaudio.transforms import MelSpectrogram
import numpy as np


__all__ = ['LogMelSpectrogram', 'RandomlyCrop', 'RandomlyCropFraction']


class LogMelSpectrogram(nn.Module):
    """MelSpectrogram in Log Scale"""
    def __init__(self, **kwargs):
        super().__init__()
        self.mel_spec = MelSpectrogram(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return torch.log(self.mel_spec(input) + 1e-6)


class RandomlyCrop(nn.Module):
    def __init__(self, length: int=48000):
        super().__init__()
        self.length = length

    def forward(self, audio: Tensor):
        if audio.shape[-1] < self.length:
            return audio

        start = np.random.choice(audio.shape[-1] - self.length)
        return audio[:, start:start + self.length]


class RandomlyCropFraction(nn.Module):
    def __init__(self, ratio: float=0.75):
        super().__init__()
        self.ratio = ratio

    def forward(self, audio: Tensor):
        length = self.ratio * audio.shape[-1]
        start = np.random.choice(audio.shape[-1] - length)
        return audio[:, start:start + self.length]

