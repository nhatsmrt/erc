import torch
from torch import nn, Tensor
from torchaudio.transforms import MelSpectrogram


__all__ = ['LogMelSpectrogram']


class LogMelSpectrogram(nn.Module):
    """MelSpectrogram in Log Scale"""
    def __init__(self):
        super().__init__()
        self.mel_spec = MelSpectrogram()

    def forward(self, input: Tensor) -> Tensor:
        return torch.log(self.mel_spec(input) + 1e-6)
