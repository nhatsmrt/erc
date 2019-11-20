import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np


__all__ = [
    'LogMelSpectrogram', 'DBScaleMelSpectrogram',
    'RandomlyCrop', 'RandomlyCropFraction',
    'FrequencyMasking', 'TimeMasking', 'Normalize',
    'DiscardFirstCoeff'
]


class LogMelSpectrogram:
    """MelSpectrogram in Log Scale"""
    def __init__(self, **kwargs):
        self.mel_spec = MelSpectrogram(**kwargs)

    def __call__(self, input: Tensor) -> Tensor:
        return torch.log(self.mel_spec(input) + 1e-6)


class DBScaleMelSpectrogram:
    """MelSpectrogram in Log Scale"""
    def __init__(self, **kwargs):
        self.mel_spec = MelSpectrogram(**kwargs)
        self.db_scale = AmplitudeToDB()

    def forward(self, input: Tensor) -> Tensor:
        return self.db_scale(self.mel_spec(input))



class RandomlyCrop:
    def __init__(self, length: int=48000):
        self.length = length

    def __call__(self, audio: Tensor):
        if audio.shape[-1] < self.length:
            return audio

        start = np.random.choice(audio.shape[-1] - self.length)
        return audio[:, start:start + self.length]


class RandomlyCropFraction:
    def __init__(self, ratio: float=0.75):
        self.ratio = ratio

    def __call__(self, audio: Tensor):
        length = int(self.ratio * audio.shape[-1])
        start = np.random.choice(audio.shape[-1] - length)
        return audio[:, start:start + length]


class FrequencyMasking:
    """
    Randomly masked some frequency channels of the log mel frequency spectrogram

    References:

        Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le.
        "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition."
        https://arxiv.org/abs/1904.08779v2
    """
    def __init__(self, max_length_mask: int):
        """
        :param max_length_mask: maximum length of mask
        """
        self.max_length_mask = max_length_mask

    def __call__(self, spectrogram: Tensor) -> Tensor:
        """
        :param spectrogram: data in mel spectrogram format. (C, F, L)
        :return: masked data.
        """
        mask_length = np.random.choice(self.max_length_mask)
        mask_start = np.random.choice(spectrogram.shape[1] - mask_length)
        spectrogram[:, mask_start:mask_start + mask_length, :] = 0
        return spectrogram


class TimeMasking:
    """
    Randomly masked an interval of time in the log mel frequency spectrogram.

    References:

        Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le.
        "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition."
        https://arxiv.org/abs/1904.08779v2
    """
    def __init__(self, max_length_mask: int, p: float=1.0):
        """
        :param max_length_mask: maximum length of mask
        :param p: fraction of length (in time) of spectrogram as an upper bound for the length of mask
        """
        self.max_length_mask = max_length_mask
        self.p = p

    def __call__(self, spectrogram: Tensor) -> Tensor:
        """
        :param spectrogram: data in mel spectrogram format. (C, F, L)
        :return: masked data.
        """
        mask_length = min(np.random.choice(self.max_length_mask), int(spectrogram.shape[2] * self.p))
        mask_start = np.random.choice(spectrogram.shape[2] - mask_length)
        spectrogram[:, :, mask_start:mask_start + mask_length] = 0
        return spectrogram


class Normalize:
    """
    Normalization across time dimension for each frequency/MFCC coefficient.

    References:

        Haytham Fayek. "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs)
        and What's In-Between." https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fn:1

        https://musicinformationretrieval.com/mfcc.html
    """
    def __call__(self, input):
        """
        :param input: (C, freq, time)
        :return:
        """
        return (input - input.mean(-1)) / input.std(-1)



DiscardFirstCoeff = lambda mfcc: mfcc[:, 1:, :]
