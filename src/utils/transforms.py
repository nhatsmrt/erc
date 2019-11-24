import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np


__all__ = [
    'LogMelSpectrogram', 'DBScaleMelSpectrogram',
    'RandomlyCrop', 'RandomlyCropFraction',
    'RandomFlip',
    'FrequencyMasking', 'TimeMasking', 'NormalizeAcrossTime',
    'DiscardFirstCoeff', 'TimePad', 'AugmentDelta'
]


class LogMelSpectrogram:
    """MelSpectrogram in Log Scale"""
    def __init__(self, **kwargs):
        self.mel_spec = MelSpectrogram(**kwargs)

    def __call__(self, input: Tensor) -> Tensor:
        return torch.log(self.mel_spec(input) + 1e-6)


class DBScaleMelSpectrogram:
    """MelSpectrogram in DB Scale"""
    def __init__(self, **kwargs):
        self.mel_spec = MelSpectrogram(**kwargs)
        self.db_scale = AmplitudeToDB()

    def __call__(self, input: Tensor) -> Tensor:
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
    Randomly masked some frequency channels of the log mel frequency spectrogram. Apply after normalization.

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
        mask = torch.ones(size=spectrogram.shape, dtype=spectrogram.dtype, device=spectrogram.device)
        mask[:, mask_start:mask_start + mask_length, :] = 0
        return spectrogram * mask


class TimeMasking:
    """
    Randomly masked an interval of time in the log mel frequency spectrogram. Apply after normalization.

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
        mask = torch.ones(size=spectrogram.shape, dtype=spectrogram.dtype, device=spectrogram.device)
        mask[:, :, mask_start:mask_start + mask_length] = 0
        return spectrogram * mask


class NormalizeAcrossTime:
    """
    Normalization across time dimension for each frequency/MFCC coefficient.

    References:

        Haytham Fayek. "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs)
        and What's In-Between." https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

        https://musicinformationretrieval.com/mfcc.html

        https://www.kaggle.com/c/freesound-audio-tagging/discussion/54082
    """
    def __call__(self, input):
        """
        :param input: (C, freq, time)
        :return:
        """
        return (input - input.mean(-1, keepdims=True)) / (input.std(-1, keepdims=True) + 1e-6)


class DiscardFirstCoeff:
    """
    Discard the first MFCC coefficient.

    References:

        Haytham Fayek. "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs)
        and What's In-Between." https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fn:1

        https://musicinformationretrieval.com/mfcc.html
    """
    def __call__(self, mfcc: Tensor) -> Tensor:
        """
        :param mfcc: (C, freq, time)
        :return:
        """
        return mfcc[:, 1:, :]


class RandomFlip:
    def __init__(self, prob: bool=0.5):
        self.prob = prob

    def __call__(self, spectrogram: Tensor) -> Tensor:
        if np.random.rand() < self.prob:
            spectrogram = torch.flip(spectrogram, dims=(-1, ))
        return spectrogram

class TimePad:
    def __init__(self, length, exact: bool=True, pad: str='center'):
        self.length = length
        self.exact = exact
        self.pad = pad

    def __call__(self, spectrogram: Tensor) -> Tensor:
        if self.exact:
            spectrogram = spectrogram[:, :, :self.length]
        if spectrogram.shape[-1] < self.length:
            total_pad = self.length - spectrogram.shape[-1]
            if self.pad == 'center':
                left_pad = total_pad // 2
                right_pad = total_pad - left_pad
                spectrogram = torch.cat(
                    [
                        torch.zeros((spectrogram.shape[0], spectrogram.shape[1], left_pad)),
                        spectrogram,
                        torch.zeros((spectrogram.shape[0], spectrogram.shape[1], right_pad))
                    ],
                    dim=-1
                )
            elif self.pad == 'left':
                spectrogram = torch.cat(
                    [
                        torch.zeros((spectrogram.shape[0], spectrogram.shape[1], total_pad)),
                        spectrogram
                    ],
                    dim=-1
                )
            elif self.pad == 'right':
                spectrogram = torch.cat(
                    [
                        spectrogram,
                        torch.zeros((spectrogram.shape[0], spectrogram.shape[1], total_pad))
                    ],
                    dim=-1
                )

        return spectrogram


class AugmentDelta:
    """
    Augment the spectrogram with 1st and 2nd order deltas
    """
    def __call__(self, spectrogram: Tensor) -> Tensor:
        delta = compute_deltas(spectrogram)
        delta2 = compute_deltas(delta)
        return torch.cat([spectrogram, delta, delta2], dim=0)


#### DIRECTLY FROM TORCHAUDIO MASTER BRANCH
def compute_deltas(specgram, win_length=5, mode="replicate"):
    # type: (Tensor, int, str) -> Tensor
    r"""Compute delta coefficients of a tensor, usually a spectrogram:
    .. math::
        d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N} n^2}
    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is (`win_length`-1)//2.
    Args:
        specgram (torch.Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int): The window length used for computing delta
        mode (str): Mode parameter passed to padding
    Returns:
        deltas (torch.Tensor): Tensor of audio of dimension (..., freq, time)
    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = (
        torch
        .arange(-n, n + 1, 1, device=specgram.device, dtype=specgram.dtype)
        .repeat(specgram.shape[1], 1, 1)
    )

    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom

    # unpack batch
    output = output.reshape(shape)

    return output


class ComputeDeltas(torch.nn.Module):
    r"""Compute delta coefficients of a tensor, usually a spectrogram.
    See `torchaudio.functional.compute_deltas` for more details.
    Args:
        win_length (int): The window length used for computing delta.
    """
    __constants__ = ['win_length']

    def __init__(self, win_length=5, mode="replicate"):
        super(ComputeDeltas, self).__init__()
        self.win_length = win_length
        self.mode = mode

    def forward(self, specgram):
        r"""
        Args:
            specgram (torch.Tensor): Tensor of audio of dimension (channel, freq, time)
        Returns:
            deltas (torch.Tensor): Tensor of audio of dimension (channel, freq, time)
        """
        return compute_deltas(specgram, win_length=self.win_length, mode=self.mode)
