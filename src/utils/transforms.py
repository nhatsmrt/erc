import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import librosa


__all__ = [
    'LogMelSpectrogram', 'DBScaleMelSpectrogram',
    'RandomlyCrop', 'RandomlyCropFraction',
    'RandomFlip', 'RandomCropCenter', 'CropCenter', 'Threshold', 'Smooth',
    'FrequencyMasking', 'TimeMasking', 'NormalizeAcrossTime',
    'DiscardFirstCoeff', 'TimePad', 'AugmentDelta',
    'Noise', 'SpeedNPitch'
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

class Noise:
    def __call__(self, data):
        """
        Adding White Noise.
        """
        # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
        noise_amp = 0.1 * np.random.uniform() * torch.max(data)   # more noise reduce the value to 0.5
        data = data + noise_amp * torch.randn(size=data.shape)
        return data
    
def pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data
    
def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3
    return (data * dyn_change)

class SpeedNPitch:
    def __call__(self, data):
        """
        Speed and Pitch Tuning.
        """
        # you can change low and high here
        length_change = np.random.uniform(low=0.8, high = 1)
        speed_fac = 1.2  / length_change
        tmp = np.interp(np.arange(0, data.shape[-1], speed_fac), np.arange(0, data.shape[-1]), data.view(-1))
        minlen = min(data.shape[-1], tmp.shape[-1])
        data *= 0
        data[:, 0:minlen] = torch.FloatTensor(tmp[0:minlen])
        return data


class RandomFlip:
    def __init__(self, prob: bool=0.5):
        self.prob = prob

    def __call__(self, spectrogram: Tensor) -> Tensor:
        if np.random.rand() < self.prob:
            spectrogram = torch.flip(spectrogram, dims=(-1, ))
        return spectrogram

class RandomCropCenter:
    def __init__(self, length: int=48000):
        self.length = length

    def __call__(self, audio: Tensor):
        if audio.shape[-1] < self.length:
            return audio

        start = np.random.randint(self.length // 2, audio.shape[-1] - self.length // 2 )
        audio = audio[:, start - self.length // 2 : start - self.length // 2 + self.length]

        return audio

class CropCenter:
    def __init__(self, length: int=48000):
        self.length = length

    def __call__(self, audio: Tensor):
        if audio.shape[-1] < self.length:
            return audio

        start = audio.shape[-1] // 2
        audio = audio[:, start - self.length // 2 : start - self.length // 2 + self.length]

        return audio

class Threshold:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def __call__(self, audio: Tensor):
        audio[torch.abs(audio) / torch.max(audio) < self.lower] = 0
        audio[torch.abs(audio) / torch.max(audio) > self.upper] = 0
        return audio

def gauss(n=26,sigma=10):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r]

class Smooth:
    def __call__(self, audio: Tensor) -> Tensor:
        # plt.subplot(2, 1, 1)
        # plt.plot(range(audio.shape[-1]), audio.view(-1))

        kernel = torch.FloatTensor([[gauss()]])
        audio = F.conv1d(audio[None,:,:], kernel)[0]

        # plt.subplot(2, 1, 2)
        # plt.plot(range(audio.shape[-1]), audio.view(-1))
        # plt.show()

        return audio

class TimePad:
    def __init__(self, length, exact: bool=True, pad: str='center'):
        self.length = length
        self.exact = exact
        self.pad = pad

    def __call__(self, spectrogram: Tensor) -> Tensor:
        if self.exact:
            if spectrogram.shape[-1] >= self.length:
                start = spectrogram.shape[-1] // 2 - self.length // 2
                spectrogram = spectrogram[:, :, start : start + self.length]
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
        
        # print(spectrogram.shape[-1])
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
