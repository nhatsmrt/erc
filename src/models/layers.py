from torch import nn


__all__ = ['ICBlock', 'Block']


class ICBlock(nn.Sequential):
    """
    Putting batch normalization and dropout before weight layers.

    References:

        Guangyong Chen, Pengfei Chen, Yujun Shi, Chang-Yu Hsieh, Benben Liao, Shengyu Zhang.
        "Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks."
        https://arxiv.org/abs/1905.05928
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride: int=1, drop_p: float=0.2):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.Dropout(drop_p),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )


class Block(nn.Sequential):
    def __init__(self, in_channels, out_features, kernel_size, padding, pool_size, drop_p):
        super().__init__(
            nn.Conv2d(in_channels, out_features, kernel_size, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Dropout2d(drop_p, inplace=True)
        )

