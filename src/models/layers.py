import torch
from torch import nn, Tensor
from typing import Union, Tuple
from nntoolbox.vision.utils import compute_output_shape
from typing import Optional


__all__ = ['ICBlock', 'ICResidualBlock', 'ICBlockWithMaxPool', 'Block']


class ICBlock(nn.Sequential):
    """
    Putting batch normalization and dropout before weight layers.

    References:

        Guangyong Chen, Pengfei Chen, Yujun Shi, Chang-Yu Hsieh, Benben Liao, Shengyu Zhang.
        "Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks."
        https://arxiv.org/abs/1905.05928
    """
    def __init__(
            self, in_channels, out_channels, kernel_size: Union[int, Tuple[int, int]]=0,
            padding: Union[int, Tuple[int, int]]=0, stride: int=1, drop_p: float=0.2
    ):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(drop_p, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True)
        )


class ICBlockWithMaxPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, drop_p):
        super().__init__(
            ICBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, drop_p=drop_p),
            nn.MaxPool2d(pool_size),
        )


class ICResidualBlock(nn.Module):
    def __init__(self, in_channels: int, drop_p: float=0.2):
        super().__init__()
        self.conv_path = nn.Sequential(
            ICBlock(in_channels, in_channels, kernel_size=(4, 10), padding=(2, 5), drop_p=drop_p),
            ICBlock(in_channels, in_channels, kernel_size=(4, 10), padding=(2, 5), drop_p=drop_p)
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.conv_path(input)


class Block(nn.Sequential):
    def __init__(self, in_channels, out_features, kernel_size, padding, pool_size, drop_p):
        super().__init__(
            nn.Conv2d(in_channels, out_features, kernel_size, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Dropout(drop_p, inplace=True)
        )


class Uout(nn.Module):
    """
    Adding uniform noise to each hidden unit.

    References:

        Xiang Li, Shuo Chen, Xiaolin Hu, Jian Yang.
        "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift."
        https://arxiv.org/abs/1801.05134
    """
    def __init__(self, beta: float):
        assert beta > 0.0
        super().__init__()
        self.beta = beta

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            noise = torch.rand(size=input.shape, dtype=input.dtype, device=input.device)
            noise = 2 * self.beta * noise - self.beta
            input = input + noise
        return input


# class Uout2d(Uout):
#     """
#     Adding uniform noise to each channel
#
#     """
#     def forward(self, input: Tensor) -> Tensor:
#         if self.training:
#             noise = torch.rand(
#                 size=(input.shape[0], input.shape[1]), dtype=input.dtype, device=input.device
#             )[:, :, None, None]
#             noise = 2 * self.beta * noise - self.beta
#             input = input + noise
#         return input


class ConvGRU2dCell(nn.Module):
    def __init__(
            self, in_channels: int, hidden_channels: int,
            kernel_size_input, kernel_size_hidden=(3, 3),
            padding_input=0, padding_hidden=1,
            bias: bool=True
    ):
        super().__init__()
        self.input_transform = nn.Conv2d(
            in_channels, hidden_channels * 3, kernel_size=kernel_size_input, padding=padding_input, bias=bias
        )
        self.hidden_transform = nn.Conv2d(
            hidden_channels, hidden_channels * 3, kernel_size=kernel_size_hidden, padding=padding_hidden, bias=False
        )
        self.hidden_channels = hidden_channels

    def forward(self, input: Tensor, hidden: Optional[Tensor]) -> Tensor:
        if hidden is None:
            hidden_h, hidden_w = self.compute_hidden_shape(input.shape[2], input.shape[3])
            hidden = torch.zeros(
                size=(input.shape[0], self.hidden_channels, hidden_h, hidden_w)
            ).to(input.device).to(input.dtype)

        input_1, input_2, input_3 = self.input_transform(input).chunk(3, 1)
        hidden_1, hidden_2, hidden_3 = self.hidden_transform(hidden).chunk(3, 1)

        reset_gate = torch.sigmoid(input_1 + hidden_1)
        update_gate = torch.sigmoid(input_2 + hidden_2)
        candidate = torch.tanh(input_3 + reset_gate * (hidden_3))

        return update_gate * candidate + (1 - update_gate) * hidden

    def compute_hidden_shape(self, height: int, width: int) -> Tuple[int, int]:
        return (
            compute_output_shape(
                height, self.input_transform.padding[0],
                self.input_transform.kernel_size[0],
                self.input_transform.dilation[0],
                self.input_transform.stride[0]
            ),
            compute_output_shape(
                width, self.input_transform.padding[1],
                self.input_transform.kernel_size[1],
                self.input_transform.dilation[1],
                self.input_transform.stride[1]
            )
        )


class ConvGRU2d(nn.Module):
    def __init__(
            self, in_channels: int, hidden_channels: int,
            kernel_size_input, kernel_size_hidden=(3, 3),
            padding_input=0, padding_hidden=1,
            bias: bool=True
    ):
        super().__init__()
        self.in_channels, self.hidden_channels, self.bias = in_channels, hidden_channels, bias
        self.cell = ConvGRU2dCell(
            in_channels, hidden_channels,
            kernel_size_input=kernel_size_input, kernel_size_hidden=kernel_size_hidden,
            padding_input=padding_input, padding_hidden=padding_hidden,
            bias=bias
        )

    def forward(self, input: Tensor):
        """
        :param input: (T, N, C, H, W)
        :return:
        """
        hidden = None
        outputs = []

        for t in range(len(input)):
            hidden = self.cell(input[t], hidden)
            outputs.append(hidden)

        return torch.stack(outputs, dim=0), hidden


class ConvolutionalLayerV2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding: int=0, stride: int=1, dropout_p: float=0.2):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.Dropout(dropout_p),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        )


class ResidualBlockPreactivationV2(nn.Module):
    def __init__(self, in_channels: int, dropout_p: float=0.2):
        super().__init__()
        self.conv_path = nn.Sequential(
            ConvolutionalLayerV2(in_channels, in_channels, (4, 10), (2, 5), dropout_p=dropout_p),
            ConvolutionalLayerV2(in_channels, in_channels, (4, 10), (2, 5), dropout_p=dropout_p)
        )

    def forward(self, input):
        return input + self.conv_path(input)
