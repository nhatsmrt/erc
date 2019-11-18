from torch import nn, Tensor
from nntoolbox.components import ConcatPool

__all__ = ['RNNModel']


class RNNModel(nn.Module):
    def __init__(self, hidden_size: int=128):
        super().__init__()
        self.gru = nn.GRU(input_size=40, hidden_size=hidden_size, num_layers=2, dropout=0.5, bidirectional=True)
        self.concat_pool = ConcatPool(concat_dim=-1, pool_dim=0)
        self.op = nn.Linear(hidden_size * 6, 6)


    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: (N, 1, C, T)
        :return:
        """
        input = input.squeeze(1).permute(2, 0, 1)
        output, _ = self.gru(input)
        return self.op(self.concat_pool(output))



