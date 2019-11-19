from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nntoolbox.sequence.utils import extract_last
import torch

__all__ = ['RNNModel']


class SequenceConcatPool(nn.Module):
    def forward(self, input: Tensor, lengths: Tensor):
        """
        :param input: (seq_length, batch_size, n_features)
        :param lengths: (batch_size)
        :return: (batch_size, n_features * 3)
        """
        last = extract_last(input, lengths)
        input_pooled = []
        for i in range(input.shape[1]):
            example = input[:lengths[i], i, :] # (seq_length, n_features)
            avg_pool = example.mean(0)  # (n_features)
            max_pool = example.max(0)[0]  # (n_features)
            input_pooled.append(torch.cat([avg_pool, max_pool], -1))
        return torch.cat([last, torch.stack(input_pooled, dim=0)], dim=-1)


class RNNModel(nn.Module):
    def __init__(self, hidden_size: int=128):
        super().__init__()
        self.input_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(input_size=40, hidden_size=hidden_size, num_layers=2, dropout=0.25, bidirectional=True)
        # self.pool = SequenceConcatPool()
        self.pool = extract_last
        self.op_dropout = nn.Dropout(0.25)
        self.op = nn.Linear(hidden_size * 2, 6)

    def forward(self, input: Tensor, lengths: Tensor) -> Tensor:
        """
        :param input: (N, 1, C, T)
        :return:
        """
        input = input.squeeze(1).permute(2, 0, 1)  # (T, N, C)
        input_packed = pack_padded_sequence(input, lengths, enforce_sorted=False)
        output_packed, _ = self.gru(input_packed)
        output, _ = pad_packed_sequence(output_packed)

        return self.op(self.op_dropout(self.pool(output, lengths)))



