import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalBiLSTM(nn.Module):
    """Bi-LSTM over a sequence of per-frame spatial embeddings.
    Consumes variable-length sequences (8-10 frames) via pack_padded_sequence."""

    def __init__(self, input_size=512, hidden_size=64, num_classes=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def embed(self, padded_seqs, lengths):
        packed = pack_padded_sequence(
            padded_seqs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_directions, batch, hidden_size) -> concat fwd/bwd final states
        return torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, hidden_size*2)

    def forward(self, padded_seqs, lengths):
        return self.classifier(self.embed(padded_seqs, lengths))
