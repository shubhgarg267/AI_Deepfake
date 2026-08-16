import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Naive-concatenation fusion baseline, matched in capacity to
    AttentionFusion (~100K params) so any accuracy delta is attributable
    to the fusion mechanism, not model size."""

    def __init__(self, spatial_dim=512, freq_dim=128, temporal_dim=128, hidden_dim=128, num_classes=1):
        super().__init__()
        concat_dim = spatial_dim + freq_dim + temporal_dim
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, spatial, freq, temporal):
        x = torch.cat([spatial, freq, temporal], dim=1)
        x = self.relu(self.fc1(x))
        return self.classifier(x)
