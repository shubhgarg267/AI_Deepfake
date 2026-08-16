import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    """Softmax-gated fusion: projects each stream to a common dim, learns a
    per-sample softmax weight over the three streams, fuses as a weighted sum."""

    def __init__(self, spatial_dim=512, freq_dim=128, temporal_dim=128, proj_dim=128, num_classes=1):
        super().__init__()
        self.proj_spatial = nn.Linear(spatial_dim, proj_dim)
        self.proj_freq = nn.Linear(freq_dim, proj_dim)
        self.proj_temporal = nn.Linear(temporal_dim, proj_dim)
        self.gate = nn.Linear(proj_dim * 3, 3)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, spatial, freq, temporal, return_weights=False):
        p_s = self.proj_spatial(spatial)
        p_f = self.proj_freq(freq)
        p_t = self.proj_temporal(temporal)

        gate_logits = self.gate(torch.cat([p_s, p_f, p_t], dim=1))
        weights = torch.softmax(gate_logits, dim=1)  # (batch, 3)

        stacked = torch.stack([p_s, p_f, p_t], dim=1)  # (batch, 3, proj_dim)
        fused = (weights.unsqueeze(2) * stacked).sum(dim=1)  # (batch, proj_dim)

        logits = self.classifier(fused)
        if return_weights:
            return logits, weights
        return logits
