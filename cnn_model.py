"""CNN-LSTM architecture for HAR.

Architecture rationale (mentioned in report Q3 - temporal alignment):
  - Conv1d blocks extract local, translation-invariant patterns from the
    6-channel × 300-timestep input.
  - BatchNorm + ReLU + Dropout for regularization.
  - MaxPool1d downsamples the time dimension (300 -> 75).
  - Bidirectional LSTM models long-range temporal dependencies in both
    directions over the conv feature map.
  - Avg + Max pooling over time, concatenated, gives a fixed-size
    representation invariant to where in the window the activity occurred.
  - Final FC head -> 6 logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self,
                 in_channels: int = 6,
                 n_classes: int = 6,
                 conv_channels: int = 64,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        c1, c2 = conv_channels, conv_channels * 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1), nn.ReLU(inplace=True), nn.Dropout(dropout),

            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout),

            nn.Conv1d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=c2, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # avg+max over time, bidirectional -> 4 * lstm_hidden
        head_in = lstm_hidden * 4
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 300)
        h = self.conv(x)                # (B, c2, 75)
        h = h.transpose(1, 2)           # (B, 75, c2)
        h, _ = self.lstm(h)             # (B, 75, 2*lstm_hidden)
        h_avg = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h = torch.cat([h_avg, h_max], dim=1)
        return self.head(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
