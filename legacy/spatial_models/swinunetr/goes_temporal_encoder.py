from __future__ import annotations

import torch
import torch.nn as nn


class GOESTemporalEncoder(nn.Module):
    """Encode fixed-bin GOES FDCF sub-daily features into one conditioning vector."""

    def __init__(
        self,
        in_features: int = 4,
        hidden_size: int = 128,
        out_features: int = 384,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_features)
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, goes_subdaily: torch.Tensor) -> torch.Tensor:
        if goes_subdaily.ndim != 4:
            raise ValueError(
                "GOES sub-daily tensor must have shape [B, T, bins, F], "
                f"got {tuple(goes_subdaily.shape)}"
            )

        batch_size, ts_length, bins_per_day, n_features = goes_subdaily.shape
        x = goes_subdaily.reshape(batch_size, ts_length * bins_per_day, n_features)
        x = torch.log1p(torch.clamp(x, min=0.0))
        x = self.input_norm(x)
        _, hidden = self.gru(x)
        hidden = hidden.transpose(0, 1).reshape(batch_size, -1)
        return self.proj(hidden)
