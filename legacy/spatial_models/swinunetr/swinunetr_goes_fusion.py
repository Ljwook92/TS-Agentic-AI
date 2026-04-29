from __future__ import annotations

import torch

from spatial_models.swinunetr.goes_temporal_encoder import GOESTemporalEncoder
from spatial_models.swinunetr.swinunetr import SwinUNETR


class SwinUNETRGOESFusion(SwinUNETR):
    """SwinUNETR-3D with GOES sub-daily temporal conditioning at the deep feature."""

    def __init__(
        self,
        *args,
        goes_in_features: int = 4,
        goes_hidden_size: int = 128,
        feature_size: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(*args, feature_size=feature_size, **kwargs)
        self.feature_size = feature_size
        self.goes_temporal_encoder = GOESTemporalEncoder(
            in_features=goes_in_features,
            hidden_size=goes_hidden_size,
            out_features=16 * feature_size,
        )

    def forward(self, x_in: torch.Tensor, goes_subdaily: torch.Tensor) -> torch.Tensor:
        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])

        dec4 = self.encoder10(hidden_states_out[4])
        goes_bias = self.goes_temporal_encoder(goes_subdaily).to(dtype=dec4.dtype)
        dec4 = dec4 + goes_bias.view(goes_bias.shape[0], goes_bias.shape[1], 1, 1, 1)

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        out = self.decoder1(dec0, enc0)
        return self.out(out)
