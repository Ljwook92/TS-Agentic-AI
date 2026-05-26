from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_models.swinunetr.swinunetr import SwinUNETR


class GOESSpatialEncoder(nn.Module):
    """Encode GOES cumulative spatial maps onto the SwinUNETR bottleneck grid."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(32, out_channels // 4)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, goes_spatial: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if goes_spatial.ndim != 5:
            raise ValueError(
                "Expected GOES spatial tensor with shape [B, C, T, H, W], "
                f"got {tuple(goes_spatial.shape)}"
            )
        x = torch.nan_to_num(goes_spatial.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = self.net(x)
        return F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)


class SwinUNETRGOESSpatialFusion(SwinUNETR):
    """SwinUNETR-3D with spatial GOES progression maps fused at the bottleneck."""

    def __init__(
        self,
        *args,
        goes_in_channels: int = 6,
        feature_size: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(*args, feature_size=feature_size, **kwargs)
        self.feature_size = feature_size
        self.goes_spatial_encoder = GOESSpatialEncoder(
            in_channels=goes_in_channels,
            out_channels=16 * feature_size,
        )
        # Start near VIIRS-only behavior, then learn how much spatial GOES
        # progression should perturb the deep representation.
        self.goes_gate_logit = nn.Parameter(torch.tensor(-6.0))

    def forward(self, x_in: torch.Tensor, goes_spatial: torch.Tensor) -> torch.Tensor:
        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])

        dec4 = self.encoder10(hidden_states_out[4])
        goes_bias = self.goes_spatial_encoder(goes_spatial, target_shape=dec4.shape[2:]).to(dtype=dec4.dtype)
        goes_gate = torch.sigmoid(self.goes_gate_logit) * 0.1
        dec4 = dec4 + goes_gate * goes_bias

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        out = self.decoder1(dec0, enc0)
        return self.out(out)


class SwinUNETRGOESSpatialDecoderGate(SwinUNETR):
    """Use GOES spatial maps as decoder attention gates instead of residual features."""

    def __init__(
        self,
        *args,
        goes_in_channels: int = 6,
        feature_size: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(*args, feature_size=feature_size, **kwargs)
        self.feature_size = feature_size
        self.goes_gate5 = GOESSpatialEncoder(goes_in_channels, 8 * feature_size)
        self.goes_gate4 = GOESSpatialEncoder(goes_in_channels, 4 * feature_size)
        self.goes_gate3 = GOESSpatialEncoder(goes_in_channels, 2 * feature_size)
        self.goes_gate2 = GOESSpatialEncoder(goes_in_channels, feature_size)
        # Initial multiplier is tiny, so the model starts very close to VIIRS-only.
        self.gate_alpha_logit = nn.Parameter(torch.tensor(-6.0))

    def apply_gate(self, feature: torch.Tensor, gate_encoder: nn.Module, goes_spatial: torch.Tensor) -> torch.Tensor:
        gate_logits = gate_encoder(goes_spatial, target_shape=feature.shape[2:]).to(dtype=feature.dtype)
        gate = torch.sigmoid(gate_logits)
        alpha = torch.sigmoid(self.gate_alpha_logit) * 0.1
        return feature * (1.0 + alpha * gate)

    def forward(self, x_in: torch.Tensor, goes_spatial: torch.Tensor) -> torch.Tensor:
        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])

        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec3 = self.apply_gate(dec3, self.goes_gate5, goes_spatial)

        dec2 = self.decoder4(dec3, enc3)
        dec2 = self.apply_gate(dec2, self.goes_gate4, goes_spatial)

        dec1 = self.decoder3(dec2, enc2)
        dec1 = self.apply_gate(dec1, self.goes_gate3, goes_spatial)

        dec0 = self.decoder2(dec1, enc1)
        dec0 = self.apply_gate(dec0, self.goes_gate2, goes_spatial)

        out = self.decoder1(dec0, enc0)
        return self.out(out)
