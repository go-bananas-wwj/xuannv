from torch import nn
import torch


class LearnedSpatialResampling(nn.Module):
    """Learned spatial resampling for cross-scale pyramid exchanges.
    
    Supports arbitrary integer scale factors for both upsampling and downsampling.
    Uses resize-conv pattern (Upsample + Conv2d) to avoid checkerboard artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

        if scale_factor > 1:
            # Upsampling: use resize-conv to avoid checkerboard artifacts
            assert scale_factor == int(scale_factor), (
                f"Upsample scale_factor must be integer, got {scale_factor}"
            )
            self.upsample = nn.Upsample(
                scale_factor=int(scale_factor),
                mode='bilinear',
                align_corners=False,
            )
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        elif scale_factor < 1:
            # Downsampling
            stride = int(1 / scale_factor)
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=stride * 2 - 1,
                stride=stride,
                padding=stride - 1,
            )
        else:
            # Same resolution
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'upsample'):
            x = self.upsample(x)
        return self.conv(x)
