from torch import nn
import torch
import torch.nn.functional as F


class LearnedSpatialResampling(nn.Module):
    """Learned spatial resampling for cross-scale pyramid exchanges.
    
    Supports arbitrary integer scale factors for both upsampling and downsampling.
    Uses resize-conv pattern (Upsample + Conv2d) to avoid checkerboard artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

        if scale_factor > 1:
            # Upsampling: use resize-conv with TWO conv layers to increase receptive field
            # and better blend interpolation blocks (especially for large scale factors)
            assert scale_factor == int(scale_factor), (
                f"Upsample scale_factor must be integer, got {scale_factor}"
            )
            self.upsample = nn.Upsample(
                scale_factor=int(scale_factor),
                mode='bilinear',
                align_corners=False,
            )
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        elif scale_factor < 1:
            # Downsampling: use interpolate + conv for NPU-friendly backward
            # (large-kernel stride conv triggers slow Conv2DBackpropInput on NPU)
            self.downsample = nn.Upsample(
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False,
            )
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            # Same resolution
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.conv2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'upsample'):
            x = self.upsample(x)
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        x = self.conv1(x)
        if self.conv2 is not None:
            x = F.gelu(x)
            x = self.conv2(x)
        return x
