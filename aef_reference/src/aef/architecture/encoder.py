from einops import rearrange
import torch
from torch.functional import F 
import torch.nn as nn

from src.aef.architecture.STPBlock import STPBlock


class STPEncoder(nn.Module):
    """
    Space Time Precision encoder.
    
    Key change: space/time pathways now provide global context via GAP + projection
    instead of upsampling low-res features, which eliminates grid/checkerboard artifacts
    from bilinear upsampling of coarse feature maps.
    """
    
    def __init__(self, input_channels: int, d_s: int = 1024, d_t: int = 512, d_p: int = 128, num_blocks: int = 15):
        super().__init__()
        self.space_dim = d_s
        self.time_dim = d_t
        self.precision_dim = d_p
        
        # Project inputs to common latent space
        self.input_projection = nn.Linear(input_channels, self.precision_dim)
        
        # Pathway-specific projections
        self.space_projection = nn.Linear(self.precision_dim, self.space_dim)
        self.time_projection = nn.Linear(self.precision_dim, self.time_dim)
        
        # STP blocks as per paper
        self.blocks = nn.ModuleList([STPBlock(d_s, d_t, d_p) for _ in range(num_blocks)])
        
        # Global context projections: low-res space/time -> precision_dim for broadcasting
        self.space_to_precision = nn.Linear(d_s, d_p)
        self.time_to_precision = nn.Linear(d_t, d_p)
        
        # Light channel fusion (1x1 conv to avoid introducing spatial/y-correlation)
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(d_p, d_p, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_p, d_p, kernel_size=1),
        )
        
        # Output norm
        self.norm = nn.LayerNorm(self.precision_dim)
        
    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, H, W, C) - preprocessed multi-source data
            timestamps: Millisecond timestamps (B, T)
            
        Returns:
            Features at full resolution: (B, T, H, W, precision_dim)
        """
        B, T, H, W, C = x.shape
        
        # Project inputs to common latent space
        x_proj = self.input_projection(x)
        
        # Initialize features at different resolutions using pathway projections
        # Space pathway: project to space_dim and downsample to 1/8L
        space_features = self.space_projection(x_proj)
        space_features = F.adaptive_avg_pool2d(
            rearrange(space_features, 'b t h w c -> (b t) c h w'),
            (H // 8, W // 8)
        )
        space_features = rearrange(space_features, '(b t) c h w -> b t h w c', b=B, t=T)
        
        # Time pathway: project to time_dim and downsample to 1/4L
        time_features = self.time_projection(x_proj)
        time_features = F.adaptive_avg_pool2d(
            rearrange(time_features, 'b t h w c -> (b t) c h w'),
            (H // 4, W // 4)
        )
        time_features = rearrange(time_features, '(b t) c h w -> b t h w c', b=B, t=T)
        
        # Precision pathway: keep at precision_dim and full resolution (H, W)
        precision_features = F.adaptive_avg_pool2d(
            rearrange(x_proj, 'b t h w c -> (b t) c h w'),
            (H, W)
        )
        precision_features = rearrange(precision_features, '(b t) c h w -> b t h w c', b=B, t=T)

        # Apply STP blocks
        for block in self.blocks:
            space_features, time_features, precision_features = block(
                space_features, time_features, precision_features, timestamps
            )
        
        # Global context from space/time: GAP -> project -> broadcast
        # This avoids upsampling artifacts entirely
        space_global = space_features.mean(dim=(2, 3))          # (B, T, d_s)
        space_ctx = self.space_to_precision(space_global)        # (B, T, d_p)
        space_broadcast = space_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, self.precision_dim)
        
        time_global = time_features.mean(dim=(2, 3))            # (B, T, d_t)
        time_ctx = self.time_to_precision(time_global)           # (B, T, d_p)
        time_broadcast = time_ctx.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, self.precision_dim)
        
        # Combine: precision (spatial detail) + global context from space/time
        final_features = precision_features + space_broadcast + time_broadcast
        
        # Light spatial fusion
        final_2d = rearrange(final_features, 'b t h w c -> (b t) c h w')
        final_2d = self.spatial_fusion(final_2d)
        final_features = rearrange(final_2d, '(b t) c h w -> b t h w c', b=B, t=T)
        
        return self.norm(final_features)
