import torch
from torch import nn
from torch.functional import F
from typing import Tuple
from einops import rearrange

from src.aef.architecture.stp_operators import SpaceOperator, PrecisionOperator, TimeOperator


class STPBlock(nn.Module):
    """Single STP block with three simultaneous operators and global context exchange.
    
    Key change: pyramid exchanges now use GAP + broadcast instead of learned
    upsampling, eliminating grid/checkerboard artifacts from bilinear upsampling
    of coarse feature maps inside the block loop.
    """
    
    def __init__(self, space_dim: int = 1024, time_dim: int = 512, precision_dim: int = 128):
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        
        self.space_op = SpaceOperator(self.space_dim)
        self.time_op = TimeOperator(self.time_dim)
        self.precision_op = PrecisionOperator(self.precision_dim)
        
        # Global context projections for cross-scale exchange
        # space -> precision, time -> precision, etc.
        self.space_to_precision_proj = nn.Linear(space_dim, precision_dim)
        self.time_to_precision_proj = nn.Linear(time_dim, precision_dim)
        self.precision_to_space_proj = nn.Linear(precision_dim, space_dim)
        self.precision_to_time_proj = nn.Linear(precision_dim, time_dim)
        self.space_to_time_proj = nn.Linear(space_dim, time_dim)
        self.time_to_space_proj = nn.Linear(time_dim, space_dim)
        
    def forward(self, space_x: torch.Tensor, time_x: torch.Tensor, precision_x: torch.Tensor, 
                timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Apply operators
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps)
        precision_out = self.precision_op(precision_x)
        
        B, T = space_out.shape[:2]
        
        # --- Global context exchange (no upsampling, no grid artifacts) ---
        
        # Space global context
        space_global = space_out.mean(dim=(2, 3))  # (B, T, space_dim)
        
        # Time global context
        time_global = time_out.mean(dim=(2, 3))    # (B, T, time_dim)
        
        # Precision global context
        precision_global = precision_out.mean(dim=(2, 3))  # (B, T, precision_dim)
        
        # Exchange via projection + broadcast
        # space contributes to time and precision
        space_to_time = self.space_to_time_proj(space_global).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,time_dim)
        space_to_precision = self.space_to_precision_proj(space_global).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,prec_dim)
        
        # time contributes to space and precision
        time_to_space = self.time_to_space_proj(time_global).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,space_dim)
        time_to_precision = self.time_to_precision_proj(time_global).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,prec_dim)
        
        # precision contributes to space and time
        precision_to_space = self.precision_to_space_proj(precision_global).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,space_dim)
        precision_to_time = self.precision_to_time_proj(precision_global).unsqueeze(2).unsqueeze(3)  # (B,T,1,1,time_dim)
        
        # Broadcast to spatial dimensions and add
        space_H, space_W = space_out.shape[2:4]
        time_H, time_W = time_out.shape[2:4]
        precision_H, precision_W = precision_out.shape[2:4]
        
        space_exchange = space_out + time_to_space.expand(B, T, space_H, space_W, self.space_dim) \
                                   + precision_to_space.expand(B, T, space_H, space_W, self.space_dim)
        
        time_exchange = time_out + space_to_time.expand(B, T, time_H, time_W, self.time_dim) \
                                 + precision_to_time.expand(B, T, time_H, time_W, self.time_dim)
        
        precision_exchange = precision_out + space_to_precision.expand(B, T, precision_H, precision_W, self.precision_dim) \
                                           + time_to_precision.expand(B, T, precision_H, precision_W, self.precision_dim)
        
        return space_exchange, time_exchange, precision_exchange
