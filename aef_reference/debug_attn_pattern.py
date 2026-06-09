"""检查大 dim 下 attention 矩阵的结构"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from einops import rearrange
from src.aef.architecture.stp_operators import SpaceOperator

torch.manual_seed(42)

def analyze_attn(dim, num_heads, H, W):
    x = torch.randn(1, 1, H, W, dim).npu()
    op = SpaceOperator(dim=dim, num_heads=num_heads).npu()
    op.eval()
    
    with torch.no_grad():
        x_flat = rearrange(x, 'b t h w c -> (b t) (h w) c')
        x_norm = op.norm1(x_flat)
        qkv = op.qkv(x_norm)
        qkv = rearrange(qkv, 'bt hw (three heads d) -> three bt heads hw d',
                       three=3, heads=op.num_heads, d=op.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_logits = (q @ k.transpose(-2, -1)) * (op.head_dim ** -0.5)
        attn = torch.softmax(attn_logits, dim=-1)
    
    attn_np = attn.cpu().numpy()[0]  # (heads, HW, HW)
    
    print(f"\n=== dim={dim}, heads={num_heads}, H={H}, W={W} ===")
    for hi in range(min(4, num_heads)):
        a = attn_np[hi]  # (HW, HW)
        # Check diagonal dominance
        diag_mean = np.diag(a).mean()
        offdiag_mean = (a.sum() - np.diag(a).sum()) / (a.shape[0] * (a.shape[0] - 1))
        print(f"  Head {hi}: diag_mean={diag_mean:.4f}, offdiag_mean={offdiag_mean:.4f}")
    
    # Overall stats
    print(f"  All heads: max={attn_np.max():.4f}, min={attn_np.min():.4f}, std={attn_np.std():.4f}")
    
    # Entropy
    entropy = -np.mean(attn_np * np.log(attn_np + 1e-10))
    max_entropy = np.log(H * W)
    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}% of max)")

for dim in [64, 256, 512, 1024]:
    analyze_attn(dim=dim, num_heads=8, H=16, W=16)
