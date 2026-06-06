"""冒烟测试 — 验证模型前向+mask+损失能跑通."""
from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from haidian_recon.models.hre_model import HREModel
from haidian_recon.data.masking import FourLayerMask
from haidian_recon.losses.reconstruction import reconstruction_loss
from haidian_recon.losses.uniformity import uniformity_loss


def test_smoke():
    print("=== HRE Smoke Test ===")

    # 配置
    source_channels = {
        "tianyi_sar": 1,
        "s2": 6,
        "landsat": 6,
        "planet": 4,
    }
    B = 2
    T = 1
    H = W = 128

    # 1. 构建模型
    print("[1/6] Building model...")
    model = HREModel(
        source_channels=source_channels,
        image_size=128,
        patch_size=8,
        embed_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=1,
        num_heads=8,
        output_dim=64,
        use_gradient_checkpointing=False,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 2. 构建虚拟batch [B, T, C, H, W]
    print("[2/6] Creating dummy batch...")
    batch = {}
    for name, ch in source_channels.items():
        batch[name] = torch.randn(B, T, ch, H, W)
        batch[f"{name}_valid"] = torch.ones(B, dtype=torch.bool)

    # 3. 应用mask
    print("[3/6] Applying four-layer mask...")
    masking = FourLayerMask(
        source_names=list(source_channels.keys()),
        image_size=128,
        patch_size=8,
    )
    masked_batch, mask_info = masking(batch)

    print(f"  Modality roles: {mask_info['modality_roles']}")
    print(f"  Encode sources: {mask_info['encode_sources']}")
    print(f"  Decode sources: {mask_info['decode_sources']}")

    # 4. 前向传播
    print("[4/6] Forward pass...")
    output = model(masked_batch, mask_info)
    print(f"  Embedding shape: {output['embedding'].shape}")
    print(f"  Reconstructions: {list(output['reconstructions'].keys())}")
    for k, v in output["reconstructions"].items():
        print(f"    {k}: {v.shape}")

    # 5. 计算损失
    print("[5/6] Computing losses...")
    loss_recon = reconstruction_loss(output["reconstructions"], batch, mask_info)
    loss_uniform = uniformity_loss(output["embedding"])
    print(f"  Recon loss: {loss_recon.item():.4f}")
    print(f"  Uniform loss: {loss_uniform.item():.4f}")

    # 6. 反向传播
    print("[6/6] Backward pass...")
    loss = loss_recon + 0.01 * loss_uniform
    loss.backward()
    print("  Backward OK")

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Has gradient: {has_grad}")

    print("\n=== All smoke tests passed! ===")


if __name__ == "__main__":
    test_smoke()
