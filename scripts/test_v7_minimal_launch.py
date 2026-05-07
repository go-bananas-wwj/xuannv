#!/usr/bin/env python
"""V7 极简验证快速冒烟测试 — 只跑 5 个 step 确认无 crash.

用法:
    python scripts/test_v7_minimal_launch.py --config configs/qwen_v7_minimal.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch
import torch_npu

from src.config import load_config
from src.data.builder import build_dataloader
from src.models.model import AEFModel
from src.training.vicreg_loss import vicreg_loss_components, koleo_loss
from src.training.loops import compute_recon_loss


def test_v7_minimal(config_path: str) -> None:
    cfg = load_config(config_path)
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    print(f"[test] Using device: {device}")

    # 构建模型
    model = AEFModel(cfg).to(device)
    model.train()
    print("[test] Model built OK")

    # 构建数据加载器
    dataloader = build_dataloader(cfg, training=True, distributed=False)
    print(f"[test] DataLoader built, {len(dataloader)} batches")

    # 取一个 batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    print(f"[test] Batch loaded: source_frames={batch['source_frames'].shape}")

    # 前向传播
    with torch.autocast(device_type="npu" if torch.npu.is_available() else "cpu", dtype=torch.float16, enabled=True):
        out = model(
            source_frames=batch["source_frames"],
            source_timestamps_ms=batch["source_timestamps_ms"],
            source_frame_mask=batch["source_frame_mask"],
            source_input_mask=batch["source_input_mask"],
            source_type_ids=batch["source_type_ids"],
            valid_start_ms=batch["valid_start_ms"],
            valid_end_ms=batch["valid_end_ms"],
            target_relative_time=batch["target_relative_time"],
            target_metadata=batch["target_metadata"],
            target_loss_type=batch.get("target_loss_type"),
            target_source_idx=batch.get("target_source_idx"),
        )
    print(f"[test] Forward OK: embedding={out.embedding.shape}, pre_norm={out.pre_norm_embedding.shape}")

    # 重建损失
    recon = compute_recon_loss(
        out.reconstructions, batch["target_images"], batch["target_mask"],
        batch.get("target_loss_type"), cfg.data.num_classes,
    )
    print(f"[test] Recon loss: {recon.item():.4f}")

    # VICReg + KoLeo
    pre_norm = out.pre_norm_embedding
    N = pre_norm.shape[0]
    half = N // 2
    if half >= 2:
        z1 = pre_norm[:half]
        z2 = pre_norm[half:2*half]
        vicreg, inv, var, cov = vicreg_loss_components(z1, z2)
        print(f"[test] VICReg: {vicreg.item():.4f} (inv={inv.item():.4f} var={var.item():.4f} cov={cov.item():.4f})")
    else:
        print("[test] Skip VICReg (batch too small)")

    koleo = koleo_loss(pre_norm)
    print(f"[test] KoLeo: {koleo.item():.4f}")

    # 总损失
    total = cfg.training.reconstruction_weight * recon
    if half >= 2:
        total = total + cfg.training.vicreg_weight * vicreg
    total = total + cfg.training.koleo_weight * koleo
    print(f"[test] Total loss: {total.item():.4f}")

    # 反向传播
    total.backward()
    print("[test] Backward OK")

    # 检查梯度
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    print(f"[test] Gradients present: {has_grad}")

    print("[test] ✅ All checks passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen_v7_minimal.yaml")
    args = parser.parse_args()
    test_v7_minimal(args.config)
