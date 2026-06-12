#!/usr/bin/env python3
"""冒烟测试 — 验证月度采样 + 模型前向 + 重建损失能跑通.

支持 legacy 和多分辨率配置（source_frames 为 Tensor 或 List[Tensor]）。
"""
from __future__ import annotations

import argparse
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
from src.config import load_config
from src.models.model import AEFModel
from src.data.builder import build_dataloader
from src.training.loops import compute_recon_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xuannv_v12_clean.yaml")
    args = parser.parse_args()

    print("=" * 60)
    print("Smoke Test")
    print("=" * 60)

    # 1. 加载配置
    cfg = load_config(args.config)
    cfg.data.preload = False
    cfg.data.max_patches = getattr(cfg.data, "max_patches", None) or 2
    print(f"\n[1/5] Config loaded: {args.config}")
    print(f"  Target sources: {len(cfg.data.target_sources)}")
    print(f"  Recon weights: {getattr(cfg.training, 'source_recon_weights', 'N/A')}")
    print(f"  use_multires: {getattr(cfg.data, 'use_multires', False)}")

    # 2. 创建数据集 / DataLoader
    loader = build_dataloader(cfg, training=False, distributed=False)
    item = next(iter(loader))
    print(f"\n[2/5] Dataset: 1 batch")
    print(f"  Patch: {item['patch_id']}, YearMonth: {item['year_month']}")
    if isinstance(item["source_frames"], list):
        print(f"  Source frames (list): {[f.shape for f in item['source_frames']]}")
        print(f"  Target images (list): {[t.shape for t in item['target_images']]}")
    else:
        print(f"  Source frames: {item['source_frames'].shape}")
        print(f"  Target images: {item['target_images'].shape}")
    print(f"  Target mask: {item['target_mask']}")

    # 3. 创建模型
    device = "npu:0"
    model = AEFModel(cfg).to(device)
    print(f"\n[3/5] Model created on {device}")

    # 4. 前向测试
    def to_device(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            return [x.to(device) for x in v]
        return v

    batch = {k: to_device(v) for k, v in item.items()}

    try:
        with torch.no_grad():
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
        print(f"\n[4/5] Forward pass OK")
        print(f"  Embedding map: {out.embedding_map.shape}")
        if isinstance(out.reconstructions, list):
            print(f"  Reconstructions (list): {[r.shape for r in out.reconstructions]}")
        else:
            print(f"  Reconstructions: {out.reconstructions.shape}")
        print(f"  Reconstruction mask: {batch['target_mask']}")
    except Exception as e:
        print(f"\n[4/5] Forward FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. 损失测试
    try:
        recon = compute_recon_loss(
            out.reconstructions,
            batch["target_images"],
            batch["target_mask"],
            batch.get("target_loss_type"),
            cfg.data.num_classes,
        )
        print(f"\n[5/5] Reconstruction loss: {recon.item():.4f}")
    except Exception as e:
        print(f"\n[5/5] Loss FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Smoke Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
