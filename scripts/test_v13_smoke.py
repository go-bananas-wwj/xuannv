#!/usr/bin/env python3
"""V13 冒烟测试 — 验证月度采样 + 无时间条件Decoder能否跑通."""
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

print("="*60)
print("V13 Smoke Test")
print("="*60)

# 1. 加载配置
cfg = load_config("configs/xuannv_v12_clean.yaml")
cfg.data.preload = False
print(f"\n[1/5] Config loaded")
print(f"  Target sources: {len(cfg.data.target_sources)}")
print(f"  Recon weights: {getattr(cfg.training, 'source_recon_weights', 'N/A')}")

# 2. 创建数据集
ds = HarbinPatchDataset(cfg)
print(f"\n[2/5] Dataset: {len(ds)} monthly samples")

item = ds[0]
print(f"  Patch: {item['patch_id']}, YearMonth: {item['year_month']}")
print(f"  Source frames: {item['source_frames'].shape}")
print(f"  Target images: {item['target_images'].shape}")
print(f"  Target mask: {item['target_mask']}")

# 3. 创建模型
device = "npu:0"
model = AEFModel(cfg).to(device)
print(f"\n[3/5] Model created on {device}")

# 4. 前向测试
batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
         for k, v in item.items()}

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
    print(f"  Reconstructions: {out.reconstructions.shape}")
    print(f"  Reconstruction mask: {batch['target_mask']}")
except Exception as e:
    print(f"\n[4/5] Forward FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 损失测试
from src.training.losses import reconstruction_loss

try:
    recon = reconstruction_loss(out.reconstructions, batch["target_images"], batch["target_mask"])
    print(f"\n[5/5] Reconstruction loss: {recon.item():.4f}")
except Exception as e:
    print(f"\n[5/5] Loss FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("  V13 Smoke Test PASSED")
print("="*60)
