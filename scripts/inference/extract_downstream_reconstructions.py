#!/usr/bin/env python3
"""批量提取下游任务重建结果，保存为npy文件.

对每个 patch × 月份，分别提取:
- DEM 重建 (连续值)
- WorldCover 分类 (argmax)
- Dynamic World 分类 (argmax)
- JRC Water 重建 (连续值)
- S2 RGB 重建 (连续值)

输出:
  /workspace/outputs/aef_qwen_v5_mixed_scale/downstream_recon_2026/{target_name}/{patch_id}_{month}.npy
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
from tqdm import tqdm

from src.inference.engine import load_backbone
from src.utils.device import get_device

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v5_harbin_inference.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/epoch_best_epoch161.pt"
OUTPUT_ROOT = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/downstream_recon_2026")

MONTHLY_WINDOWS = {
    "2026-01": (1735689600000.0, 1738368000000.0),
    "2026-02": (1738368000000.0, 1740787200000.0),
    "2026-03": (1740787200000.0, 1743465600000.0),
    "2026-04": (1743465600000.0, 1746057600000.0),
    "2026-05": (1746057600000.0, 1748736000000.0),
}

# 目标源配置: (名称, source_idx, 是否为分类, 可视化通道数)
TARGET_CONFIGS = [
    ("dem", 3, False, 1),
    ("worldcover", 4, True, 1),
    ("dynamic_world", 5, True, 1),
    ("jrc_water", 6, False, 1),
    ("s2_recon", 0, False, 3),
]


def extract_reconstruction(model, dataset, pidx, ws, we, device, target_source_idx):
    """为指定 patch 和时间窗口提取指定目标源的重建结果."""
    batch = dataset[pidx]
    batch["valid_start_ms"] = torch.tensor(ws, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(we, dtype=torch.float64)

    # 构造 target tensors
    num_tgt = len(TARGET_CONFIGS)
    meta_dim = getattr(dataset.cfg.data, "metadata_dim", 4)
    B = 1

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)
        else:
            batch_dev[k] = v

    target_relative_time = torch.zeros(B, 1, device=device)
    target_metadata = torch.zeros(B, 1, meta_dim, device=device)
    target_source_idx_t = torch.tensor([target_source_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(
            source_frames=batch_dev["source_frames"],
            source_timestamps_ms=batch_dev["source_timestamps_ms"],
            source_frame_mask=batch_dev["source_frame_mask"],
            source_input_mask=batch_dev["source_input_mask"],
            source_type_ids=batch_dev["source_type_ids"],
            valid_start_ms=batch_dev["valid_start_ms"],
            valid_end_ms=batch_dev["valid_end_ms"],
            target_relative_time=target_relative_time,
            target_metadata=target_metadata,
            target_source_idx=target_source_idx_t,
        )

    # reconstructions: [B, T_tgt=1, C, H, W]
    recon = output.reconstructions[0, 0].cpu().numpy()  # [C, H, W]
    return recon


def main():
    device = get_device()
    model, dataset, cfg = load_backbone(CONFIG_PATH, CKPT_PATH, device=device)

    print(f"Loaded {len(dataset.patches)} patches")
    print(f"Target configs: {[t[0] for t in TARGET_CONFIGS]}")

    # 创建输出目录
    for name, _, _, _ in TARGET_CONFIGS:
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)

    total = len(dataset.patches) * len(MONTHLY_WINDOWS) * len(TARGET_CONFIGS)
    pbar = tqdm(total=total, desc="Extracting reconstructions")

    for pid in dataset.patches:
        pidx = dataset.patches.index(pid)
        for month, (ws, we) in MONTHLY_WINDOWS.items():
            for name, src_idx, is_cls, _ in TARGET_CONFIGS:
                out_path = OUTPUT_ROOT / name / f"{pid}_{month}.npy"
                if out_path.exists():
                    pbar.update(1)
                    continue

                try:
                    recon = extract_reconstruction(model, dataset, pidx, ws, we, device, src_idx)

                    if is_cls:
                        # 分类任务: argmax 获取类别图
                        recon = np.argmax(recon, axis=0)  # [H, W]
                    else:
                        # 连续任务: 取所需通道
                        if name == "s2_recon":
                            recon = recon[[2, 1, 0], :, :]  # RGB
                            recon = np.clip(recon, 0, 1)
                        elif recon.shape[0] == 1:
                            recon = recon[0]  # [H, W]

                    np.save(out_path, recon)
                except Exception as e:
                    tqdm.write(f"  Skip {pid} {month} {name}: {e}")

                pbar.update(1)

    pbar.close()
    print(f"\nDone. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
