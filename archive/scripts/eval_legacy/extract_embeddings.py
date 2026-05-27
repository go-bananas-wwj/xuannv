#!/usr/bin/env python3
"""
从训练好的backbone提取所有patch、所有月份的embedding。

输出: /workspace/xuannv/data/embeddings/{exp_name}/
  - {patch_id}_{year}-{month:02d}.npy  → embedding_map [D, H, W]
  - metadata.json                       → 记录patch列表和月份列表
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import numpy as np
import torch
import torch_npu
from pathlib import Path
from tqdm import tqdm

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.utils.checkpoint import load_checkpoint


MONTHS = [(2025, 4), (2025, 6), (2025, 8), (2025, 9), (2025, 10)]


def month_to_window(year: int, month: int):
    """将年月转换为valid_start_ms, valid_end_ms"""
    import calendar
    import time
    start_sec = time.mktime((year, month, 1, 0, 0, 0, 0, 0, 0))
    last_day = calendar.monthrange(year, month)[1]
    end_sec = time.mktime((year, month, last_day, 23, 59, 59, 0, 0, 0))
    return int(start_sec * 1000), int(end_sec * 1000)


def extract_embeddings_for_experiment(
    checkpoint_path: str,
    config_path: str,
    output_dir: Path,
    device_str: str = "npu:0",
) -> dict:
    """为某个实验提取所有patch、所有月份的embedding"""

    device = torch.device(device_str)
    cfg = load_config(config_path)

    # 禁用preload以加速启动
    cfg.data.preload = False
    cfg.data.num_workers = 4

    dataset = HarbinPatchDataset(cfg=cfg)
    print(f"Dataset: {len(dataset.patches)} patches, {len(dataset.monthly_samples)} monthly samples")

    # 加载模型
    model = AEFModel(cfg=cfg).to(device)
    state_dict = load_checkpoint(checkpoint_path, device=device_str)
    model.load_state_dict(state_dict)
    model.eval()

    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "patch_ids": dataset.patches,
        "months": [f"{y}-{m:02d}" for y, m in MONTHS],
        "embedding_dim": cfg.model.embedding_dim,
    }

    # 按(pach_id, year, month)索引样本
    sample_map = {}
    for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
        sample_map[(pid, year, month)] = idx

    extracted_count = 0
    with torch.no_grad():
        for year, month in MONTHS:
            month_str = f"{year}-{month:02d}"
            print(f"\n  Extracting {month_str}...")

            valid_start, valid_end = month_to_window(year, month)

            for pid in tqdm(dataset.patches, desc=f"  {month_str}"):
                key = (pid, year, month)
                if key not in sample_map:
                    print(f"    Skip {pid}: no data for {month_str}")
                    continue

                idx = sample_map[key]
                try:
                    sample = dataset[idx]
                except Exception as e:
                    print(f"    Skip {pid}: {e}")
                    continue

                # 准备输入
                batch_dev = {}
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        batch_dev[k] = v.unsqueeze(0).to(device)

                # 前向传播
                out = model(
                    source_frames=batch_dev["source_frames"],
                    source_timestamps_ms=batch_dev["source_timestamps_ms"],
                    source_frame_mask=batch_dev["source_frame_mask"],
                    source_input_mask=batch_dev["source_input_mask"],
                    source_type_ids=batch_dev["source_type_ids"],
                    valid_start_ms=batch_dev["valid_start_ms"],
                    valid_end_ms=batch_dev["valid_end_ms"],
                    target_relative_time=batch_dev["target_relative_time"],
                    target_metadata=batch_dev["target_metadata"],
                )

                emb_map = out.embedding_map  # [1, D, H, W]
                emb_map = emb_map[0].cpu().numpy()  # [D, H, W]

                np.save(output_dir / f"{pid}_{month_str}.npy", emb_map)
                extracted_count += 1

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n✅ Embedding提取完成: {output_dir}")
    print(f"   Total files: {extracted_count}")
    return metadata


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Backbone checkpoint path")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="npu:0", help="Device")
    args = parser.parse_args()

    extract_embeddings_for_experiment(
        args.checkpoint,
        args.config,
        Path(args.output_dir),
        args.device,
    )


if __name__ == "__main__":
    main()
