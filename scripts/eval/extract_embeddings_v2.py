#!/usr/bin/env python
"""提取所有 patch × 12 个月的 embedding（单月策略）。支持断点续传。

用法:
    python extract_embeddings_v2.py \
        --config configs/round4_8gpu/round4_full_vicreg_baseline.yaml \
        --checkpoint /workspace/outputs/.../epoch_best_xxx.pt \
        --output-dir /workspace/outputs/xxx/evaluation/embeddings \
        --device npu:0 \
        --batch-size 16 \
        --save-every 500
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import json
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch_npu

# 强制无缓冲输出
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.data.dataset import HarbinPatchDataset
from src.models.model import AEFModel
from src.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="训练配置文件路径")
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    p.add_argument("--output-dir", required=True, help="embedding 输出目录")
    p.add_argument("--device", default="npu:0", help="设备")
    p.add_argument("--batch-size", type=int, default=16, help="批大小")
    p.add_argument("--save-every", type=int, default=500, help="每 N 个样本保存一次断点")
    return p.parse_args()


def collate_batch(items: list[dict]) -> dict:
    """手动 batch 数据."""
    batch = {}
    for key in ["source_frames", "source_timestamps_ms", "source_frame_mask",
                "source_input_mask", "source_type_ids", "valid_start_ms",
                "valid_end_ms", "target_relative_time", "target_metadata"]:
        batch[key] = torch.stack([item[key] for item in items])
    batch["patch_id"] = [item["patch_id"] for item in items]
    batch["year_month"] = [item["year_month"] for item in items]
    return batch


def save_partial(output_dir: Path, global_mean: np.ndarray, spatial_maps: np.ndarray,
                 patch_ids: list, months: list, processed: int):
    """保存断点文件（原子写入：先写临时文件再重命名）."""
    partial_path = output_dir / "patch_embeddings_partial.npz"
    temp_path = output_dir / "patch_embeddings_partial_tmp.npz"
    np.savez(
        temp_path,
        global_mean=global_mean,
        spatial_maps=spatial_maps,
        patch_ids=np.array(patch_ids),
        months=np.array(months),
        processed=np.array(processed),
    )
    temp_path.rename(partial_path)
    print(f"      [断点] 已保存 {processed} 个样本到 partial 文件", flush=True)


def load_partial(output_dir: Path) -> tuple | None:
    """加载断点文件，返回 (global_mean, spatial_maps, patch_ids, months, processed)."""
    partial_path = output_dir / "patch_embeddings_partial.npz"
    if not partial_path.exists():
        return None
    data = np.load(partial_path)
    processed = int(data["processed"])
    print(f"      [断点续传] 发现 partial 文件，已处理 {processed} 个样本，跳过...", flush=True)
    return (
        data["global_mean"],
        data["spatial_maps"],
        data["patch_ids"].tolist(),
        data["months"].tolist(),
        processed,
    )


def extract_embeddings(cfg, checkpoint_path: str, output_dir: Path, device_str: str,
                       batch_size: int, save_every: int):
    device = torch.device(device_str)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    print("[1/4] 加载模型...")
    model = AEFModel(cfg).to(device)
    state_dict = load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 创建数据集（不预加载，快速启动）
    print("[2/4] 创建数据集...")
    cfg.data.preload = True
    dataset = HarbinPatchDataset(cfg)
    print(f"      数据集大小: {len(dataset)} 个样本")

    total = len(dataset)
    t0 = time.time()

    # 尝试加载断点
    partial = load_partial(output_dir)
    if partial is not None:
        all_global_mean = [partial[0]]
        all_spatial_maps = [partial[1]]
        all_patch_ids = partial[2]
        all_months = partial[3]
        processed = partial[4]
        start_idx = processed
    else:
        all_global_mean = []
        all_spatial_maps = []
        all_patch_ids = []
        all_months = []
        processed = 0
        start_idx = 0

    print(f"[3/4] 提取 embedding... (从 {start_idx}/{total} 开始)", flush=True)

    try:
        with torch.no_grad():
            batch_items = []
            for idx in range(start_idx, total):
                item = dataset[idx]
                batch_items.append(item)

                if len(batch_items) == batch_size or idx == total - 1:
                    batch = collate_batch(batch_items)

                    # forward
                    out = model(
                        source_frames=batch["source_frames"].to(device),
                        source_timestamps_ms=batch["source_timestamps_ms"].to(device),
                        source_frame_mask=batch["source_frame_mask"].to(device),
                        source_input_mask=batch["source_input_mask"].to(device),
                        source_type_ids=batch["source_type_ids"].to(device),
                        valid_start_ms=batch["valid_start_ms"].to(device),
                        valid_end_ms=batch["valid_end_ms"].to(device),
                        target_relative_time=batch["target_relative_time"].to(device),
                        target_metadata=batch["target_metadata"].to(device),
                        skip_decoder=True,
                    )

                    emb_map = out.embedding_map.cpu().numpy()  # [B, D, H, W]
                    emb_global = out.embedding.cpu().numpy()    # [B, D]

                    all_spatial_maps.append(emb_map)
                    all_global_mean.append(emb_global)

                    for pid, ym in zip(batch["patch_id"], batch["year_month"]):
                        all_patch_ids.append(pid)
                        month_val = int(ym[1]) if isinstance(ym, (tuple, list)) else int(ym)
                        all_months.append(month_val)

                    processed += len(batch_items)
                    batch_items = []

                    # 进度打印
                    if processed % 160 == 0 or idx == total - 1:
                        speed = processed / (time.time() - t0 + 1e-6)
                        print(f"      {processed}/{total}  ({speed:.1f} samples/sec)", flush=True)

                    # 定期保存断点 + 清理 NPU 缓存
                    if processed % save_every == 0 and processed > start_idx:
                        gm = np.concatenate(all_global_mean, axis=0)
                        sm = np.concatenate(all_spatial_maps, axis=0)
                        save_partial(output_dir, gm, sm, all_patch_ids, all_months, processed)
                        torch.npu.empty_cache()

            # 处理最后一批
            if batch_items:
                batch = collate_batch(batch_items)
                out = model(
                    source_frames=batch["source_frames"].to(device),
                    source_timestamps_ms=batch["source_timestamps_ms"].to(device),
                    source_frame_mask=batch["source_frame_mask"].to(device),
                    source_input_mask=batch["source_input_mask"].to(device),
                    source_type_ids=batch["source_type_ids"].to(device),
                    valid_start_ms=batch["valid_start_ms"].to(device),
                    valid_end_ms=batch["valid_end_ms"].to(device),
                    target_relative_time=batch["target_relative_time"].to(device),
                    target_metadata=batch["target_metadata"].to(device),
                    skip_decoder=True,
                )
                emb_map = out.embedding_map.cpu().numpy()
                emb_global = out.embedding.cpu().numpy()
                all_spatial_maps.append(emb_map)
                all_global_mean.append(emb_global)
                for pid, ym in zip(batch["patch_id"], batch["year_month"]):
                    all_patch_ids.append(pid)
                    month_val = int(ym[1]) if isinstance(ym, (tuple, list)) else int(ym)
                    all_months.append(month_val)
                processed += len(batch_items)

    except Exception as e:
        print(f"\n[ERROR] 提取过程中异常: {e}", flush=True)
        traceback.print_exc()
        if processed > start_idx:
            print(f"[INFO] 保存断点 ({processed}/{total})...", flush=True)
            gm = np.concatenate(all_global_mean, axis=0)
            sm = np.concatenate(all_spatial_maps, axis=0)
            save_partial(output_dir, gm, sm, all_patch_ids, all_months, processed)
        raise

    # 4. 整理并保存
    print("[4/4] 保存 embedding...")
    spatial_maps = np.concatenate(all_spatial_maps, axis=0)   # [N, D, H, W]
    global_mean = np.concatenate(all_global_mean, axis=0)     # [N, D]

    num_patches = len(dataset.patches)
    D = global_mean.shape[1]
    H, W = spatial_maps.shape[-2:]

    spatial_maps = spatial_maps.reshape(num_patches, 12, D, H, W)
    global_mean = global_mean.reshape(num_patches, 12, D)
    patch_ids_arr = np.array(all_patch_ids).reshape(num_patches, 12)[:, 0]
    months_arr = np.array(all_months).reshape(num_patches, 12)[0, :]

    np.savez(
        output_dir / "patch_embeddings.npz",
        global_mean=global_mean.astype(np.float32),
        spatial_maps=spatial_maps.astype(np.float32),
        patch_ids=patch_ids_arr,
        months=months_arr,
    )

    # 删除断点文件
    partial_path = output_dir / "patch_embeddings_partial.npz"
    if partial_path.exists():
        partial_path.unlink()

    elapsed = time.time() - t0
    print(f"      完成! 耗时: {elapsed:.1f}s")
    print(f"      输出: {output_dir / 'patch_embeddings.npz'}")
    print(f"      形状: global_mean={global_mean.shape}, spatial_maps={spatial_maps.shape}")

    # 保存元数据
    meta = {
        "num_patches": int(num_patches),
        "num_months": 12,
        "embedding_dim": int(D),
        "spatial_size": [int(H), int(W)],
        "checkpoint": checkpoint_path,
        "config": str(cfg.experiment.name),
        "elapsed_seconds": elapsed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    extract_embeddings(cfg, args.checkpoint, Path(args.output_dir), args.device,
                       args.batch_size, args.save_every)
