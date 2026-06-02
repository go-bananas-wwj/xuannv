#!/usr/bin/env python3
"""
多卡并行提取 Embedding — 8 NPU 同时处理.

用法:
    python scripts/eval/extract_embeddings_multigpu.py \
        --checkpoint /workspace/outputs/xuannv_backbone_v10_temporal/epoch_100.pt \
        --config configs/xuannv_v10_temporal.yaml \
        --output /workspace/outputs/v10_eval_e100/embeddings.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)


def extract_worker(gpu_id, patch_indices, checkpoint, config_path, output_path):
    """单个 NPU worker: 处理分配到的 patches."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"npu:0")  # 绑定到唯一的可见 NPU
    
    print(f"[GPU {gpu_id}] Loading model...")
    cfg = load_config(config_path)
    cfg.data.preload = False
    cfg.data.manifest_path = "/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered"
    
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    
    results = {}
    
    for idx in patch_indices:
        patch_id = f"patch_{idx:06d}"
        try:
            sample = dataset[idx]
            source_frames = sample["source_frames"].unsqueeze(0).to(device)
            source_timestamps_ms = sample["source_timestamps_ms"].unsqueeze(0).to(device)
            source_frame_mask = sample["source_frame_mask"].unsqueeze(0).to(device)
            source_input_mask = sample["source_input_mask"].unsqueeze(0).to(device)
            source_type_ids = sample["source_type_ids"].unsqueeze(0).to(device)
            
            # Full embedding
            with torch.no_grad():
                out = model(
                    source_frames=source_frames,
                    source_timestamps_ms=source_timestamps_ms,
                    source_frame_mask=source_frame_mask,
                    source_input_mask=source_input_mask,
                    source_type_ids=source_type_ids,
                    valid_start_ms=source_timestamps_ms.min(),
                    valid_end_ms=source_timestamps_ms.max(),
                    target_relative_time=torch.zeros(1, 1, device=device),
                    target_metadata=torch.zeros(1, 1, cfg.data.metadata_dim, device=device),
                )
            full_emb = F.normalize(out.embedding[0], dim=0).cpu().numpy()
            
            # Before window
            emb_before = _extract_window(
                model, source_frames, source_timestamps_ms, source_frame_mask,
                source_input_mask, source_type_ids, BEFORE_WINDOW, device, cfg.data.metadata_dim
            )
            
            # After window
            emb_after = _extract_window(
                model, source_frames, source_timestamps_ms, source_frame_mask,
                source_input_mask, source_type_ids, AFTER_WINDOW, device, cfg.data.metadata_dim
            )
            
            results[patch_id] = {
                'full': full_emb,
                'before': emb_before if emb_before is not None else np.zeros_like(full_emb),
                'after': emb_after if emb_after is not None else np.zeros_like(full_emb),
            }
        except Exception as e:
            print(f"[GPU {gpu_id}] Error {patch_id}: {e}")
            continue
        
        if len(results) % 10 == 0:
            print(f"[GPU {gpu_id}] {len(results)}/{len(patch_indices)} done")
    
    # 保存部分结果
    np.savez(output_path, **results)
    print(f"[GPU {gpu_id}] Saved {len(results)} embeddings to {output_path}")


@torch.no_grad()
def _extract_window(model, source_frames, source_timestamps_ms, source_frame_mask,
                     source_input_mask, source_type_ids, window, device, metadata_dim):
    valid_start, valid_end = window
    frame_mask = source_frame_mask.clone()
    B, S, T = frame_mask.shape
    
    for b in range(B):
        for s in range(S):
            for t in range(T):
                ts = source_timestamps_ms[b, s, t].item()
                if ts < valid_start or ts > valid_end:
                    frame_mask[b, s, t] = False
    
    if not frame_mask.any():
        return None
    
    out = model(
        source_frames=source_frames,
        source_timestamps_ms=source_timestamps_ms,
        source_frame_mask=frame_mask,
        source_input_mask=source_input_mask,
        source_type_ids=source_type_ids,
        valid_start_ms=source_timestamps_ms.min(),
        valid_end_ms=source_timestamps_ms.max(),
        target_relative_time=torch.zeros(1, 1, device=device),
        target_metadata=torch.zeros(1, 1, metadata_dim, device=device),
    )
    return F.normalize(out.embedding[0], dim=0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/xuannv_v10_temporal.yaml")
    parser.add_argument("--output", default="/workspace/outputs/v10_eval_e100/embeddings.npz")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集获取总 patches 数
    cfg = load_config(args.config)
    cfg.data.preload = False
    cfg.data.manifest_path = "/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered"
    dataset = HarbinPatchDataset(cfg)
    total_patches = len(dataset)
    
    print(f"Total patches: {total_patches}, GPUs: {args.num_gpus}")
    
    # 分配 patches 到各 GPU
    patch_lists = [[] for _ in range(args.num_gpus)]
    for i in range(total_patches):
        patch_lists[i % args.num_gpus].append(i)
    
    for i, lst in enumerate(patch_lists):
        print(f"  GPU {i}: {len(lst)} patches")
    
    # 启动多进程
    processes = []
    for gpu_id in range(args.num_gpus):
        if not patch_lists[gpu_id]:
            continue
        part_path = str(output_dir / f"embeddings_part{gpu_id}.npz")
        p = Process(target=extract_worker, args=(
            gpu_id, patch_lists[gpu_id], args.checkpoint, args.config, part_path
        ))
        p.start()
        processes.append((gpu_id, p, part_path))
    
    # 等待所有进程完成
    for gpu_id, p, part_path in processes:
        p.join()
        print(f"[GPU {gpu_id}] Finished")
    
    # 合并结果
    print("\nMerging results...")
    all_embeddings = {}
    for gpu_id, p, part_path in processes:
        if Path(part_path).exists():
            data = np.load(part_path)
            for key in data.files:
                all_embeddings[key] = {
                    'full': data[key][0],
                    'before': data[key][1],
                    'after': data[key][2],
                }
    
    # 保存合并结果
    np.savez(args.output, **{
        k: np.stack([v['full'], v['before'], v['after']])
        for k, v in all_embeddings.items()
    })
    print(f"Saved {len(all_embeddings)} embeddings to {args.output}")
    
    # 清理临时文件
    for gpu_id, p, part_path in processes:
        Path(part_path).unlink(missing_ok=True)
    
    print("Done!")


if __name__ == "__main__":
    main()
