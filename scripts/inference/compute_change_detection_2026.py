#!/usr/bin/env python3
"""计算哈尔滨 2026 年相邻月份间的变化检测分数.

输入:
  /workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026/{patch_id}_{month}.npy

输出:
  /workspace/outputs/aef_qwen_v5_mixed_scale/change_scores_2026/{patch_id}_{month_pair}.npy
  - change_score: [H, W] 每个像素的变化强度，值域 [0, 1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
from tqdm import tqdm

EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/change_scores_2026")

# 相邻月份对
MONTH_PAIRS = [
    ("2026-01", "2026-02"),
    ("2026-02", "2026-03"),
    ("2026-03", "2026-04"),
    ("2026-04", "2026-05"),
]


def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """计算两个 embedding map 的 cosine distance.
    
    Args:
        emb1, emb2: [D, H, W] numpy arrays，已 L2 normalized
    
    Returns:
        [H, W] change score，值域 [0, 1]，越大表示变化越强
    """
    # 确保是 float32
    e1 = emb1.astype(np.float32)
    e2 = emb2.astype(np.float32)
    
    # 计算 cosine similarity = dot(e1, e2) / (norm(e1) * norm(e2))
    # 由于已 normalized，直接计算点积
    cos_sim = np.sum(e1 * e2, axis=0)  # [H, W]
    
    # 裁剪到 [-1, 1] 避免数值误差
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    
    # change score: (1 - cos_sim) / 2，值域 [0, 1]
    change_score = (1.0 - cos_sim) / 2.0
    return change_score


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 从 meta.json 获取所有 patch
    meta_path = EMB_DIR / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        all_patches = sorted(set(r["patch_id"] for r in meta))
    else:
        # fallback: 从文件名解析
        all_patches = sorted(set(
            p.name.split("_")[0] + "_" + p.name.split("_")[1]
            for p in EMB_DIR.glob("patch_*.npy")
        ))
    
    print(f"Found {len(all_patches)} patches with embeddings")
    
    meta_records = []
    
    for pid in tqdm(all_patches, desc="Patches"):
        for m1, m2 in MONTH_PAIRS:
            emb1_path = EMB_DIR / f"{pid}_{m1}.npy"
            emb2_path = EMB_DIR / f"{pid}_{m2}.npy"
            
            if not emb1_path.exists() or not emb2_path.exists():
                continue
            
            try:
                emb1 = np.load(emb1_path)  # [D, H, W]
                emb2 = np.load(emb2_path)  # [D, H, W]
                
                change_score = compute_cosine_distance(emb1, emb2)
                
                out_path = OUTPUT_DIR / f"{pid}_{m1}_to_{m2}.npy"
                np.save(out_path, change_score)
                
                meta_records.append({
                    "patch_id": pid,
                    "month_pair": f"{m1}_to_{m2}",
                    "shape": change_score.shape,
                    "mean_score": float(change_score.mean()),
                    "max_score": float(change_score.max()),
                })
            except Exception as e:
                print(f"\n  Skip {pid} {m1}->{m2}: {e}")
    
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta_records, f, indent=2)
    
    print(f"\nDone. Computed {len(meta_records)} change maps to {OUTPUT_DIR}")
    
    # 打印全局统计
    if meta_records:
        all_means = [r["mean_score"] for r in meta_records]
        print(f"  Mean change score across all: {np.mean(all_means):.4f}")
        print(f"  Std change score: {np.std(all_means):.4f}")


if __name__ == "__main__":
    main()
