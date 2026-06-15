#!/usr/bin/env python3
"""为海淀标注任务提取两时相 embedding.

输入: 63 个 labeljson 中的 patch, 取 2025-12 与 2026-04-30 两个月份.
输出: npz 文件, 包含 patch_ids / emb_dec / emb_apr.

用法:
    python scripts/eval/extract_haidian_embeddings.py \
        --config configs/config_multires_v2d.yaml \
        --checkpoint /workspace/xuannv/outputs/exp_multires_v2d_quick_0614/epoch_53.pt \
        --output /workspace/xuannv/out/haidian_label/embeddings_v2d53.npz \
        --device npu:0
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.engine import load_backbone, extract_embedding_for_month


def discover_labeled_patches(label_dir: str) -> list[str]:
    """从 labeljson 目录发现所有 patch_id."""
    files = glob.glob(os.path.join(label_dir, "*.json"))
    pids = set()
    for f in files:
        m = re.search(r"(patch_\d+)", os.path.basename(f))
        if m:
            pids.add(m.group(1))
    return sorted(pids)


def main() -> int:
    parser = argparse.ArgumentParser(description="提取海淀标注任务两时相 embedding")
    parser.add_argument("--config", required=True, help="模型配置文件")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--output", default="/workspace/xuannv/out/haidian_label/embeddings_v2d53.npz",
                        help="输出 npz 路径")
    parser.add_argument("--device", default="npu:0", help="运行设备")
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson",
                        help="标注 JSON 目录")
    parser.add_argument("--dec-year", type=int, default=2025, help="before 时相年份")
    parser.add_argument("--dec-month", type=int, default=12, help="before 时相月份")
    parser.add_argument("--apr-year", type=int, default=2026, help="after 时相年份")
    parser.add_argument("--apr-month", type=int, default=4, help="after 时相月份")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patch_ids = discover_labeled_patches(args.label_dir)
    print(f"[Extract] 发现 {len(patch_ids)} 个标注 patch")

    print("[Extract] 加载模型...")
    model, dataset, cfg = load_backbone(args.config, args.checkpoint, device=args.device, eval_mode=True)
    device = torch.device(args.device)

    emb_dec_list: list[np.ndarray] = []
    emb_apr_list: list[np.ndarray] = []
    valid_pids: list[str] = []

    for pid in patch_ids:
        try:
            emb_dec = extract_embedding_for_month(
                model, dataset, pid, args.dec_year, args.dec_month, device, normalize=True
            )
            emb_apr = extract_embedding_for_month(
                model, dataset, pid, args.apr_year, args.apr_month, device, normalize=True
            )
        except Exception as e:
            print(f"[Extract] {pid} 提取失败，跳过: {e}")
            continue

        if emb_dec.shape != emb_apr.shape:
            print(f"[Extract] {pid} 两时相 shape 不一致 {emb_dec.shape} vs {emb_apr.shape}，跳过")
            continue

        emb_dec_list.append(emb_dec)
        emb_apr_list.append(emb_apr)
        valid_pids.append(pid)
        print(f"[Extract] {pid} 完成，shape={emb_dec.shape}")

    if not valid_pids:
        print("[Extract] 没有成功提取任何 patch")
        return 1

    emb_dec_arr = np.stack(emb_dec_list, axis=0)  # [N, D, H, W]
    emb_apr_arr = np.stack(emb_apr_list, axis=0)
    patch_ids_arr = np.array(valid_pids)

    np.savez_compressed(
        out_path,
        patch_ids=patch_ids_arr,
        emb_dec=emb_dec_arr,
        emb_apr=emb_apr_arr,
    )
    print(f"[Extract] 保存到 {out_path}，N={len(valid_pids)}, shape={emb_dec_arr.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
