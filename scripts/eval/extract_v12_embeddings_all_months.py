#!/usr/bin/env python3
"""提取 V12 模型所有 patch 所有月份的 embedding_map 和 pre_norm_map.

输出: /workspace/outputs/{exp_name}/eval/embeddings_all_months/{patch_id}.npz
每个 npz 包含:
  - embedding_map: [N_months, D, H, W]  (L2 normalized, 推理空间)
  - pre_norm_map:  [N_months, D, H, W]  (训练空间，原始幅度)
  - year_months:   [N_months, 2]         int, e.g. [[2023, 1], ...]
  - valid_starts:  [N_months]            float64 ms
  - valid_ends:    [N_months]            float64 ms
"""
import sys, os, argparse, time
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch.utils.data import DataLoader


def collate_fn(batch):
    """Batch dict of tensors, leave str/list as list."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    # ── 加载模型 ──
    print(f"加载配置: {args.config}")
    cfg = load_config(args.config)
    model = AEFModel(cfg).to(args.device)
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 创建 Dataset / DataLoader ──
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    print(f"总月度样本数: {len(dataset.monthly_samples)}")
    print(f"Patches: {len(dataset.patches)}")
    print(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # ── 收集每个 patch 的月度 embeddings ──
    patch_data = {}  # patch_id -> list of dicts

    print("开始提取 embeddings...")
    t0 = time.time()
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            patch_ids = batch["patch_id"]
            year_months = batch["year_month"]  # list of (y, m) tuples
            valid_start = batch["valid_start_ms"].to(args.device)
            valid_end = batch["valid_end_ms"].to(args.device)

            # Forward
            output = model(
                source_frames=batch["source_frames"].to(args.device),
                source_timestamps_ms=batch["source_timestamps_ms"].to(args.device),
                source_frame_mask=batch["source_frame_mask"].to(args.device),
                source_input_mask=batch["source_input_mask"].to(args.device),
                source_type_ids=batch["source_type_ids"].to(args.device),
                valid_start_ms=valid_start,
                valid_end_ms=valid_end,
                target_relative_time=batch["target_relative_time"].to(args.device),
                target_metadata=batch["target_metadata"].to(args.device),
            )

            emb_map = output.embedding_map.cpu()      # [B, D, H, W]
            pre_map = output.pre_norm_map.cpu() if output.pre_norm_map is not None else None

            B = emb_map.shape[0]
            for i in range(B):
                pid = patch_ids[i]
                ym = year_months[i]  # tuple (year, month)
                if pid not in patch_data:
                    patch_data[pid] = []
                entry = {
                    "year_month": ym,
                    "valid_start": float(valid_start[i].cpu()),
                    "valid_end": float(valid_end[i].cpu()),
                    "embedding_map": emb_map[i].numpy().astype(np.float32),
                    "pre_norm_map": pre_map[i].numpy().astype(np.float32) if pre_map is not None else None,
                }
                patch_data[pid].append(entry)

            n_batches += 1
            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                print(f"  已处理 {n_batches * args.batch_size}/{len(dataset.monthly_samples)} 样本 "
                      f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n提取完成，耗时 {elapsed:.1f}s，共 {len(patch_data)} patches")

    # ── 保存为 per-patch npz ──
    print(f"保存到: {args.output_dir}")
    saved = 0
    for pid, entries in patch_data.items():
        entries.sort(key=lambda e: (e["year_month"][0], e["year_month"][1]))
        n_months = len(entries)
        D, H, W = entries[0]["embedding_map"].shape

        emb_stack = np.stack([e["embedding_map"] for e in entries], axis=0)  # [N, D, H, W]
        pre_stack = np.stack([e["pre_norm_map"] for e in entries], axis=0) if entries[0]["pre_norm_map"] is not None else None
        year_months = np.array([e["year_month"] for e in entries], dtype=np.int32)
        valid_starts = np.array([e["valid_start"] for e in entries], dtype=np.float64)
        valid_ends = np.array([e["valid_end"] for e in entries], dtype=np.float64)

        np.savez(
            os.path.join(args.output_dir, f"{pid}.npz"),
            embedding_map=emb_stack,
            pre_norm_map=pre_stack,
            year_months=year_months,
            valid_starts=valid_starts,
            valid_ends=valid_ends,
        )
        saved += 1
        if saved % 50 == 0:
            print(f"  已保存 {saved}/{len(patch_data)} patches")

    print(f"\n全部完成，保存 {saved} 个 npz 文件到 {args.output_dir}")

    # 打印一个样本的 shape 信息
    sample = next(iter(patch_data.values()))
    print(f"\n样本 shape 信息:")
    print(f"  embedding_map: {sample[0]['embedding_map'].shape}")
    print(f"  pre_norm_map:  {sample[0]['pre_norm_map'].shape if sample[0]['pre_norm_map'] is not None else 'None'}")
    print(f"  平均每 patch 月份数: {sum(len(v) for v in patch_data.values()) / len(patch_data):.1f}")


if __name__ == "__main__":
    main()
