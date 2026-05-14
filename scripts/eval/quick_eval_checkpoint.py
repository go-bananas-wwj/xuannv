#!/usr/bin/env python3
"""快速评估 checkpoint 的时间敏感性 AUC.

直接从 checkpoint 提取 N 个样本的 embedding，计算 temporal AUC。
支持 pre-norm 和 l2-norm 两种模式。
"""
import sys, os, argparse, time, random
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", choices=["pre-norm", "l2-norm"], default="pre-norm",
                        help="使用哪种 embedding 空间计算距离")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="随机采样的样本数")
    parser.add_argument("--gap-threshold", type=int, default=6,
                        help="时间间隔阈值（月）")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    print(f"加载配置: {args.config}")
    cfg = load_config(args.config)
    model = AEFModel(cfg).to(args.device)
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    n_total = len(dataset.monthly_samples)
    print(f"总样本数: {n_total}，本次提取: {args.n_samples}")

    random.seed(42)
    indices = list(range(n_total))
    random.shuffle(indices)
    use_indices = indices[:args.n_samples]

    # 收集 embedding
    embeddings = []
    patch_ids = []
    year_months = []

    print("提取 embeddings...")
    t0 = time.time()
    for loop_idx, idx in enumerate(use_indices):
        batch = dataset[idx]
        pid = batch["patch_id"]
        ym = batch["year_month"]

        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.unsqueeze(0).to(args.device)

        with torch.no_grad():
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

        if args.mode == "pre-norm":
            emb = out.pre_norm_embedding[0].cpu().numpy().astype(np.float32)
        else:
            emb = out.embedding[0].cpu().numpy().astype(np.float32)

        embeddings.append(emb)
        patch_ids.append(pid)
        year_months.append(ym)

        if (loop_idx + 1) % 50 == 0:
            print(f"  {loop_idx+1}/{args.n_samples} ({time.time()-t0:.1f}s)")

    embeddings = np.stack(embeddings, axis=0)
    print(f"提取完成，耗时 {time.time()-t0:.1f}s，shape={embeddings.shape}")

    # 构建 pairs 并计算 AUC
    from collections import defaultdict
    patch_series = defaultdict(list)
    for i, (pid, ym) in enumerate(zip(patch_ids, year_months)):
        patch_series[pid].append((ym[0], ym[1], embeddings[i]))

    for pid in patch_series:
        patch_series[pid].sort(key=lambda x: (x[0], x[1]))

    distances = []
    labels = []
    gap_sizes = []

    for pid, series in patch_series.items():
        n = len(series)
        for i in range(n):
            for j in range(i + 1, n):
                y1, m1, emb1 = series[i]
                y2, m2, emb2 = series[j]
                month_gap = (y2 - y1) * 12 + (m2 - m1)
                if month_gap < 1:
                    continue

                dist = np.linalg.norm(emb1 - emb2)
                distances.append(dist)
                gap_sizes.append(month_gap)
                labels.append(1 if month_gap >= args.gap_threshold else 0)

    distances = np.array(distances)
    labels = np.array(labels)

    result = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "n_samples": args.n_samples,
        "gap_threshold": args.gap_threshold,
        "n_pairs": len(distances),
        "n_pos": int(labels.sum()),
        "n_neg": int((1-labels).sum()),
    }

    if len(np.unique(labels)) == 2:
        auc = roc_auc_score(labels, distances)
        result["auc"] = float(auc)

        d0 = distances[labels == 0]
        d1 = distances[labels == 1]
        result["dist_mean_neg"] = float(d0.mean())
        result["dist_mean_pos"] = float(d1.mean())
        result["dist_median_neg"] = float(np.median(d0))
        result["dist_median_pos"] = float(np.median(d1))
        result["separation"] = float(d1.mean() - d0.mean())

        print(f"\n{'='*50}")
        print(f"Mode: {args.mode}")
        print(f"AUC = {auc:.4f}")
        print(f"Pairs: total={len(distances)}, pos={result['n_pos']}, neg={result['n_neg']}")
        print(f"Distance: neg_mean={d0.mean():.4f}, pos_mean={d1.mean():.4f}")
        print(f"Separation: {result['separation']:.4f}")
        print(f"{'='*50}")
    else:
        result["auc"] = None
        print(f"\nWarning: 只有一类标签，无法计算 AUC")

    # Per-gap stats
    unique_gaps = sorted(set(gap_sizes))
    result["per_gap"] = []
    for g in unique_gaps[:12]:
        d_g = [d for d, gs in zip(distances, gap_sizes) if gs == g]
        if d_g:
            result["per_gap"].append({
                "gap": g,
                "n": len(d_g),
                "mean": float(np.mean(d_g)),
                "median": float(np.median(d_g)),
            })

    import json
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n结果保存: {args.output}")


if __name__ == "__main__":
    main()
