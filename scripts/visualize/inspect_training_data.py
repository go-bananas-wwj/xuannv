#!/usr/bin/env python3
"""可视化训练数据 — 随机抽多个 patch，展示所有输入/目标/窗口信息.

用法:
    cd /workspace/xuannv
    python scripts/visualize/inspect_training_data.py \
        --config configs/qwen_v4_cd_upgrade.yaml \
        --output /workspace/outputs/aef_qwen_v4_cd_upgrade/training_data_inspection.png \
        --num-patches 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.data.builder import build_dataloader


def timestamp_to_str(ts_ms: float) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000.0)
    return dt.strftime("%Y-%m-%d")


def plot_one_patch(batch_item: dict, fig, gs_row: int, input_sources: list, target_sources: list):
    """在指定 grid row 绘制单个 patch 的所有信息."""
    patch_id = batch_item["patch_id"]
    if isinstance(patch_id, torch.Tensor):
        patch_id = str(patch_id.item())
    else:
        patch_id = str(patch_id)

    source_frames = batch_item["source_frames"].numpy()       # (S_inp, T, C, H, W)
    source_ts = batch_item["source_timestamps_ms"].numpy()    # (S_inp, T)
    source_mask = batch_item["source_frame_mask"].numpy()     # (S_inp, T)
    source_input_mask = batch_item["source_input_mask"].numpy()  # (S_inp,)

    target_images = batch_item["target_images"].numpy()       # (S_tgt, C, H//2, W//2)
    target_mask = batch_item["target_mask"].numpy()           # (S_tgt,)
    target_loss_type = batch_item["target_loss_type"].numpy()

    w1_start = batch_item["valid_start_w1"].item()
    w1_end = batch_item["valid_end_w1"].item()
    w2_start = batch_item["valid_start_w2"].item()
    w2_end = batch_item["valid_end_w2"].item()

    spatial_mask = batch_item.get("spatial_mask")
    if spatial_mask is not None:
        spatial_mask = spatial_mask.numpy()

    S_inp, T, C, H, W = source_frames.shape
    S_tgt = target_images.shape[0]

    # ---- Row 0: 输入帧（前几个） ----
    for s_idx, src_name in enumerate(input_sources):
        if not source_input_mask[s_idx]:
            ax = fig.add_subplot(gs[gs_row, s_idx * 2])
            ax.text(0.5, 0.5, f"{src_name.upper()}\nNOT AVAIL", ha="center", va="center", fontsize=8, color="red")
            ax.axis("off")
            ax = fig.add_subplot(gs[gs_row, s_idx * 2 + 1])
            ax.axis("off")
            continue
        valid_t = [t for t in range(T) if source_mask[s_idx, t]]
        n_show = min(len(valid_t), 2)
        for j in range(n_show):
            ax = fig.add_subplot(gs[gs_row, s_idx * 2 + j])
            t_idx = valid_t[j]
            frame = source_frames[s_idx, t_idx]
            ts_str = timestamp_to_str(source_ts[s_idx, t_idx])
            if src_name in ("s2", "landsat"):
                rgb = frame[:3].transpose(1, 2, 0)
                rgb = np.clip((rgb + 3) / 6, 0, 1)
                ax.imshow(rgb)
            elif src_name == "s1":
                ax.imshow(frame[0], cmap="gray", vmin=-3, vmax=3)
            ax.set_title(f"{src_name.upper()} t={t_idx}\n{ts_str}", fontsize=6)
            ax.axis("off")

    # ---- Row 1: 输入帧（中间帧） + 重建目标 ----
    for s_idx, src_name in enumerate(input_sources):
        if not source_input_mask[s_idx]:
            continue
        valid_t = [t for t in range(T) if source_mask[s_idx, t]]
        if len(valid_t) > 2:
            ax = fig.add_subplot(gs[gs_row + 1, s_idx * 2])
            t_idx = valid_t[len(valid_t) // 2]
            frame = source_frames[s_idx, t_idx]
            ts_str = timestamp_to_str(source_ts[s_idx, t_idx])
            if src_name in ("s2", "landsat"):
                rgb = frame[:3].transpose(1, 2, 0)
                rgb = np.clip((rgb + 3) / 6, 0, 1)
                ax.imshow(rgb)
            elif src_name == "s1":
                ax.imshow(frame[0], cmap="gray", vmin=-3, vmax=3)
            ax.set_title(f"{src_name.upper()} mid\n{ts_str}", fontsize=6)
            ax.axis("off")
        else:
            ax = fig.add_subplot(gs[gs_row + 1, s_idx * 2])
            ax.axis("off")
        ax = fig.add_subplot(gs[gs_row + 1, s_idx * 2 + 1])
        ax.axis("off")

    # 重建目标 (放在第1行后半部分)
    for t_idx, tgt_name in enumerate(target_sources):
        col = 6 + t_idx
        if col >= 12:
            continue
        ax = fig.add_subplot(gs[gs_row + 1, col])
        if not target_mask[t_idx]:
            ax.text(0.5, 0.5, f"{tgt_name}\nN/A", ha="center", va="center", fontsize=6, color="red")
            ax.axis("off")
            continue
        data = target_images[t_idx]
        C2, H2, W2 = data.shape
        if target_loss_type[t_idx] == 1:
            # 分类目标: 将映射值恢复为整数类别索引
            if C2 > 1:
                # one-hot 编码 (旧版)
                label = np.argmax(data, axis=0)
            else:
                # 单通道映射值 (新版): 乘以 (num_classes-1) 恢复
                num_cls = 11 if tgt_name == "worldcover" else 9
                label = np.round(data[0] * (num_cls - 1)).astype(int)
            ax.imshow(label, cmap="tab20", vmin=0, vmax=10)
        elif C2 >= 3:
            rgb = data[:3].transpose(1, 2, 0)
            rgb = np.clip((rgb + 3) / 6, 0, 1)
            ax.imshow(rgb)
        else:
            ax.imshow(data[0], cmap="viridis")
        ax.set_title(tgt_name, fontsize=6)
        ax.axis("off")

    # ---- Row 2: 时间线 + 掩码 + 统计 ----
    # 时间线
    ax_time = fig.add_subplot(gs[gs_row + 2, :4])
    ax_time.set_xlim(0, 10)
    ax_time.set_ylim(0, 3)
    ax_time.axis("off")

    all_ts_flat = []
    for s_idx in range(len(input_sources)):
        if source_input_mask[s_idx]:
            for t in range(T):
                if source_mask[s_idx, t]:
                    all_ts_flat.append(source_ts[s_idx, t])
    all_ts_flat = sorted(set(all_ts_flat))

    if len(all_ts_flat) >= 2:
        t_min, t_max = all_ts_flat[0], all_ts_flat[-1]
        span = t_max - t_min
        y_pos = {"s2": 2.5, "s1": 1.5, "landsat": 0.5}
        colors = {"s2": "green", "s1": "blue", "landsat": "red"}
        for s_idx, src_name in enumerate(input_sources):
            if not source_input_mask[s_idx]:
                continue
            for t in range(T):
                if source_mask[s_idx, t]:
                    x = 1 + 8 * (source_ts[s_idx, t] - t_min) / span
                    ax_time.scatter(x, y_pos[src_name], c=colors[src_name], s=5, alpha=0.6)
        for y_base in [0.3, 1.3, 2.3]:
            w1_x0 = 1 + 8 * (w1_start - t_min) / span
            w1_x1 = 1 + 8 * (w1_end - t_min) / span
            w2_x0 = 1 + 8 * (w2_start - t_min) / span
            w2_x1 = 1 + 8 * (w2_end - t_min) / span
            ax_time.axvspan(w1_x0, w1_x1, ymin=y_base, ymax=y_base+0.4, alpha=0.2, color="orange")
            ax_time.axvspan(w2_x0, w2_x1, ymin=y_base, ymax=y_base+0.4, alpha=0.2, color="purple")
        ax_time.set_yticks([0.5, 1.5, 2.5])
        ax_time.set_yticklabels(["Landsat", "S1", "S2"], fontsize=6)
        ax_time.set_title(f"W1: {timestamp_to_str(w1_start)}~{timestamp_to_str(w1_end)} | W2: {timestamp_to_str(w2_start)}~{timestamp_to_str(w2_end)}", fontsize=8)

    # 空间掩码
    ax_mask = fig.add_subplot(gs[gs_row + 2, 4:6])
    if spatial_mask is not None:
        ax_mask.imshow(spatial_mask, cmap="gray", vmin=0, vmax=1)
        mask_ratio = 1 - spatial_mask.mean()
        ax_mask.set_title(f"CT Mask\n{mask_ratio*100:.1f}% masked", fontsize=7)
    else:
        ax_mask.text(0.5, 0.5, "No mask", ha="center", va="center", fontsize=8)
    ax_mask.axis("off")

    # 统计文本
    stats_text = [f"Patch: {patch_id}"]
    for s_idx, src_name in enumerate(input_sources):
        if not source_input_mask[s_idx]:
            stats_text.append(f"{src_name}: N/A")
            continue
        valid_count = int(source_mask[s_idx].sum())
        frames = source_frames[s_idx, source_mask[s_idx]]
        stats_text.append(f"{src_name}: {valid_count}f μ={frames.mean():.2f} σ={frames.std():.2f}")

    ax_stats = fig.add_subplot(gs[gs_row + 2, 6:])
    ax_stats.text(0.05, 0.95, "\n".join(stats_text), transform=ax_stats.transAxes,
                  fontsize=6, verticalalignment="top", family="monospace",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    ax_stats.axis("off")


def inspect_batch(batch, output_path: Path, num_patches: int):
    """可视化 batch 中的多个 patch."""
    input_sources = ["s2", "s1", "landsat"]
    target_sources = ["s2", "s1", "landsat", "dem", "worldcover", "dynamic_world", "jrc_water"]

    batch_size = batch["source_frames"].shape[0]
    n_show = min(num_patches, batch_size)

    fig = plt.figure(figsize=(24, 4 * n_show))
    global gs
    gs = fig.add_gridspec(3 * n_show, 12, hspace=0.3, wspace=0.15)

    for i in range(n_show):
        item = {
            "patch_id": batch["patch_id"][i] if isinstance(batch["patch_id"], list) else batch["patch_id"],
            "source_frames": batch["source_frames"][i],
            "source_timestamps_ms": batch["source_timestamps_ms"][i],
            "source_frame_mask": batch["source_frame_mask"][i],
            "source_input_mask": batch["source_input_mask"][i],
            "target_images": batch["target_images"][i],
            "target_mask": batch["target_mask"][i],
            "target_loss_type": batch["target_loss_type"][i],
            "valid_start_w1": batch["valid_start_w1"][i],
            "valid_end_w1": batch["valid_end_w1"][i],
            "valid_start_w2": batch["valid_start_w2"][i],
            "valid_end_w2": batch["valid_end_w2"][i],
            "spatial_mask": batch.get("spatial_mask", [None] * batch_size)[i] if batch.get("spatial_mask") is not None else None,
        }
        plot_one_patch(item, fig, i * 3, input_sources, target_sources)

    fig.suptitle(f"Training Data Inspection — {n_show} Patches (adjacent_month mode)", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")

    # 数据质量检查
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    all_ok = True
    for i in range(n_show):
        patch_id = batch["patch_id"][i] if isinstance(batch["patch_id"], list) else str(batch["patch_id"])
        issues = []
        source_input_mask = batch["source_input_mask"][i].numpy()
        source_mask = batch["source_frame_mask"][i].numpy()
        source_frames = batch["source_frames"][i].numpy()
        target_mask = batch["target_mask"][i].numpy()
        target_images = batch["target_images"][i].numpy()

        for s_idx, src_name in enumerate(input_sources):
            if not source_input_mask[s_idx]:
                issues.append(f"  {src_name}: N/A")
                continue
            valid_count = int(source_mask[s_idx].sum())
            if valid_count < 4:
                issues.append(f"  {src_name}: too few frames ({valid_count})")
            frames = source_frames[s_idx, source_mask[s_idx]]
            nan_ratio = np.isnan(frames).sum() / frames.size
            if nan_ratio > 0.01:
                issues.append(f"  {src_name}: high NaN ({nan_ratio*100:.1f}%)")

        for t_idx, tgt_name in enumerate(target_sources):
            if not target_mask[t_idx]:
                issues.append(f"  {tgt_name}: missing target")

        if issues:
            print(f"\nPatch {patch_id}:")
            for issue in issues:
                print(issue)
            all_ok = False
        else:
            print(f"Patch {patch_id}: ✅")

    if all_ok:
        print("\n✅ All patches passed quality check")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen_v4_cd_upgrade.yaml")
    parser.add_argument("--output", type=str, default="/workspace/outputs/aef_qwen_v4_cd_upgrade/training_data_inspection.png")
    parser.add_argument("--num-patches", type=int, default=5, help="可视化的 patch 数量")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"Loading dataset with config: {args.config}")
    print(f"  window_mode: {getattr(cfg.data, 'window_mode', 'random_split')}")
    print(f"  ct_mask_ratio: {getattr(cfg.data, 'ct_mask_ratio', 0.0)}")

    dataloader = build_dataloader(cfg, training=True, distributed=False, world_size=1, rank=0)

    batch = next(iter(dataloader))
    print(f"\nBatch loaded: {batch['source_frames'].shape[0]} patches")

    inspect_batch(batch, Path(args.output), args.num_patches)


if __name__ == "__main__":
    main()
