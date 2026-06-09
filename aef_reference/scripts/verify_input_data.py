"""随机采样一个patch，可视化送给模型的输入数据，验证数据预处理是否正确."""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv/aef_reference")

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.aef.data.haidian_dataset import HaidianAEFDataset


def _tensor_to_rgb(frame: np.ndarray) -> np.ndarray:
    """将 (C, H, W) 转为 RGB，每通道 min-max 归一化."""
    C = frame.shape[0]
    rgb = np.zeros((3, frame.shape[1], frame.shape[2]), dtype=np.float32)
    for c in range(min(C, 3)):
        ch = frame[c]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            rgb[c] = (ch - mn) / (mx - mn)
    return np.transpose(rgb, (1, 2, 0))


def visualize_patch_input(patch_id: str | None = None) -> str:
    """可视化单个patch的输入数据."""
    dataset = HaidianAEFDataset(
        data_root="/workspace/xuannv/data_raw/haidian/scenes",
        planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
        stats_dir="/workspace/xuannv/statistics/haidian",
        split="train",
        train_ratio=0.9,
        seed=42,
        max_frames=16,
        start_date="2025-12-01",
        end_date="2026-04-30",
        source_names=["s1", "s2", "tianyi_sar", "landsat", "planet"],
    )

    # 随机采样或指定patch
    if patch_id is None:
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        patch_id = sample["patch_id"]
    else:
        sample = None
        for idx in range(len(dataset)):
            s = dataset[idx]
            if s["patch_id"] == patch_id:
                sample = s
                break
        if sample is None:
            raise ValueError(f"Patch {patch_id} not found in dataset")

    source_data = sample["source_data"]
    timestamps = sample["timestamps"]

    # 取每个源时间中点最近的帧
    sources_to_viz = ["s1", "s2", "tianyi_sar", "landsat", "planet"]
    frames = {}
    for src in sources_to_viz:
        if src not in source_data:
            continue
        data = source_data[src]  # (T, H, W, C)
        ts = timestamps[src]     # (T,)
        T = data.shape[0]
        if T == 0:
            continue
        # 取时间中点最近的帧
        ts_np = ts.numpy()
        mid_t = (ts_np.min() + ts_np.max()) / 2.0
        closest_idx = int(np.argmin(np.abs(ts_np - mid_t)))
        frame = data[closest_idx].numpy()  # (H, W, C)
        # 转为 (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        frames[src] = frame

    # 创建大图
    n_sources = len(frames)
    fig, axes = plt.subplots(1, n_sources, figsize=(4 * n_sources, 4))
    if n_sources == 1:
        axes = [axes]

    titles = {
        "s1": "S1 (Sentinel-1)",
        "s2": "S2 (Sentinel-2)",
        "tianyi_sar": "TIANYI_SAR",
        "landsat": "LANDSAT [30m→10m]",
        "planet": "PLANET [3m→10m]",
    }

    for ax, (src, frame) in zip(axes, frames.items()):
        if src == "tianyi_sar" and frame.shape[0] == 1:
            # 单通道灰度
            img = frame[0]
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
            ax.imshow(img, cmap="gray")
        elif src == "s1" and frame.shape[0] == 2:
            # SAR: VV, VH -> R=VV, G=VH, B=0
            rgb = np.zeros((frame.shape[1], frame.shape[2], 3), dtype=np.float32)
            for c in range(2):
                ch = frame[c]
                mn, mx = ch.min(), ch.max()
                if mx > mn:
                    rgb[:, :, c] = (ch - mn) / (mx - mn)
            ax.imshow(np.clip(rgb, 0, 1))
        else:
            rgb = _tensor_to_rgb(frame)
            ax.imshow(np.clip(rgb, 0, 1))

        ax.set_title(titles.get(src, src), fontsize=12, fontweight="bold")
        ax.axis("off")

        # 打印像素统计
        print(f"\n[{src}] {patch_id}")
        print(f"  shape: {frame.shape}")
        print(f"  min: {frame.min():.4f}, max: {frame.max():.4f}, mean: {frame.mean():.4f}, std: {frame.std():.4f}")
        print(f"  unique_values_ratio: {len(np.unique(frame)) / frame.size:.4f}")

    fig.suptitle(f"Input Data Visualization — {patch_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = Path("outputs/viz_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"input_verify_{patch_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-id", type=str, default=None, help="指定patch ID，不指定则随机")
    args = parser.parse_args()
    visualize_patch_input(args.patch_id)
