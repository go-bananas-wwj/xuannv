"""5源合并大图可视化预览 — 随机初始化模型."""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch_npu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    C = arr.shape[-1]
    rgb = np.zeros((*arr.shape[:2], 3), dtype=np.float32)
    if C == 1:
        ch = arr[..., 0]
        ch_min, ch_max = ch.min(), ch.max()
        gray = (ch - ch_min) / (ch_max - ch_min + 1e-8)
        rgb[..., 0] = gray
        rgb[..., 1] = gray
        rgb[..., 2] = gray
    else:
        for c in range(min(C, 3)):
            ch = arr[..., c]
            ch_min, ch_max = ch.min(), ch.max()
            rgb[..., c] = (ch - ch_min) / (ch_max - ch_min + 1e-8)
    return rgb


def embed_to_rgb_shared(student_emb: np.ndarray, aef_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W, D = aef_emb.shape
    aef_flat = aef_emb.reshape(-1, D)
    mean = aef_flat.mean(axis=0)
    _, _, vh = np.linalg.svd(aef_flat - mean, full_matrices=False)
    basis = vh[:3].T

    aef_proj = (aef_flat - mean) @ basis
    aef_proj = (aef_proj - aef_proj.min(axis=0)) / (aef_proj.max(axis=0) - aef_proj.min(axis=0) + 1e-8)
    aef_rgb = aef_proj.reshape(H, W, 3)

    student_proj = (student_emb.reshape(-1, D) - mean) @ basis
    student_proj = (student_proj - student_proj.min(axis=0)) / (student_proj.max(axis=0) - student_proj.min(axis=0) + 1e-8)
    student_rgb = student_proj.reshape(H, W, 3)
    return student_rgb, aef_rgb


def main():
    device = "npu:0"
    torch.npu.set_device(0)

    # 5源模型（随机初始化，不加载旧权重）
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources={"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4},
        decode_sources={
            "s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4,
            "dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1,
        },
        per_source_latent=32,
        enable_text_align=False,
    ).to(device)
    model.eval()
    print("[VizPreview] 5-source model initialized (random weights)")

    # 数据集（带时间筛选）
    dataset = HaidianAEFDataset(
        data_root="/workspace/xuannv/data_raw/haidian/scenes",
        planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
        stats_dir="/workspace/xuannv/statistics/haidian",
        image_size=128,
        source_names=["s1", "s2", "tianyi_sar", "landsat", "planet", "dem", "worldcover", "dynamic_world", "jrc_water"],
        split="val",
        seed=42,
        start_date="20251201",
        end_date="20260430",
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))

    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"]

    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)

    patch_id = batch.get("patch_ids", ["b0"])[0]
    viz_dir = "/workspace/xuannv/aef_reference/outputs/viz_preview"
    os.makedirs(viz_dir, exist_ok=True)

    # ===== 合并大图 =====
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.25, wspace=0.15,
                           height_ratios=[1.2, 1.5, 1.5, 1.5])

    # Row 0: 5个输入源
    input_sources = ["s1", "s2", "tianyi_sar", "landsat", "planet"]
    for i, src in enumerate(input_sources):
        ax = fig.add_subplot(gs[0, i])
        if src not in batch["source_data"]:
            ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", fontsize=14)
            ax.axis("off")
            continue

        data = batch["source_data"][src][0]
        ts = batch["timestamps"][src][0]
        T = data.shape[0]

        if src == "planet":
            valid_idx = None
            for t in range(T):
                if data[t].abs().max() > 0.001:
                    valid_idx = t
                    break
            if valid_idx is None:
                ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", fontsize=14)
                ax.axis("off")
                continue
            frame = data[valid_idx]
        else:
            center = ts.mean()
            t_idx = (ts - center).abs().argmin().item()
            frame = data[t_idx]

        rgb = tensor_to_rgb(frame)
        ax.imshow(rgb)
        title = src.upper()
        if src == "landsat":
            title += " [30m→10m]"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Row 1-3: Embedding PCA RGB + diff
    ax_student = fig.add_subplot(gs[1, :])
    ax_aef = fig.add_subplot(gs[2, :])
    ax_diff = fig.add_subplot(gs[3, :])

    student_emb = out["student_embeddings"][0].cpu().numpy()
    if "aef_embedding" in batch and batch["aef_embedding"] is not None:
        aef_emb = batch["aef_embedding"][0].permute(1, 2, 0).cpu().numpy()
        student_rgb, aef_rgb = embed_to_rgb_shared(student_emb, aef_emb)
        diff = np.abs(student_rgb - aef_rgb).mean(axis=-1)

        ax_student.imshow(student_rgb)
        ax_student.set_title(f"Student Embedding (PCA RGB) — {patch_id}", fontsize=14, fontweight="bold")
        ax_student.axis("off")

        ax_aef.imshow(aef_rgb)
        ax_aef.set_title(f"AEF Official Embedding (PCA RGB) — {patch_id}", fontsize=14, fontweight="bold")
        ax_aef.axis("off")

        im = ax_diff.imshow(diff, cmap="hot")
        ax_diff.set_title(f"|Student - AEF| (mean) — {patch_id}", fontsize=14, fontweight="bold")
        ax_diff.axis("off")
        plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
    else:
        ax_student.text(0.5, 0.5, "AEF Embedding Not Available", ha="center", va="center", fontsize=14)
        ax_student.axis("off")
        ax_aef.axis("off")
        ax_diff.axis("off")

    plt.suptitle("5-Source Patch Visualization Preview (Random Init)", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(os.path.join(viz_dir, "viz_preview_5source.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VizPreview] Saved to {viz_dir}/viz_preview_5source.png")


if __name__ == "__main__":
    main()
