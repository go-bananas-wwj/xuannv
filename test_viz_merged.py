"""5源2行合并大图可视化预览 — 随机初始化模型 + 多patch."""
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


def viz_one_patch(model, device, sample, step, viz_dir):
    """生成单张2行合并大图."""
    patch_id = sample["patch_id"]
    batch = collate_fn([sample])
    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"]

    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.2, height_ratios=[1, 1.3])
    gs_top = gs[0].subgridspec(1, 5, wspace=0.15)
    gs_bottom = gs[1].subgridspec(1, 3, wspace=0.15)

    # Row 0: 5个输入源
    input_sources = ["s1", "s2", "tianyi_sar", "landsat", "planet"]
    for i, src in enumerate(input_sources):
        ax = fig.add_subplot(gs_top[0, i])
        if src not in batch["source_data"]:
            ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", fontsize=12)
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
                ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", fontsize=12)
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
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Row 1: Student PCA | AEF PCA | Diff
    ax_student = fig.add_subplot(gs_bottom[0, 0])
    ax_aef = fig.add_subplot(gs_bottom[0, 1])
    ax_diff = fig.add_subplot(gs_bottom[0, 2])

    student_emb = out["student_embeddings"][0].cpu().numpy()
    if "aef_embedding" in batch and batch["aef_embedding"] is not None:
        aef_emb = batch["aef_embedding"][0].permute(1, 2, 0).cpu().numpy()
        student_rgb, aef_rgb = embed_to_rgb_shared(student_emb, aef_emb)
        diff = np.abs(student_rgb - aef_rgb).mean(axis=-1)

        ax_student.imshow(student_rgb)
        ax_student.set_title("Student (PCA RGB)", fontsize=12, fontweight="bold")
        ax_student.axis("off")

        ax_aef.imshow(aef_rgb)
        ax_aef.set_title("AEF Official (PCA RGB)", fontsize=12, fontweight="bold")
        ax_aef.axis("off")

        im = ax_diff.imshow(diff, cmap="hot")
        ax_diff.set_title("|Student - AEF|", fontsize=12, fontweight="bold")
        ax_diff.axis("off")
        plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
    else:
        for ax in (ax_student, ax_aef, ax_diff):
            ax.text(0.5, 0.5, "AEF N/A", ha="center", va="center", fontsize=12)
            ax.axis("off")

    plt.suptitle(f"{patch_id} @ Step {step}", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(os.path.join(viz_dir, f"viz_step_{step:06d}_{patch_id}_preview.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VizPreview] Saved {patch_id}")


def main():
    device = "npu:0"
    torch.npu.set_device(0)

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

    viz_dir = "/workspace/xuannv/aef_reference/outputs/viz_preview"
    os.makedirs(viz_dir, exist_ok=True)

    target_patches = {"patch_000036", "patch_000069", "patch_000091", "patch_000120", "patch_000150"}
    found_patches = set()

    for split in ["val", "train"]:
        dataset = HaidianAEFDataset(
            data_root="/workspace/xuannv/data_raw/haidian/scenes",
            planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
            stats_dir="/workspace/xuannv/statistics/haidian",
            image_size=128,
            source_names=["s1", "s2", "tianyi_sar", "landsat", "planet", "dem", "worldcover", "dynamic_world", "jrc_water"],
            split=split,
            seed=42,
            start_date="20251201",
            end_date="20260430",
        )
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample["patch_id"] in target_patches and sample["patch_id"] not in found_patches:
                viz_one_patch(model, device, sample, step=0, viz_dir=viz_dir)
                found_patches.add(sample["patch_id"])

    print(f"[VizPreview] Done. Found {len(found_patches)}/{len(target_patches)} target patches.")


if __name__ == "__main__":
    main()
