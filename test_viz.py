"""测试可视化代码 — 使用真实 patch 数据 + 已有 checkpoint."""
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
from einops import rearrange
from torch.utils.data import DataLoader

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn


def embed_to_rgb(emb: np.ndarray) -> np.ndarray:
    """64D -> 3D PCA RGB. 输入 (B, H, W, D) 或 (D, H, W)."""
    if emb.ndim == 3:
        emb = emb[np.newaxis, ...]
    B, H, W, D = emb.shape
    emb_flat = emb.reshape(-1, D)
    mean = emb_flat.mean(axis=0, keepdims=True)
    emb_centered = emb_flat - mean
    u, s, vt = np.linalg.svd(emb_centered, full_matrices=False)
    rgb = u[:, :3] * s[:3]
    rgb_min, rgb_max = rgb.min(axis=0, keepdims=True), rgb.max(axis=0, keepdims=True)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    return rgb.reshape(B, H, W, 3)


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """将 (H, W, C) tensor 转为归一化 RGB numpy. 对每通道单独 min-max."""
    arr = t.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    C = arr.shape[-1]
    rgb = np.zeros((*arr.shape[:2], 3), dtype=np.float32)
    for c in range(min(C, 3)):
        ch = arr[..., c]
        ch_min, ch_max = ch.min(), ch.max()
        rgb[..., c] = (ch - ch_min) / (ch_max - ch_min + 1e-8)
    return rgb


def show_input_raw(batch: dict, idx: int = 0) -> np.ndarray:
    """显示原始输入数据的一个时间步（取中间帧）. 返回 figure."""
    source_names = ["s1", "s2", "landsat"]
    fig, axes = plt.subplots(1, len(source_names), figsize=(6 * len(source_names), 6))
    if len(source_names) == 1:
        axes = [axes]
    for i, src in enumerate(source_names):
        data = batch["source_data"][src][idx]  # (T, H, W, C)
        T = data.shape[0]
        frame = data[T // 2]  # 取中间帧
        rgb = tensor_to_rgb(frame)
        axes[i].imshow(rgb)
        axes[i].set_title(f"{src.upper()} Raw Input (frame {T//2}/{T})")
        axes[i].axis("off")
    plt.suptitle("Real Multi-source Input Patch Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def show_reconstruction(batch: dict, predictions: dict, idx: int = 0) -> np.ndarray:
    """显示重建对比：Target vs Pred，每通道单独归一化."""
    src_list = list(predictions.keys())[:3]
    if not src_list:
        return None
    fig, axes = plt.subplots(2, len(src_list), figsize=(6 * len(src_list), 10))
    if len(src_list) == 1:
        axes = axes.reshape(2, 1)

    for i, src in enumerate(src_list):
        # Pred
        pred = predictions[src][idx].detach().cpu()  # (H, W, C)
        pred_rgb = tensor_to_rgb(pred)
        axes[1, i].imshow(pred_rgb)
        axes[1, i].set_title(f"{src.upper()} Pred")
        axes[1, i].axis("off")

        # Target — 取时间中点最近帧
        x = batch["source_data"][src][idx]
        ts = batch["timestamps"][src][idx]
        T = x.shape[0]
        center = ts.mean()
        t_idx = (ts - center).abs().argmin().item()
        tgt = x[t_idx]  # (H, W, C)
        tgt_rgb = tensor_to_rgb(tgt)
        axes[0, i].imshow(tgt_rgb)
        axes[0, i].set_title(f"{src.upper()} Target (frame {t_idx}/{T})")
        axes[0, i].axis("off")

    plt.suptitle("Reconstruction Comparison (per-channel norm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def show_pca_rgb(aef_emb: torch.Tensor, student_emb: torch.Tensor, idx: int = 0) -> np.ndarray:
    """AEF 官方 embedding vs Student embedding 的 PCA RGB 对比."""
    # AEF: (B, 64, H, W) -> (B, H, W, 64)
    aef_np = aef_emb[idx].permute(1, 2, 0).cpu().numpy()  # (H, W, 64)
    # Student: (B, H, W, 64)
    student_np = student_emb[idx].cpu().numpy()

    aef_rgb = embed_to_rgb(aef_np)[0]  # (H, W, 3)
    student_rgb = embed_to_rgb(student_np)[0]
    diff = np.abs(aef_rgb - student_rgb).mean(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(aef_rgb)
    axes[0].set_title("AEF Official Embedding (PCA RGB)")
    axes[0].axis("off")
    axes[1].imshow(student_rgb)
    axes[1].set_title("Student Embedding (PCA RGB)")
    axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("|AEF - Student| (mean)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.suptitle("Embedding PCA RGB Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def show_embedding_channels(emb: np.ndarray, title: str, n_dims: int = 4) -> np.ndarray:
    """显示 embedding 的 n_dims 个随机维度."""
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))
    if n_dims == 1:
        axes = [axes]
    dims = np.random.choice(emb.shape[-1], n_dims, replace=False)
    for i, d in enumerate(dims):
        im = axes[i].imshow(emb[..., d], cmap="viridis")
        axes[i].set_title(f"dim {d}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    device = "npu:0"
    torch.npu.set_device(0)

    # 模型
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources={"s1": 2, "s2": 6, "landsat": 6},
        decode_sources={
            "s1": 2, "s2": 6, "landsat": 6,
            "dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1,
        },
        per_source_latent=32,
        enable_text_align=False,
    ).to(device)

    ckpt_path = "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/step_000500_seed42.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[VizTest] Loaded checkpoint from {ckpt_path}")

    # 数据集
    dataset = HaidianAEFDataset(
        data_root="/workspace/xuannv/data_raw/haidian/scenes",
        planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
        stats_dir="/workspace/xuannv/statistics/haidian",
        image_size=128,
        source_names=["s1", "s2", "landsat"],
        split="val",
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(loader))

    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"]

    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)

    out = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    viz_dir = "/workspace/xuannv/aef_reference/outputs/aef_distill_seed42/test_viz"
    os.makedirs(viz_dir, exist_ok=True)

    # 1. 原始输入数据
    fig = show_input_raw(batch, idx=0)
    fig.savefig(os.path.join(viz_dir, "01_raw_input.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VizTest] Saved 01_raw_input.png")

    # 2. PCA RGB 对比（AEF 官方 vs Student）
    aef_emb = batch["aef_embedding"].to(device)  # (B, 64, H, W)
    student_emb = out["student_embeddings"]  # (B, H, W, 64)
    fig = show_pca_rgb(aef_emb, student_emb, idx=0)
    fig.savefig(os.path.join(viz_dir, "02_pca_rgb_aef_vs_student.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VizTest] Saved 02_pca_rgb_aef_vs_student.png")

    # 3. Embedding 单维度对比
    aef_np = aef_emb[0].permute(1, 2, 0).cpu().numpy()
    student_np = student_emb[0].cpu().numpy()
    fig = show_embedding_channels(aef_np, "AEF Official Embedding Channels", n_dims=4)
    fig.savefig(os.path.join(viz_dir, "03_aef_embedding_dims.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig = show_embedding_channels(student_np, "Student Embedding Channels", n_dims=4)
    fig.savefig(os.path.join(viz_dir, "04_student_embedding_dims.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VizTest] Saved 03/04_embedding_dims.png")

    # 4. 重建对比
    predictions = {src: rec[:, 0] for src, rec in out["reconstructions"].items()}
    fig = show_reconstruction(batch, predictions, idx=0)
    if fig:
        fig.savefig(os.path.join(viz_dir, "05_reconstruction.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VizTest] Saved 05_reconstruction.png")

    print(f"[VizTest] All saved to {viz_dir}")


if __name__ == "__main__":
    main()
