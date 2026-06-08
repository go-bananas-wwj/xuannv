"""
可视化：随机选取 5 个 patch，对比学生 embedding 与 AEF 教师 embedding。

用法:
    python scripts/visualize/compare_embeddings.py \
        --checkpoint outputs/aef_haidian/step_005000.pt \
        --output_dir outputs/visualizations \
        --num_patches 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu  # NPU support must be imported before using npu device
import torch.nn.functional as F

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn


def load_model(checkpoint_path: str | None, device: str = "cpu") -> AlphaEarthFoundations:
    source_channels = {
        "tianyi_sar": 1,
        "s1": 2,
        "s2": 6,
        "landsat": 6,
        "planet": 4,
    }
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources=source_channels,
        decode_sources=source_channels,
        per_source_latent=32,
        enable_text_align=False,
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print(f"[Load] Checkpoint from {checkpoint_path}")
    else:
        print("[Load] No checkpoint provided, using random initialization")
    model.to(device)
    model.eval()
    return model


def cosine_similarity_map(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """
    student: (B, H, W, D)
    teacher: (B, H, W, D)
    return: (B, H, W) cosine similarity
    """
    s = F.normalize(student, p=2, dim=-1)
    t = F.normalize(teacher, p=2, dim=-1)
    return (s * t).sum(dim=-1)


def embedding_to_rgb(emb: torch.Tensor, dims: tuple[int, int, int] = (0, 21, 42)) -> np.ndarray:
    """
    emb: (H, W, D)
    return: (H, W, 3) RGB image, normalized to [0, 1]
    """
    rgb = emb[..., dims].cpu().numpy()
    # per-channel z-score -> [0, 1]
    for i in range(3):
        c = rgb[..., i]
        mean, std = c.mean(), c.std()
        if std > 1e-6:
            c = (c - mean) / (2 * std)
        c = np.clip(c, -1, 1)
        c = (c + 1) / 2
        rgb[..., i] = c
    return rgb


def visualize_patch(
    axs,
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor | None,
    patch_id: str,
):
    """
    axs: 1x4 或 1x3 matplotlib axes
    student_emb: (H, W, D)
    teacher_emb: (H, W, D) or None
    """
    H, W, D = student_emb.shape

    # Student RGB
    rgb_s = embedding_to_rgb(student_emb)
    axs[0].imshow(rgb_s)
    axs[0].set_title(f"Student Embedding\n{D}D @ {H}x{W}")
    axs[0].axis("off")

    if teacher_emb is not None:
        # Teacher RGB
        rgb_t = embedding_to_rgb(teacher_emb)
        axs[1].imshow(rgb_t)
        axs[1].set_title(f"Teacher Embedding\n{D}D @ {H}x{W}")
        axs[1].axis("off")

        # Cosine similarity
        sim = cosine_similarity_map(student_emb.unsqueeze(0), teacher_emb.unsqueeze(0))[0]
        sim_np = sim.cpu().numpy()
        im = axs[2].imshow(sim_np, cmap="RdYlGn", vmin=-1, vmax=1)
        axs[2].set_title(f"Cosine Similarity\nmean={sim_np.mean():.3f} std={sim_np.std():.3f}")
        axs[2].axis("off")
        plt.colorbar(im, ax=axs[2], fraction=0.046)

        # Histogram: per-dimension mean comparison
        s_mean = student_emb.mean(dim=(0, 1)).cpu().numpy()
        t_mean = teacher_emb.mean(dim=(0, 1)).cpu().numpy()
        axs[3].plot(s_mean, label="Student", alpha=0.7)
        axs[3].plot(t_mean, label="Teacher", alpha=0.7)
        axs[3].set_title("Per-Dimension Mean")
        axs[3].set_xlabel("Dimension")
        axs[3].set_ylabel("Mean Value")
        axs[3].legend()
        axs[3].grid(True, alpha=0.3)
    else:
        # No teacher: show norm map and per-dim stats
        norms = student_emb.norm(dim=-1).cpu().numpy()
        im = axs[1].imshow(norms, cmap="viridis")
        axs[1].set_title(f"Embedding Norm\nmean={norms.mean():.3f} std={norms.std():.3f}")
        axs[1].axis("off")
        plt.colorbar(im, ax=axs[1], fraction=0.046)

        s_mean = student_emb.mean(dim=(0, 1)).cpu().numpy()
        axs[2].plot(s_mean)
        axs[2].set_title("Per-Dimension Mean")
        axs[2].set_xlabel("Dimension")
        axs[2].set_ylabel("Mean Value")
        axs[2].grid(True, alpha=0.3)

        axs[3].axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations")
    parser.add_argument("--num-patches", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    model = load_model(args.checkpoint, device)

    # Build dataset
    source_names = ["tianyi_sar", "s1", "s2", "landsat", "planet"]
    dataset = HaidianAEFDataset(
        data_root="data_raw/haidian/scenes",
        planet_root="data_raw/beijing/planetscene",
        stats_dir="statistics/haidian",
        split="val",
        image_size=128,
        source_names=source_names,
        max_frames=16,
    )

    if len(dataset) == 0:
        print("[Error] Dataset is empty")
        return

    # Random sample
    indices = random.sample(range(len(dataset)), min(args.num_patches, len(dataset)))
    print(f"[Viz] Selected patches: {indices}")

    samples = [dataset[i] for i in indices]
    batch = collate_fn(samples)

    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"].to(device)

    with torch.no_grad():
        out = model(source_data, timestamps, valid_periods)

    student_embs = out["embeddings"]  # (B, H, W, 64)

    has_teacher = "aef_embedding" in batch
    teacher_embs = None
    if has_teacher:
        teacher_embs = batch["aef_embedding"].to(device)  # (B, 64, H, W)
        teacher_embs = teacher_embs.permute(0, 2, 3, 1)  # (B, H, W, 64)
        # Interpolate if size mismatch
        B, H, W, D = student_embs.shape
        if teacher_embs.shape[1] != H or teacher_embs.shape[2] != W:
            teacher_embs = F.interpolate(
                teacher_embs.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)

    n_cols = 4 if has_teacher else 4
    fig, axes = plt.subplots(args.num_patches, n_cols, figsize=(n_cols * 4, args.num_patches * 3.5))
    if args.num_patches == 1:
        axes = axes.reshape(1, -1)

    patch_ids = batch.get("patch_ids", [f"patch_{i}" for i in indices])

    for i, idx in enumerate(indices):
        student_emb = student_embs[i]
        teacher_emb = teacher_embs[i] if teacher_embs is not None else None
        visualize_patch(axes[i], student_emb, teacher_emb, patch_ids[i])
        axes[i][0].set_ylabel(patch_ids[i], fontsize=10, rotation=0, ha="right", va="center")

    fig.suptitle(
        f"Embedding Comparison — {args.num_patches} Random Patches\n"
        f"Checkpoint: {args.checkpoint or 'Random Init'}",
        fontsize=12,
    )
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"embedding_compare_{args.seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
