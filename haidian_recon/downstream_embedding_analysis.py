"""HRE Embedding 空间质量分析."""
from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from haidian_recon.config import Config
from haidian_recon.data.dataset import HaidianReconDataset, collate_fn
from haidian_recon.models.hre_model import HREModel


def extract_embeddings(
    checkpoint_path: str,
    data_root: str,
    stats_dir: str,
    planet_root: str,
    cache_dir: str,
    split: str = "val",
) -> tuple[np.ndarray, list]:
    """提取所有 patch 的 64D embedding."""
    device = torch.device("npu:0")
    cfg = Config()
    source_channels = {s["name"]: s["channels"] for s in cfg.data.sources}

    model = HREModel(
        source_channels=source_channels,
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        use_gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()

    dataset = HaidianReconDataset(
        data_root=data_root,
        planet_root=planet_root,
        stats_dir=stats_dir,
        split=split,
        image_size=cfg.data.image_size,
        source_names=list(source_channels.keys()),
        cache_dir=cache_dir,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=False)

    embeddings = []
    patch_names = []

    with torch.no_grad():
        for batch in loader:
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None and not valid_mask[0].item():
                continue

            patch_name = batch.get("patch_name", ["unknown"])[0]
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            output = model(batch_device, mask_info=None)
            emb = output["embedding"][0].cpu().numpy()
            embeddings.append(emb)
            patch_names.append(patch_name)

    return np.stack(embeddings), patch_names


def compute_uniformity_stats(embeddings: np.ndarray) -> dict:
    """计算 embedding 的均匀性统计."""
    # L2 归一化
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Pairwise cosine similarity
    sim_matrix = normed @ normed.T
    # 排除对角线
    mask = ~np.eye(len(sim_matrix), dtype=bool)
    off_diag_sim = sim_matrix[mask]

    return {
        "mean_cosine_sim": float(off_diag_sim.mean()),
        "std_cosine_sim": float(off_diag_sim.std()),
        "min_cosine_sim": float(off_diag_sim.min()),
        "max_cosine_sim": float(off_diag_sim.max()),
        "mean_l2_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
        "std_l2_norm": float(np.linalg.norm(embeddings, axis=1).std()),
    }


def plot_pca(embeddings: np.ndarray, patch_names: list, output_path: str) -> None:
    """PCA 降维可视化."""
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(len(emb_2d)), cmap="tab20", alpha=0.7, s=50)
    ax.set_title(f"PCA of HRE Embeddings (explained var: {pca.explained_variance_ratio_.sum():.2%})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, label="Patch index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PCA plot saved to {output_path}")


def plot_norm_distribution(embeddings: np.ndarray, output_path: str) -> None:
    """Embedding L2 范数分布."""
    norms = np.linalg.norm(embeddings, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(norms, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(norms.mean(), color="red", linestyle="--", label=f"Mean={norms.mean():.3f}")
    ax.set_title("Distribution of Embedding L2 Norms")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Norm distribution saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data_raw/haidian/scenes")
    parser.add_argument("--planet-root", type=str, default="data_raw/beijing/planetscene")
    parser.add_argument("--stats-dir", type=str, default="statistics/haidian")
    parser.add_argument("--cache-dir", type=str, default="haidian_recon/.cache")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-dir", type=str, default="outputs/hre_eval")
    args = parser.parse_args()

    print("Extracting embeddings...")
    embeddings, patch_names = extract_embeddings(
        args.checkpoint,
        args.data_root,
        args.stats_dir,
        args.planet_root,
        args.cache_dir,
        split=args.split,
    )
    print(f"Extracted {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    print("\nComputing uniformity stats...")
    stats = compute_uniformity_stats(embeddings)
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating PCA plot...")
    plot_pca(embeddings, patch_names, os.path.join(args.output_dir, "embedding_pca.png"))

    print("Generating norm distribution plot...")
    plot_norm_distribution(embeddings, os.path.join(args.output_dir, "embedding_norms.png"))

    results = {
        "checkpoint": args.checkpoint,
        "n_patches": len(embeddings),
        "embedding_dim": int(embeddings.shape[1]),
        "uniformity_stats": stats,
    }
    with open(os.path.join(args.output_dir, "embedding_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/embedding_analysis.json")


if __name__ == "__main__":
    main()
