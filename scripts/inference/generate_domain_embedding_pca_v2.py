#!/usr/bin/env python3
"""生成全域 Embedding PCA-RGB 拼接图 (V2 修正版).

修正点:
- tile_size 固定为 64 (原始 embedding 空间尺寸)
- 使用 patches_meta.json 的 ix/iy 定位
- 全局 PCA fit + 逐 patch transform
- 全局 min-max 归一化到 [0, 255]
- iy 轴翻转 (iy=0 在顶部)
- 可选 feather 边缘平滑

Usage:
    python scripts/inference/generate_domain_embedding_pca_v2.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.decomposition import PCA
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate domain-wide PCA-RGB embedding mosaic")
    parser.add_argument(
        "--embeddings-dir",
        default="/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026",
        help="Path to monthly embeddings directory",
    )
    parser.add_argument(
        "--patches-meta",
        default="/workspace/xuannv_show/data/harbin/patches_meta.json",
        help="Path to patches_meta.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/outputs/aef_qwen_v5_mixed_scale/domain_wide_v2",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Output tile size per patch (embedding spatial size)",
    )
    parser.add_argument(
        "--months",
        default="2026-01,2026-02,2026-03,2026-04,2026-05",
        help="Comma-separated month list",
    )
    parser.add_argument(
        "--feather",
        action="store_true",
        default=True,
        help="Apply slight Gaussian blur to tile edges",
    )
    parser.add_argument(
        "--no-feather",
        action="store_false",
        dest="feather",
        help="Disable feathering",
    )
    return parser.parse_args()


def load_patches_meta(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def gather_embeddings_for_month(
    emb_dir: Path, patches: list[dict], month: str
) -> dict[str, np.ndarray]:
    """Gather all valid embedding arrays for a given month."""
    embs = {}
    for p in tqdm(patches, desc=f"Loading {month}", leave=False):
        pid = p["patch_id"]
        fpath = emb_dir / f"{pid}_{month}.npy"
        if fpath.exists():
            embs[pid] = np.load(fpath)
    return embs


def fit_global_pca(all_embs: dict[str, np.ndarray], max_samples: int = 500_000) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on global sample across all patches."""
    pieces = []
    for emb in all_embs.values():
        D, H, W = emb.shape
        pieces.append(emb.reshape(D, -1).T)
    flat = np.concatenate(pieces, axis=0)

    if flat.shape[0] > max_samples:
        indices = np.random.choice(flat.shape[0], max_samples, replace=False)
        flat_sample = flat[indices]
    else:
        flat_sample = flat

    pca = PCA(n_components=3)
    pca.fit(flat_sample)

    rgb_all = pca.transform(flat_sample)
    vmin = rgb_all.min(axis=0)
    vmax = rgb_all.max(axis=0)
    return pca, vmin, vmax


def patch_to_rgb(
    emb: np.ndarray,
    pca: PCA,
    vmin: np.ndarray,
    vmax: np.ndarray,
    tile_size: int,
    feather: bool = True,
) -> Image.Image:
    """Convert a single patch embedding to PCA-RGB PIL Image."""
    D, H, W = emb.shape
    flat = emb.reshape(D, -1).T
    rgb = pca.transform(flat).reshape(H, W, 3)
    rgb = (rgb - vmin) / (vmax - vmin + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)

    if tile_size != H or tile_size != W:
        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

    if feather:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        edge_mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(edge_mask)
        draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=255, width=2)
        img = Image.composite(blurred, img, edge_mask)

    return img


def build_mosaic(
    patches: list[dict],
    emb_dict: dict[str, np.ndarray],
    pca: PCA,
    vmin: np.ndarray,
    vmax: np.ndarray,
    tile_size: int,
    feather: bool = True,
) -> Image.Image:
    """Build full mosaic canvas from patches using ix/iy."""
    ixs = [p["ix"] for p in patches]
    iys = [p["iy"] for p in patches]
    ix_min, ix_max = min(ixs), max(ixs)
    iy_min, iy_max = min(iys), max(iys)

    n_cols = ix_max - ix_min + 1
    n_rows = iy_max - iy_min + 1

    canvas_w = n_cols * tile_size
    canvas_h = n_rows * tile_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), (26, 26, 46))

    for p in tqdm(patches, desc="Building mosaic", leave=False):
        pid = p["patch_id"]
        if pid not in emb_dict:
            continue
        emb = emb_dict[pid]
        img = patch_to_rgb(emb, pca, vmin, vmax, tile_size, feather=feather)
        col = p["ix"] - ix_min
        row = p["iy"] - iy_min
        x = col * tile_size
        y = (n_rows - 1 - row) * tile_size  # flip Y so iy=0 is at top
        canvas.paste(img, (x, y))

    return canvas


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patches = load_patches_meta(args.patches_meta)
    months = [m.strip() for m in args.months.split(",")]

    print(f"[domain_pca_v2] Patches meta: {len(patches)}")
    print(f"[domain_pca_v2] Months: {months}")
    print(f"[domain_pca_v2] Tile size: {args.tile_size}")
    print(f"[domain_pca_v2] Output dir: {output_dir}")

    for month in months:
        print(f"\n[domain_pca_v2] Processing {month} ...")
        embs = gather_embeddings_for_month(emb_dir, patches, month)
        if not embs:
            print(f"[domain_pca_v2] No embeddings found for {month}, skipping")
            continue

        print(f"[domain_pca_v2] Loaded {len(embs)} patches. Fitting global PCA ...")
        pca, vmin, vmax = fit_global_pca(embs)

        print(f"[domain_pca_v2] Building mosaic ...")
        mosaic = build_mosaic(
            patches, embs, pca, vmin, vmax, args.tile_size, feather=args.feather
        )

        out_path = output_dir / f"domain_embedding_pca_{month}.png"
        mosaic.save(out_path, "PNG")
        print(f"[domain_pca_v2] Saved: {out_path} ({mosaic.size[0]}x{mosaic.size[1]})")

    print("\n[domain_pca_v2] All done!")


if __name__ == "__main__":
    main()
