#!/usr/bin/env python3
"""生成 OlmoEarth spatial tokens 的空间拼接可视化图 (类似 AEF viz).

类似 /workspace/raw/aef_embeddings/haidian_2025_visualization.png:
- 每个 patch 取 tokens 前 3 个 channel 作为 RGB (per-patch normalized)
- 按 UTM 网格位置拼接成全局 mosaic
- 支持 Haidian (320 patches) 和 Harbin (424 patches)

用法:
    python scripts/visualize_olmoearth_spatial_mosaic.py --region haidian --month 04
    python scripts/visualize_olmoearth_spatial_mosaic.py --region haidian --month 2026/04
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace/xuannv")

REGIONS = {
    "haidian": {
        "tokens_root": "/workspace/outputs/olmoearth_haidian",
        "meta": "/workspace/raw/haidian_olmoearth/patches_meta.json",
        "n_patches": 320,
    },
    "harbin": {
        "tokens_root": "/workspace/outputs/olmoearth_harbin",
        "meta": "/workspace/xuannv_show/data/harbin/patches_meta.json",
        "n_patches": 424,
    },
}


def load_meta(meta_path: str):
    with open(meta_path) as f:
        data = json.load(f)
    # haidian: list format
    if isinstance(data, list):
        return {p["patch_id"]: p for p in data}
    # harbin: list format (same)
    if isinstance(data, dict) and "patches" in data:
        return {f"patch_{p['id']:06d}": p for p in data["patches"]}
    return {p["patch_id"]: p for p in data}


def load_tokens(tokens_root: str, month: str):
    path = Path(tokens_root) / month / "spatial_tokens.npz"
    d = np.load(path)
    tokens = d["tokens"].astype(np.float32)  # (N, 32, 32, 768)
    patch_ids = [str(p) for p in d["patch_ids"]]
    return tokens, patch_ids


def render_mosaic(tokens, patch_ids, meta_dict, out_path: str, scale: int = 4):
    """拼接 mosaic.

    tokens: (N, H, W, C) — 取前 3 channel 作为 RGB
    patch_ids: list[str]
    meta_dict: patch_id -> {bounds/utm_bounds, ...}
    scale: 上采样倍数 (32x32 太小，默认上采样 4x -> 128x128 per patch)
    """
    # 从 UTM bounds 推断网格位置 (bounds = [left, bottom, right, top])
    left_vals = []
    bottom_vals = []
    for pid in patch_ids:
        info = meta_dict.get(pid, {})
        b = info.get("bounds", info.get("utm_bounds", [0, 0, 0, 0]))
        left_vals.append(b[0])
        bottom_vals.append(b[1])

    # 推断网格坐标 (假设等间距 1280m)
    min_left, max_left = min(left_vals), max(left_vals)
    min_bottom, max_bottom = min(bottom_vals), max(bottom_vals)
    step = 1280.0  # 标准 patch size

    grid_xs = [int(round((v - min_left) / step)) for v in left_vals]
    grid_ys = [int(round((v - min_bottom) / step)) for v in bottom_vals]

    min_gx, max_gx = min(grid_xs), max(grid_xs)
    min_gy, max_gy = min(grid_ys), max(grid_ys)
    nx = max_gx - min_gx + 1
    ny = max_gy - min_gy + 1

    H, W = tokens.shape[1], tokens.shape[2]
    patch_h = H * scale
    patch_w = W * scale

    # 画布 (黑色背景)
    canvas = np.zeros((ny * patch_h, nx * patch_w, 3), dtype=np.uint8)

    for pid, tok, gx, gy in zip(patch_ids, tokens, grid_xs, grid_ys):
        # 取前 3 个 channel -> RGB
        rgb = tok[:, :, :3]  # (H, W, 3)

        # per-patch min-max 归一化
        vmin, vmax = rgb.min(), rgb.max()
        if vmax > vmin:
            rgb = (rgb - vmin) / (vmax - vmin)
        else:
            rgb = np.zeros_like(rgb)

        rgb = (rgb * 255).astype(np.uint8)

        # 上采样
        if scale > 1:
            img = Image.fromarray(rgb)
            img = img.resize((patch_w, patch_h), Image.Resampling.NEAREST)
            rgb = np.array(img)

        # 放置到画布 (翻转 Y，让北在上)
        ix = gx - min_gx
        iy = (ny - 1) - (gy - min_gy)
        y0 = iy * patch_h
        x0 = ix * patch_w
        canvas[y0:y0 + patch_h, x0:x0 + patch_w] = rgb

    # 保存
    img = Image.fromarray(canvas)
    img.save(out_path)
    print(f"Saved mosaic: {out_path}  ({canvas.shape[1]}x{canvas.shape[0]} px, "
          f"grid={nx}x{ny}, patch={patch_w}x{patch_h})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=["haidian", "harbin"])
    parser.add_argument("--month", required=True, help="如 04 或 2026/04")
    parser.add_argument("--scale", type=int, default=4, help="上采样倍数")
    parser.add_argument("--out", default=None, help="输出路径")
    args = parser.parse_args()

    cfg = REGIONS[args.region]
    tokens, patch_ids = load_tokens(cfg["tokens_root"], args.month)
    meta_dict = load_meta(cfg["meta"])

    month_name = args.month.replace("/", "_")
    out = args.out or f"/workspace/outputs/olmoearth_viz/{args.region}_olmoearth_{month_name}_mosaic.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    render_mosaic(tokens, patch_ids, meta_dict, out, scale=args.scale)


if __name__ == "__main__":
    main()
