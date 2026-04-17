#!/usr/bin/env python3
"""Precompute global overview arrays for AlphaEarth Harbin."""
from __future__ import annotations

import numpy as np
import rasterio
from pathlib import Path

ALPHA_2023_PATH = Path("/workspace/outputs/alphaearth_harbin/alphaearth_harbin_2023.tif")
ALPHA_2024_PATH = Path("/workspace/outputs/alphaearth_harbin/alphaearth_harbin_2024.tif")
CACHE_PATH = Path("/workspace/outputs/alphaearth_harbin/global_overview.npz")


def main():
    print("Opening datasets...")
    ds_2023 = rasterio.open(ALPHA_2023_PATH)
    ds_2024 = rasterio.open(ALPHA_2024_PATH)
    H, W = ds_2023.shape
    print(f"Image size: {W} x {H}")

    rgb_2023 = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_2024 = np.zeros((H, W, 3), dtype=np.uint8)
    score = np.zeros((H, W), dtype=np.float32)

    chunk_h = 1024
    chunk_w = 1024
    total_chunks = ((H + chunk_h - 1) // chunk_h) * ((W + chunk_w - 1) // chunk_w)
    processed = 0

    for y in range(0, H, chunk_h):
        for x in range(0, W, chunk_w):
            win_h = min(chunk_h, H - y)
            win_w = min(chunk_w, W - x)
            window = rasterio.windows.Window(x, y, win_w, win_h)

            emb_2023 = ds_2023.read(window=window)  # [64, win_h, win_w]
            emb_2024 = ds_2024.read(window=window)

            for i, band_idx in enumerate([0, 21, 42]):
                b23 = emb_2023[band_idx]
                p2, p98 = np.percentile(b23, [2, 98])
                norm = np.clip((b23 - p2) / (p98 - p2 + 1e-8), 0, 1)
                rgb_2023[y:y+win_h, x:x+win_w, i] = (norm * 255).astype(np.uint8)

                b24 = emb_2024[band_idx]
                p2, p98 = np.percentile(b24, [2, 98])
                norm = np.clip((b24 - p2) / (p98 - p2 + 1e-8), 0, 1)
                rgb_2024[y:y+win_h, x:x+win_w, i] = (norm * 255).astype(np.uint8)

            D, h, w = emb_2023.shape
            fb = emb_2023.reshape(D, -1)
            fa = emb_2024.reshape(D, -1)
            nb = np.linalg.norm(fb, axis=0, keepdims=True)
            na = np.linalg.norm(fa, axis=0, keepdims=True)
            fb = fb / np.maximum(nb, 1e-8)
            fa = fa / np.maximum(na, 1e-8)
            cos_sim = np.sum(fb * fa, axis=0)
            score_chunk = ((1.0 - np.clip(cos_sim, -1.0, 1.0)) / 2.0).reshape(h, w)
            score[y:y+win_h, x:x+win_w] = score_chunk.astype(np.float32)

            processed += 1
            if processed % 5 == 0 or processed == total_chunks:
                print(f"  Chunk {processed}/{total_chunks} done")

    ds_2023.close()
    ds_2024.close()

    print(f"Saving cache to {CACHE_PATH} ...")
    np.savez(CACHE_PATH, rgb_2023=rgb_2023, rgb_2024=rgb_2024, change_score=score)
    print("Done.")


if __name__ == "__main__":
    main()
