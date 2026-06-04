#!/usr/bin/env python3
"""基于 xuannv_show 的 MLP 下游任务模型，在新 embedding 上推理并生成全域 mosaic 大图.

参考前端展示风格:
- WorldCover: 7 类分类，ESA 官方配色
- Dynamic World: 5 类分类，Google 配色
- JRC Water: 2 类分类，蓝色配色
- Building Extraction: 2 类分类，红色配色

输出:
  /workspace/outputs/aef_qwen_v5_mixed_scale/domain_downstream_mlp/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import joblib
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from tqdm import tqdm

# ============ 配置 ============
MODEL_DIR = Path("/workspace/xuannv_show/backend/models")
EMB_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/monthly_embeddings_2026")
DATA_ROOT = Path("/workspace/raw/harbin_scenes")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/domain_downstream_mlp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
MONTH_LABELS = {
    "2026-01": "Jan", "2026-02": "Feb", "2026-03": "Mar",
    "2026-04": "Apr", "2026-05": "May"
}

PATCH_H, PATCH_W = 133, 134

# ============ 颜色配置 (参考前端) ============
# ESA WorldCover 官方配色
WORLDCOVER_COLORS = [
    [0, 100, 0],      # 0: Tree cover
    [255, 187, 34],   # 1: Shrubland
    [255, 255, 76],   # 2: Grassland
    [240, 150, 255],  # 3: Cropland
    [250, 0, 0],      # 4: Built-up
    [180, 180, 180],  # 5: Bare/sparse
    [240, 240, 240],  # 6: Snow/ice
    [0, 100, 200],    # 7: Water
    [0, 150, 160],    # 8: Wetland
    [0, 207, 117],    # 9: Mangroves
    [250, 230, 160],  # 10: Moss/lichen
]

# Dynamic World 配色
DYNAMIC_WORLD_COLORS = [
    [65, 155, 223],   # 0: Water
    [57, 125, 73],    # 1: Trees
    [136, 176, 83],   # 2: Grass
    [122, 135, 198],  # 3: Flooded veg
    [228, 150, 53],   # 4: Crops
    [223, 195, 90],   # 5: Shrub/scrub
    [196, 40, 27],    # 6: Built
    [165, 155, 143],  # 7: Bare
    [179, 159, 225],  # 8: Snow/ice
]

# JRC Water 配色
JRC_WATER_COLORS = [
    [200, 200, 200],  # 0: Non-water
    [0, 100, 200],    # 1: Water
]

# Building 配色
BUILDING_COLORS = [
    [200, 200, 200],  # 0: Non-building
    [250, 0, 0],      # 1: Building
]


# ============ Grid 映射 ============
def load_grid_mapping():
    patches = []
    for i in range(424):
        pid = f"patch_{i:06d}"
        tifs = list((DATA_ROOT / "s1" / pid).glob("*.tif"))
        if tifs:
            with rasterio.open(tifs[0]) as src:
                ulx = src.transform.c
                uly = src.transform.f
                patches.append((pid, ulx, uly))
    unique_y = sorted(set(p[2] for p in patches), reverse=True)
    unique_x = sorted(set(p[1] for p in patches))
    y_to_row = {y: i for i, y in enumerate(unique_y)}
    x_to_col = {x: i for i, x in enumerate(unique_x)}
    grid = {}
    for pid, ulx, uly in patches:
        grid[pid] = (y_to_row[uly], x_to_col[ulx])
    return grid


GRID = load_grid_mapping()
N_ROWS = max(v[0] for v in GRID.values()) + 1
N_COLS = max(v[1] for v in GRID.values()) + 1


# ============ 推理引擎 ============
class DownstreamMLPEngine:
    """加载 xuannv_show 的 MLP 模型并在新 embedding 上推理."""

    def __init__(self):
        self.models: dict[str, dict] = {}
        _MODEL_FILES = {
            "worldcover": "worldcover_linear_probe.pkl",
            "dynamic_world": "dynamic_world_sklearn_mlp.pkl",
            "jrc_water": "jrc_water_sklearn_mlp.pkl",
            "building_extraction": "building_sklearn_mlp.pkl",
        }
        for head_id, fname in _MODEL_FILES.items():
            path = MODEL_DIR / fname
            if path.exists():
                self.models[head_id] = joblib.load(path)
                print(f"  Loaded {head_id}: {fname}")
            else:
                print(f"  WARNING: {path} not found")

    def _postprocess(self, pred: np.ndarray, head_id: str) -> np.ndarray:
        """后处理去噪 (参考 segmentation_engine.py)."""
        if head_id not in self.models:
            return pred

        n_classes = len(self.models[head_id]["classes"])

        if n_classes == 2:
            foreground = (pred == 1).astype(np.uint8)
            foreground = ndimage.binary_opening(foreground, structure=np.ones((3, 3))).astype(np.uint8)
            labeled, num = ndimage.label(foreground)
            if num > 0:
                sizes = ndimage.sum(foreground, labeled, range(1, num + 1))
                remove_labels = np.where(sizes < 5)[0] + 1
                if len(remove_labels) > 0:
                    remove_mask = np.isin(labeled, remove_labels)
                    foreground[remove_mask] = 0
            result = np.where(foreground, 1, 0).astype(np.int32)
        else:
            def _majority(v: np.ndarray) -> int:
                vals, counts = np.unique(v, return_counts=True)
                return int(vals[np.argmax(counts)])
            result = ndimage.generic_filter(pred.astype(np.int32), _majority, size=3, mode="nearest")

        return result

    def infer(self, head_id: str, patch_id: str, month: str) -> np.ndarray | None:
        """推理分类图 [H, W]."""
        if head_id not in self.models:
            return None

        emb_path = EMB_DIR / f"{patch_id}_{month}.npy"
        if not emb_path.exists():
            return None

        emb = np.load(emb_path)  # [D, H, W]
        model_dict = self.models[head_id]
        scaler = model_dict["scaler"]
        lr = model_dict["model"]

        D, H, W = emb.shape
        flat = emb.reshape(D, -1).T  # [H*W, D]
        flat_s = scaler.transform(flat)
        pred = lr.predict(flat_s).reshape(H, W)
        pred = pred.astype(np.int32)
        pred = self._postprocess(pred, head_id)
        return pred


# ============ 拼接与保存 ============
def stitch_classification(preds: dict[str, np.ndarray]) -> np.ndarray:
    """将分类图拼接成全域大图."""
    canvas = np.full((N_ROWS * PATCH_H, N_COLS * PATCH_W), -1, dtype=np.int32)
    for pid, (row, col) in GRID.items():
        if pid not in preds:
            continue
        pred = preds[pid]
        y0 = row * PATCH_H
        x0 = col * PATCH_W
        h, w = pred.shape
        canvas[y0:y0+h, x0:x0+w] = pred[:min(h, PATCH_H), :min(w, PATCH_W)]
    return canvas


def save_classification_figure(canvas, title, out_path, colors, n_classes):
    """保存分类全域图."""
    # 创建颜色映射
    cmap = ListedColormap(np.array(colors[:n_classes]) / 255.0)

    fig, ax = plt.subplots(1, 1, figsize=(20, 18))
    im = ax.imshow(canvas, cmap=cmap, vmin=-0.5, vmax=n_classes - 0.5)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(n_classes))

    # 设置 colorbar 标签
    if "worldcover" in str(out_path).lower():
        labels = ["Tree", "Shrub", "Grass", "Crop", "Built", "Bare", "Snow", "Water", "Wetland", "Mangrove", "Moss"]
    elif "dynamic_world" in str(out_path).lower():
        labels = ["Water", "Trees", "Grass", "Flooded", "Crops", "Shrub", "Built", "Bare", "Snow"]
    elif "jrc_water" in str(out_path).lower():
        labels = ["Non-water", "Water"]
    elif "building" in str(out_path).lower():
        labels = ["Non-building", "Building"]
    else:
        labels = [str(i) for i in range(n_classes)]

    cbar.ax.set_yticklabels(labels[:n_classes])

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============ 主流程 ============
def main():
    print("=" * 60)
    print("下游任务 MLP 全域可视化")
    print("=" * 60)

    engine = DownstreamMLPEngine()

    tasks = [
        ("worldcover", WORLDCOVER_COLORS, 11),
        ("dynamic_world", DYNAMIC_WORLD_COLORS, 9),
        ("jrc_water", JRC_WATER_COLORS, 2),
        ("building_extraction", BUILDING_COLORS, 2),
    ]

    for head_id, colors, n_classes in tasks:
        if head_id not in engine.models:
            print(f"\nSkip {head_id}: model not loaded")
            continue

        print(f"\n[{head_id}] 推理中...")

        for month in MONTHS:
            label = MONTH_LABELS[month]
            print(f"  Month: {label}")

            preds = {}
            for pid in tqdm(GRID.keys(), desc=f"  {head_id} {label}", leave=False):
                pred = engine.infer(head_id, pid, month)
                if pred is not None:
                    preds[pid] = pred

            if not preds:
                print(f"    No predictions for {month}")
                continue

            canvas = stitch_classification(preds)
            out_path = OUTPUT_DIR / f"domain_{head_id}_{month}.png"
            save_classification_figure(
                canvas, f"{head_id.replace('_', ' ').title()}: {label} 2026",
                out_path, colors, n_classes
            )

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
