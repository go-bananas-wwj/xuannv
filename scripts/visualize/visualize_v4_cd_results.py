#!/usr/bin/env python3
"""
月度 Embedding 变化检测可视化脚本 (v6).
- embedding overview 保持大图
- 每个 patch 单独生成一张 1x8 子图, 放在 per_patch_examples/ 目录
- 统计图保持大图
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, f1_score

sys.path.insert(0, "/workspace/xuannv")

from demo_v2.engines.patch_image_loader import load_patch_source_rgb
from demo_v2.utils.constants import TIME_WINDOWS
from demo_v2.utils.harbin_annotations_v2 import (
    rasterize_patch_changes,
    rasterize_patch_categories,
    get_annotated_patches,
    get_period_for_patch,
    CATEGORY_TO_IDX,
    _build_patch_bounds,
)
from src.models.heads import ChangeDetectionHeadV3

EMBEDDING_DIR = Path("/workspace/outputs/aef_qwen_v4_official/monthly_embeddings_2025")
HEAD_PATH = Path("/workspace/outputs/aef_qwen_v4_official/monthly_cd_head/monthly_cd_head_v3_ohem.pt")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v4_official/visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PER_PATCH_DIR = OUTPUT_DIR / "04_per_patch_examples"
PER_PATCH_DIR.mkdir(parents=True, exist_ok=True)

# 真实图斑图片根目录
SPOT_BASE_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测图斑文件/jpg/jpg")

# 建立 Excel -> 图斑图片映射
import pandas as pd
from pyproj import Transformer

transformer_utm = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)

def wgs84_to_utm(lon, lat):
    return transformer_utm.transform(lon, lat)

def find_patch_for_point_utm(x, y, patch_bounds):
    for pid, (minx, miny, maxx, maxy) in patch_bounds.items():
        if minx <= x <= maxx and miny <= y <= maxy:
            return pid
    return None

def build_spot_mapping():
    xlsx_dir = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测清单")
    patch_bounds = _build_patch_bounds()
    period_mapping = {
        '4-6月份变化检测图斑.xlsx': ('2025-04~2025-06', {
            'before': ('6月/April', '4月{:02d}.jpg'),
            'after': ('6月/june', '6月{:02d}.jpg'),
        }),
        '6-8月份变化检测图斑.xlsx': ('2025-06~2025-08', {
            'before': ('8月/6月', '6月{:02d}.jpg'),
            'after': ('8月/8月', '8月{:02d}.jpg'),
        }),
        '8-9月份变化检测图斑.xlsx': ('2025-08~2025-09', {
            'before': ('9月/August', '8月{:02d}.jpg'),
            'after': ('9月/September', '9月{:02d}.jpg'),
        }),
        '9-10月份变化检测图斑.xlsx': ('2025-09~2025-10', {
            'before': ('10月/September', '9月{:02d}.jpg'),
            'after': ('10月/October', '10月{:02d}.jpg'),
        }),
    }
    mapping = {}
    for fname, (period, paths) in period_mapping.items():
        df = pd.read_excel(xlsx_dir / fname)
        for _, row in df.iterrows():
            try:
                fid = int(row['Id'])
                lon = float(row['经度'])
                lat = float(row['纬度'])
                remark = str(row.get('备注', ''))
            except Exception:
                continue
            x, y = wgs84_to_utm(lon, lat)
            pid = find_patch_for_point_utm(x, y, patch_bounds)
            if pid is None:
                continue
            before_dir, before_fmt = paths['before']
            after_dir, after_fmt = paths['after']
            before_jpg = SPOT_BASE_DIR / before_dir / before_fmt.format(fid)
            after_jpg = SPOT_BASE_DIR / after_dir / after_fmt.format(fid)
            if before_jpg.exists() and after_jpg.exists():
                if pid not in mapping:
                    mapping[pid] = []
                mapping[pid].append({
                    'period': period,
                    'id': fid,
                    'before_jpg': str(before_jpg),
                    'after_jpg': str(after_jpg),
                    'remark': remark,
                })
    return mapping

SPOT_MAPPING = build_spot_mapping()

MONTHS = ["2025-04", "2025-06", "2025-08", "2025-09", "2025-10"]
PERIOD_TO_MONTHS = {
    "2025-04~2025-06": ("2025-04", "2025-06"),
    "2025-06~2025-08": ("2025-06", "2025-08"),
    "2025-08~2025-09": ("2025-08", "2025-09"),
    "2025-09~2025-10": ("2025-09", "2025-10"),
}

CATEGORY_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (0.95, 0.15, 0.15, 0.80),   # construction -> 红
    2: (0.15, 0.35, 0.90, 0.80),   # demolition -> 蓝
    3: (1.00, 0.60, 0.05, 0.80),   # road -> 橙黄
    4: (0.05, 0.75, 0.95, 0.80),   # water_change -> 青
    5: (0.15, 0.80, 0.15, 0.80),   # farmland -> 绿
}
CATEGORY_NAMES = {
    0: "unchanged",
    1: "construction",
    2: "demolition",
    3: "road",
    4: "water_change",
    5: "farmland",
}


def load_head():
    state = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
    cfg = state["config"]
    head = ChangeDetectionHeadV3(
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.4),
    )
    head.load_state_dict(state["cd_head"])
    head.eval()
    return head, state["metrics"]["auc"]


def smooth_pca_to_rgb(emb: np.ndarray, out_size: int = 512) -> np.ndarray:
    """PCA -> RGB with per-channel percentile stretch and gamma correction."""
    D, H, W = emb.shape
    pca = PCA(n_components=3)
    flat = emb.reshape(D, -1).T
    rgb = pca.fit_transform(flat).T.reshape(3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))

    for c in range(3):
        ch = rgb[:, :, c]
        p1, p99 = np.percentile(ch, [1, 99])
        ch = (ch - p1) / (p99 - p1 + 1e-8)
        ch = np.clip(ch, 0, 1)
        ch = np.power(ch, 0.6)
        rgb[:, :, c] = ch

    rgb = (rgb * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb).resize((out_size, out_size), Image.Resampling.BICUBIC)
    return np.array(pil_img) / 255.0


def get_patch_category_label(patch_id: str) -> str:
    mask, _ = rasterize_patch_categories(patch_id, grid_size=64)
    cats = np.unique(mask)
    cats = cats[cats != 0]
    if len(cats) == 0:
        return "unknown"
    names = [CATEGORY_NAMES.get(int(c), f"cls{c}") for c in cats]
    return ",".join(names)


def get_primary_category(patch_id: str) -> int | None:
    mask, _ = rasterize_patch_categories(patch_id, grid_size=64)
    cats = np.unique(mask)
    cats = cats[cats != 0]
    if len(cats) == 0:
        return None
    return int(cats[0])


def load_s2_rgb(patch_id: str, month: str) -> np.ndarray | None:
    window = TIME_WINDOWS.get(month)
    if window is None:
        return None
    return load_patch_source_rgb(patch_id, "s2", window)


def load_hr_rgb(patch_id: str, month: str) -> np.ndarray | None:
    window = TIME_WINDOWS.get(month)
    if window is None:
        return None
    return load_patch_source_rgb(patch_id, "s2_hr", window)


def compute_benchmark(head):
    annotated = get_annotated_patches()
    head_recs = []
    raw_recs = []
    for pid in annotated:
        period = get_period_for_patch(pid)
        if period is None or period not in PERIOD_TO_MONTHS:
            continue
        bm, am = PERIOD_TO_MONTHS[period]
        bpath = EMBEDDING_DIR / f"{pid}_{bm}.npy"
        apath = EMBEDDING_DIR / f"{pid}_{am}.npy"
        if not bpath.exists() or not apath.exists():
            continue
        emb_b = np.load(bpath)
        emb_a = np.load(apath)
        mask, _ = rasterize_patch_changes(pid, grid_size=64)
        if mask.sum() == 0:
            continue
        labels = mask.flatten()

        with torch.no_grad():
            eb = torch.from_numpy(emb_b).unsqueeze(0).float()
            ea = torch.from_numpy(emb_a).unsqueeze(0).float()
            logits = head(eb, ea).squeeze(1)
            probs_head = torch.sigmoid(logits).squeeze().cpu().numpy().flatten()

        eb_t = torch.from_numpy(emb_b)
        ea_t = torch.from_numpy(emb_a)
        probs_raw = 1.0 - F.cosine_similarity(eb_t, ea_t, dim=0).cpu().numpy().flatten()

        try:
            auc_head = roc_auc_score(labels, probs_head)
        except Exception:
            auc_head = 0.5
        try:
            auc_raw = roc_auc_score(labels, probs_raw)
        except Exception:
            auc_raw = 0.5

        preds_head = (probs_head > 0.5).astype(int)
        ba_head = balanced_accuracy_score(labels, preds_head)
        f1_head = f1_score(labels, preds_head, zero_division=0)

        head_recs.append({
            "patch_id": pid,
            "period": period,
            "auc": float(auc_head),
            "ba": float(ba_head),
            "f1": float(f1_head),
        })
        raw_recs.append({"patch_id": pid, "period": period, "auc": float(auc_raw)})

    return head_recs, raw_recs


def plot_embedding_overview(patch_ids: list[str]):
    n_patches = len(patch_ids)
    fig, axes = plt.subplots(n_patches, len(MONTHS), figsize=(16, n_patches * 2.8))
    if n_patches == 1:
        axes = axes.reshape(1, -1)
    for i, pid in enumerate(patch_ids):
        for j, month in enumerate(MONTHS):
            ax = axes[i, j]
            path = EMBEDDING_DIR / f"{pid}_{month}.npy"
            if path.exists():
                emb = np.load(path)
                rgb = smooth_pca_to_rgb(emb, out_size=512)
                ax.imshow(rgb, interpolation="bilinear")
            else:
                ax.imshow(np.zeros((512, 512, 3)))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(month, fontsize=11, fontweight="bold")
            if j == 0:
                cat_label = get_patch_category_label(pid)
                ax.set_ylabel(f"{pid}\n({cat_label})", fontsize=9, rotation=0, ha="right", va="center")
    plt.suptitle("Monthly Embedding Overview (PCA -> RGB, contrast-enhanced)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUTPUT_DIR / "01_embedding_overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_single_patch_example(pid: str, head, head_recs_by_id: dict, raw_recs_by_id: dict, out_dir: Path):
    """为单个 patch 生成 1x8 子图; 如有真实图斑则额外生成带 spots 的 2 行版本."""
    rec = head_recs_by_id[pid]
    raw_rec = raw_recs_by_id[pid]
    period = rec["period"]
    bm, am = PERIOD_TO_MONTHS[period]

    emb_b = np.load(EMBEDDING_DIR / f"{pid}_{bm}.npy")
    emb_a = np.load(EMBEDDING_DIR / f"{pid}_{am}.npy")
    mask_bin, _ = rasterize_patch_changes(pid, grid_size=64)

    s2_b = load_s2_rgb(pid, bm)
    s2_a = load_s2_rgb(pid, am)
    hr_b = load_hr_rgb(pid, bm)
    hr_a = load_hr_rgb(pid, am)
    if s2_b is None:
        s2_b = np.zeros((512, 512, 3), dtype=np.float32)
    if s2_a is None:
        s2_a = np.zeros((512, 512, 3), dtype=np.float32)
    if hr_b is None:
        hr_b = np.zeros((512, 512, 3), dtype=np.float32)
    if hr_a is None:
        hr_a = np.zeros((512, 512, 3), dtype=np.float32)

    with torch.no_grad():
        eb = torch.from_numpy(emb_b).unsqueeze(0).float()
        ea = torch.from_numpy(emb_a).unsqueeze(0).float()
        logits = head(eb, ea).squeeze(1)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()

    rgb_b = smooth_pca_to_rgb(emb_b, out_size=512)
    rgb_a = smooth_pca_to_rgb(emb_a, out_size=512)

    titles = [
        "Before S2", "After S2", "Before HR", "After HR",
        "Before Emb", "After Emb", "GT Mask", "Head Pred",
    ]
    imgs = [s2_b, s2_a, hr_b, hr_a, rgb_b, rgb_a, mask_bin, pred]

    # --- 基础 1x8 版本 ---
    fig, axes = plt.subplots(1, 8, figsize=(18, 2.8))
    for j, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        if j == 2 or j == 3:
            ax.imshow(img, interpolation="bilinear")
            h, w = img.shape[:2]
            if h > 0 and w > 0:
                mask_hr = zoom(mask_bin.astype(float), (h / 64.0, w / 64.0), order=1)
                ax.contour(mask_hr, levels=[0.5], colors='lime', linewidths=1.5)
        elif j < 6:
            ax.imshow(img, interpolation="bilinear")
        elif j == 6:
            ax.imshow(img, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        else:
            pmin, pmax = img.min(), img.max()
            norm_img = (img - pmin) / (pmax - pmin + 1e-8) if pmax > pmin else img
            im = ax.imshow(norm_img, cmap="coolwarm", vmin=0, vmax=1, interpolation="bilinear")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("norm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")

    cat_label = get_patch_category_label(pid)
    head_auc = rec["auc"]
    raw_auc = raw_rec["auc"]
    fig.suptitle(f"{pid} | {cat_label} | Head AUC={head_auc:.3f} | Baseline AUC={raw_auc:.3f}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / f"{pid}.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    # --- 带真实图斑的 2x8 版本 ---
    spot_recs = SPOT_MAPPING.get(pid, [])
    if not spot_recs:
        return

    # 筛选与当前 period 匹配的第一个图斑
    matched_spot = None
    for sr in spot_recs:
        if sr["period"] == period:
            matched_spot = sr
            break
    if matched_spot is None:
        matched_spot = spot_recs[0]

    from PIL import Image
    spot_b = np.array(Image.open(matched_spot["before_jpg"]))
    spot_a = np.array(Image.open(matched_spot["after_jpg"]))

    fig, axes = plt.subplots(2, 8, figsize=(18, 5.8))
    # 第 1 行：原来的 8 列
    for j, (ax, img, title) in enumerate(zip(axes[0], imgs, titles)):
        if j == 2 or j == 3:
            ax.imshow(img, interpolation="bilinear")
            h, w = img.shape[:2]
            if h > 0 and w > 0:
                mask_hr = zoom(mask_bin.astype(float), (h / 64.0, w / 64.0), order=1)
                ax.contour(mask_hr, levels=[0.5], colors='lime', linewidths=1.5)
        elif j < 6:
            ax.imshow(img, interpolation="bilinear")
        elif j == 6:
            ax.imshow(img, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        else:
            pmin, pmax = img.min(), img.max()
            norm_img = (img - pmin) / (pmax - pmin + 1e-8) if pmax > pmin else img
            im = ax.imshow(norm_img, cmap="coolwarm", vmin=0, vmax=1, interpolation="bilinear")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("norm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")

    # 第 2 行：前 2 列放真实图斑
    axes[1, 0].imshow(spot_b, interpolation="bilinear")
    axes[1, 0].set_title("Before Spot (真实图斑)", fontsize=10, fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].imshow(spot_a, interpolation="bilinear")
    axes[1, 1].set_title("After Spot (真实图斑)", fontsize=10, fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    # 第 2 行第 3 列显示 remark
    axes[1, 2].axis("off")
    remark_text = matched_spot["remark"]
    spot_id_text = f"图斑 ID: {matched_spot['id']}\nPeriod: {matched_spot['period']}"
    axes[1, 2].text(0.1, 0.5, f"{remark_text}\n\n{spot_id_text}", transform=axes[1, 2].transAxes,
                    fontsize=11, va="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 其余列关闭
    for j in range(3, 8):
        axes[1, j].axis("off")

    fig.suptitle(f"{pid} | {cat_label} | Head AUC={head_auc:.3f} | Baseline AUC={raw_auc:.3f}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"{pid}_with_spots.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_category_examples_combined(category_name: str, patch_ids: list[str], head, head_recs_by_id: dict, raw_recs_by_id: dict):
    """按类别生成一张 overview 大图 (2x4 布局)."""
    n_examples = len(patch_ids)
    n_cols = 8
    fig, axes = plt.subplots(n_examples, n_cols, figsize=(19, n_examples * 2.6))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    titles = [
        "Before S2", "After S2", "Before HR", "After HR",
        "Before Emb", "After Emb", "GT Mask", "Head Pred",
    ]

    for i, pid in enumerate(patch_ids):
        rec = head_recs_by_id[pid]
        raw_rec = raw_recs_by_id[pid]
        period = rec["period"]
        bm, am = PERIOD_TO_MONTHS[period]

        emb_b = np.load(EMBEDDING_DIR / f"{pid}_{bm}.npy")
        emb_a = np.load(EMBEDDING_DIR / f"{pid}_{am}.npy")
        mask_bin, _ = rasterize_patch_changes(pid, grid_size=64)

        s2_b = load_s2_rgb(pid, bm)
        s2_a = load_s2_rgb(pid, am)
        hr_b = load_hr_rgb(pid, bm)
        hr_a = load_hr_rgb(pid, am)
        if s2_b is None:
            s2_b = np.zeros((512, 512, 3), dtype=np.float32)
        if s2_a is None:
            s2_a = np.zeros((512, 512, 3), dtype=np.float32)
        if hr_b is None:
            hr_b = np.zeros((512, 512, 3), dtype=np.float32)
        if hr_a is None:
            hr_a = np.zeros((512, 512, 3), dtype=np.float32)

        with torch.no_grad():
            eb = torch.from_numpy(emb_b).unsqueeze(0).float()
            ea = torch.from_numpy(emb_a).unsqueeze(0).float()
            logits = head(eb, ea).squeeze(1)
            pred = torch.sigmoid(logits).squeeze().cpu().numpy()

        rgb_b = smooth_pca_to_rgb(emb_b, out_size=512)
        rgb_a = smooth_pca_to_rgb(emb_a, out_size=512)

        imgs = [s2_b, s2_a, hr_b, hr_a, rgb_b, rgb_a, mask_bin, pred]
        for j, (ax, img, title) in enumerate(zip(axes[i], imgs, titles)):
            if j == 2 or j == 3:
                ax.imshow(img, interpolation="bilinear")
                h, w = img.shape[:2]
                if h > 0 and w > 0:
                    mask_hr = zoom(mask_bin.astype(float), (h / 64.0, w / 64.0), order=1)
                    ax.contour(mask_hr, levels=[0.5], colors='lime', linewidths=1.5)
            elif j < 6:
                ax.imshow(img, interpolation="bilinear")
            elif j == 6:
                ax.imshow(img, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
            else:
                pmin, pmax = img.min(), img.max()
                norm_img = (img - pmin) / (pmax - pmin + 1e-8) if pmax > pmin else img
                im = ax.imshow(norm_img, cmap="coolwarm", vmin=0, vmax=1, interpolation="bilinear")
                if i == 0:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=7)
                    cbar.set_label("norm", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            if j == 0:
                head_auc = rec["auc"]
                raw_auc = raw_rec["auc"]
                ax.set_ylabel(
                    f"{pid}\nH={head_auc:.3f} / B={raw_auc:.3f}",
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    va="center",
                )

    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[k][:3], label=CATEGORY_NAMES[k])
        for k in [1, 2, 5]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=9,
        title="GT Categories",
        title_fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    plt.suptitle(f"Change Detection Examples — {category_name.title()}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = OUTPUT_DIR / f"02_{category_name}_examples.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_performance_stats(head, head_recs: list[dict], raw_recs: list[dict]):
    head_recs = sorted(head_recs, key=lambda x: x["auc"])
    raw_recs_by_id = {r["patch_id"]: r for r in raw_recs}

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.26)

    ax1 = fig.add_subplot(gs[0, :])
    pids = [r["patch_id"] for r in head_recs]
    head_aucs = [r["auc"] for r in head_recs]
    raw_aucs = [raw_recs_by_id[pid]["auc"] for pid in pids]
    x = np.arange(len(pids))
    width = 0.35
    ax1.bar(x - width / 2, raw_aucs, width, label="Baseline (Cosine)", color="steelblue", alpha=0.8)
    ax1.bar(x + width / 2, head_aucs, width, label="Head (Best CV Fold3)", color="indianred", alpha=0.8)
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    ax1.set_ylabel("AUC", fontsize=11)
    ax1.set_title(f"Per-Patch AUC Comparison (n={len(pids)})", fontsize=12, fontweight="bold")
    ax1.set_xticks(x[::5])
    ax1.set_xticklabels([pids[i] for i in range(0, len(pids), 5)], rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(raw_aucs, head_aucs, alpha=0.6, edgecolors="black", linewidth=0.5, s=50)
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="y=x")
    ax2.set_xlabel("Baseline (Cosine) AUC", fontsize=11)
    ax2.set_ylabel("Head AUC", fontsize=11)
    ax2.set_title("Head vs Baseline AUC", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    improved = sum(h > r for h, r in zip(head_aucs, raw_aucs))
    ax2.text(
        0.05, 0.95,
        f"Improved: {improved}/{len(head_aucs)}",
        transform=ax2.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    ax3 = fig.add_subplot(gs[1, 1])
    metrics_data = [
        [r["auc"] for r in head_recs],
        [raw_recs_by_id[pid]["auc"] for pid in pids],
        [r["ba"] for r in head_recs],
        [r["f1"] for r in head_recs],
    ]
    bp = ax3.boxplot(
        metrics_data,
        tick_labels=["Head\nAUC", "Baseline\nAUC", "Head\nBA", "Head\nF1"],
        patch_artist=True,
    )
    colors = ["indianred", "steelblue", "lightgreen", "gold"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel("Score", fontsize=11)
    ax3.set_title("Metric Distributions", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1.05)

    ax4 = fig.add_subplot(gs[2, :])
    best_pid = head_recs[-1]["patch_id"]
    median_pid = head_recs[len(head_recs) // 2]["patch_id"]
    worst_pid = head_recs[0]["patch_id"]

    for pid, label, color in [
        (best_pid, "Best", "green"),
        (median_pid, "Median", "blue"),
        (worst_pid, "Worst", "red"),
    ]:
        rec = next(r for r in head_recs if r["patch_id"] == pid)
        period = rec["period"]
        bm, am = PERIOD_TO_MONTHS[period]
        emb_b = np.load(EMBEDDING_DIR / f"{pid}_{bm}.npy")
        emb_a = np.load(EMBEDDING_DIR / f"{pid}_{am}.npy")
        mask, _ = rasterize_patch_changes(pid, grid_size=64)

        with torch.no_grad():
            eb = torch.from_numpy(emb_b).unsqueeze(0).float()
            ea = torch.from_numpy(emb_a).unsqueeze(0).float()
            logits = head(eb, ea).squeeze(1)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy().reshape(-1)
        labels = mask.reshape(-1)
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        cat_label = get_patch_category_label(pid)
        ax4.plot(fpr, tpr, label=f"{pid} ({label}, {cat_label}, AUC={auc:.3f})", color=color, lw=2.2)

    ax4.plot([0, 1], [0, 1], "k--", lw=1)
    ax4.set_xlabel("False Positive Rate", fontsize=11)
    ax4.set_ylabel("True Positive Rate", fontsize=11)
    ax4.set_title("ROC Curves (Representative Patches)", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=9)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.suptitle("Model Performance Statistics", fontsize=16, fontweight="bold", y=0.995)
    out_path = OUTPUT_DIR / "03_performance_stats.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    head, head_auc_val = load_head()
    print(f"Head loaded from monthly_cd_head_v3_ohem.pt (Val AUC {head_auc_val:.4f})")

    print("Computing benchmark metrics for all patches...")
    head_recs, raw_recs = compute_benchmark(head)
    print(f"Evaluated {len(head_recs)} patches")
    head_mean_auc = np.mean([r["auc"] for r in head_recs])
    raw_mean_auc = np.mean([r["auc"] for r in raw_recs])
    print(f"  Head mean AUC (full set): {head_mean_auc:.4f}")
    print(f"  Baseline mean AUC (full set): {raw_mean_auc:.4f}")

    head_recs_by_id = {r["patch_id"]: r for r in head_recs}
    raw_recs_by_id = {r["patch_id"]: r for r in raw_recs}

    # 1. Embedding overview
    overview_patches = [
        "patch_000040", "patch_000062", "patch_000139",
        "patch_000217", "patch_000297", "patch_000350",
        "patch_000366", "patch_000386",
    ]
    plot_embedding_overview(overview_patches)

    # 2. Per-category top patches for combined and per-patch figures
    cat_to_top = {1: [], 2: [], 5: []}
    for rec in head_recs:
        pid = rec["patch_id"]
        cat_idx = get_primary_category(pid)
        if cat_idx in cat_to_top and len(cat_to_top[cat_idx]) < 4:
            cat_to_top[cat_idx].append(pid)

    for rec in sorted(head_recs, key=lambda x: x["auc"], reverse=True):
        pid = rec["patch_id"]
        cat_idx = get_primary_category(pid)
        if cat_idx in cat_to_top and pid not in cat_to_top[cat_idx] and len(cat_to_top[cat_idx]) < 4:
            cat_to_top[cat_idx].append(pid)

    # Generate combined category overviews and per-patch single figures
    example_pids = []
    for cat_idx, cat_name in {1: "construction", 2: "demolition", 5: "farmland"}.items():
        pids = cat_to_top[cat_idx]
        if pids:
            plot_category_examples_combined(cat_name, pids, head, head_recs_by_id, raw_recs_by_id)
            example_pids.extend(pids)

    # Remove duplicates while preserving order
    seen = set()
    unique_example_pids = []
    for pid in example_pids:
        if pid not in seen:
            seen.add(pid)
            unique_example_pids.append(pid)

    print(f"\nGenerating {len(unique_example_pids)} per-patch single figures...")
    for pid in unique_example_pids:
        plot_single_patch_example(pid, head, head_recs_by_id, raw_recs_by_id, PER_PATCH_DIR)
    print(f"Saved per-patch examples to {PER_PATCH_DIR}")

    # 3. Performance stats
    plot_performance_stats(head, head_recs, raw_recs)

    print(f"\nAll visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
