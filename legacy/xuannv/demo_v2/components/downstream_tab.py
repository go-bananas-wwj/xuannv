"""Tab 4: Downstream Tasks (non-change-detection)."""
from __future__ import annotations

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

matplotlib.use("Agg")

from demo_v2.cache_manager import cache
from demo_v2.utils.constants import RAW_DIR
from demo_v2.utils.visualization import fig_to_pil, colorize_worldcover


# ── 颜色表 ──
DYNAMIC_WORLD_CLASSES = {
    0: "Water", 1: "Trees", 2: "Grass", 3: "Flooded Veg",
    4: "Crops", 5: "Shrub/Scrub", 6: "Built", 7: "Bare", 8: "Snow/Ice",
}
DYNAMIC_WORLD_COLORS = {
    0: (0, 100, 200), 1: (0, 100, 0), 2: (136, 176, 83),
    3: (120, 180, 160), 4: (255, 187, 34), 5: (255, 150, 50),
    6: (250, 0, 0), 7: (180, 180, 180), 8: (222, 222, 222),
}

WORLDCOVER_6CLASS_MAP = {
    10: (0, "Tree"), 20: (1, "Shrubland"), 30: (2, "Grassland"),
    40: (3, "Cropland"), 50: (4, "Built-up"), 60: (5, "Bare"),
    80: (6, "Water"), 90: (7, "Wetland"),
}
WORLDCOVER_6CLASS_COLORS = {
    0: (0, 100, 0), 1: (255, 187, 34), 2: (136, 176, 83),
    3: (255, 187, 34), 4: (250, 0, 0), 5: (180, 180, 180),
    6: (0, 100, 200), 7: (120, 180, 160),
}

WORLDCOVER_ALL_COLORS = {
    10: (65, 155, 223), 20: (57, 125, 73), 30: (136, 176, 83),
    40: (255, 187, 34), 50: (255, 255, 76), 60: (187, 85, 29),
    70: (222, 222, 222), 80: (170, 170, 170), 90: (120, 80, 20),
    95: (140, 140, 140), 100: (100, 100, 100),
}


def _compute_grid_layout():
    metas = cache.patch_metas
    if not metas:
        return {}, 0, 0
    xs = sorted(set(round(m.bounds[0], 1) for m in metas))
    ys = sorted(set(round(m.bounds[1], 1) for m in metas))
    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: len(ys) - 1 - i for i, y in enumerate(ys)}
    layout = {}
    for m in metas:
        layout[m.patch_id] = (y_to_row[round(m.bounds[1], 1)], x_to_col[round(m.bounds[0], 1)])
    return layout, len(ys), len(xs)


def _stitch_grid(patch_maps, layout, n_rows, n_cols, fill=0):
    sample = next(iter(patch_maps.values()))
    ph, pw = sample.shape[:2]
    ndim = sample.ndim
    if ndim == 3:
        canvas = np.full((n_rows * ph, n_cols * pw, sample.shape[2]), fill, dtype=sample.dtype)
    else:
        canvas = np.full((n_rows * ph, n_cols * pw), fill, dtype=sample.dtype)
    for pid, arr in patch_maps.items():
        if pid not in layout:
            continue
        r, c = layout[pid]
        y0, x0 = r * ph, c * pw
        canvas[y0:y0 + ph, x0:x0 + pw] = arr
    return canvas


def _render_fullarea_task(
    version: str,
    label_source: str,
    class_map: dict[int, str],
    color_map: dict[int, tuple[int, int, int]],
    task_name: str,
    is_binary: bool = False,
    jrc_mode: bool = False,
) -> tuple[str, Image.Image | None]:
    """通用全区域下游任务：从 embedding map + label tif 做逐像素分类并拼接大图。"""
    import rasterio

    emb_maps = cache.embedding_maps.get(version)
    if emb_maps is None:
        return f"⚠️ {version} 无预计算 embedding map", None

    ids = cache.embedding_map_patch_ids.get(version, cache.patch_ids)
    layout, n_rows, n_cols = _compute_grid_layout()
    if not layout:
        return "⚠️ 网格布局计算失败", None

    all_emb, all_lbl = [], []
    patch_labels = {}
    rng = np.random.RandomState(42)
    sorted_classes = sorted(class_map.keys())
    cls_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    n_classes = len(sorted_classes)
    class_names = [class_map[c] for c in sorted_classes]

    for i, pid in enumerate(ids):
        if i >= emb_maps.shape[0]:
            break
        label_dir = RAW_DIR / label_source / pid
        if not label_dir.exists():
            continue
        tifs = sorted(label_dir.glob("*.tif"))
        if not tifs:
            continue

        with rasterio.open(str(tifs[0])) as src:
            raw_lbl = src.read(1)

        raw_int = raw_lbl.astype(np.int32)
        if np.issubdtype(raw_lbl.dtype, np.floating):
            nan_mask = np.isnan(raw_lbl)
            raw_int[nan_mask] = -1

        # JRC Water 特殊处理
        if jrc_mode:
            mapped = np.full_like(raw_int, fill_value=-1)
            mapped[raw_int == -128] = -1
            mapped[raw_int <= 0] = 0
            mapped[raw_int > 0] = 1
        else:
            mapped = np.full_like(raw_int, fill_value=-1)
            for orig, idx in cls_to_idx.items():
                mapped[raw_int == orig] = idx

        D, H, W = emb_maps[i].shape
        lbl_h, lbl_w = mapped.shape
        if lbl_h != H or lbl_w != W:
            from PIL import Image as PILImage
            lbl_pil = PILImage.fromarray(mapped.astype(np.int32))
            lbl_pil = lbl_pil.resize((W, H), PILImage.NEAREST)
            lbl_ds = np.array(lbl_pil, dtype=np.int32)
        else:
            lbl_ds = mapped

        patch_labels[pid] = lbl_ds
        valid = lbl_ds >= 0
        if valid.sum() == 0:
            continue
        emb_map = emb_maps[i]
        flat_emb = emb_map[:, valid].T
        flat_lbl = lbl_ds[valid]
        n = flat_emb.shape[0]
        if n > 300:
            idx = rng.choice(n, 300, replace=False)
            flat_emb = flat_emb[idx]
            flat_lbl = flat_lbl[idx]
        all_emb.append(flat_emb)
        all_lbl.append(flat_lbl)

    if not all_emb:
        return f"⚠️ 无有效标签数据 ({label_source})", None

    X = np.concatenate(all_emb)
    y = np.concatenate(all_lbl)
    if len(X) < 50:
        return f"⚠️ 有效样本不足 ({len(X)} 个)", None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Linear Probe
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_s, y)

    # Train/test split eval
    n_total = len(X)
    n_train = int(n_total * 0.7)
    split_idx = rng.permutation(n_total)
    lr_eval = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr_eval.fit(X_s[split_idx[:n_train]], y[split_idx[:n_train]])
    y_pred = lr_eval.predict(X_s[split_idx[n_train:]])
    bacc = balanced_accuracy_score(y[split_idx[n_train:]], y_pred)
    avg = "binary" if is_binary else "macro"
    f1 = f1_score(y[split_idx[n_train:]], y_pred, average=avg)

    # 逐 patch 预测
    pred_patches = {}
    D = emb_maps.shape[1]
    H, W = emb_maps.shape[2], emb_maps.shape[3]
    for pid, gt_map in patch_labels.items():
        if pid not in ids:
            continue
        idx_p = ids.index(pid)
        if idx_p >= emb_maps.shape[0]:
            continue
        flat = emb_maps[idx_p].reshape(D, -1).T
        flat_s = scaler.transform(flat)
        pred = lr.predict(flat_s).reshape(H, W)
        pred[gt_map < 0] = -1
        pred_patches[pid] = pred

    def _render_rgb(cls_map):
        rgb = np.full((*cls_map.shape, 3), 30, dtype=np.uint8)
        for idx, color in enumerate(color_map.values()):
            rgb[cls_map == idx] = color
        return rgb

    gt_rgb = {pid: _render_rgb(lbl) for pid, lbl in patch_labels.items()}
    pred_rgb = {pid: _render_rgb(p) for pid, p in pred_patches.items()}

    gt_canvas = _stitch_grid(gt_rgb, layout, n_rows, n_cols, fill=30)
    pred_canvas = _stitch_grid(pred_rgb, layout, n_rows, n_cols, fill=30)

    fig_w = max(12, n_cols * 0.3)
    fig_h = max(5, n_rows * 0.15)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h), dpi=120)
    axes[0].imshow(gt_canvas)
    axes[0].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(pred_canvas)
    axes[1].set_title("Prediction (Linear Probe)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(list(color_map.values())[i]) / 255, label=class_names[i])
        for i in range(n_classes)
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(n_classes, 8), fontsize=8, framealpha=0.9)

    fig.suptitle(f"{task_name} ({version})\nBalanced Acc: {bacc:.4f} | F1 ({avg}): {f1:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    img = fig_to_pil(fig)
    plt.close(fig)

    report = (
        f"### {task_name}\n\n"
        f"- **Model**: {version}\n"
        f"- **Classifier**: Linear Probe\n"
        f"- **Total pixels**: {len(X):,}\n"
        f"- **Classes**: {n_classes} ({', '.join(class_names)})\n"
        f"- **Balanced Accuracy**: {bacc:.4f}\n"
        f"- **F1 ({avg})**: {f1:.4f}\n"
        f"- **Patches**: {len(pred_patches)} / {len(ids)}\n"
    )
    return report, img


def _run_worldcover_eval(version: str):
    # 使用原始 WorldCover 11 类
    class_map = {
        10: "Tree", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
        50: "Built-up", 60: "Bare", 70: "Snow", 80: "Water", 90: "Wetland",
        95: "Mangroves", 100: "Moss",
    }
    return _render_fullarea_task(version, "worldcover", class_map, WORLDCOVER_ALL_COLORS,
                                  "WorldCover Land Cover Classification")


def _run_dynamic_world_eval(version: str):
    return _render_fullarea_task(version, "dynamic_world", DYNAMIC_WORLD_CLASSES,
                                  DYNAMIC_WORLD_COLORS,
                                  "Dynamic World Land Use Classification")


def _run_jrc_water_eval(version: str):
    class_map = {0: "Non-water", 1: "Water"}
    colors = {0: (180, 180, 180), 1: (0, 100, 200)}
    return _render_fullarea_task(version, "jrc_water", class_map, colors,
                                  "JRC Water Body Extraction", is_binary=True, jrc_mode=True)


def _run_building_extraction_eval(version: str):
    """建筑物提取：利用 WorldCover Built-up (50) 作为正类."""
    import rasterio
    emb_maps = cache.embedding_maps.get(version)
    if emb_maps is None:
        return "⚠️ 无 embedding map", None
    ids = cache.embedding_map_patch_ids.get(version, cache.patch_ids)
    layout, n_rows, n_cols = _compute_grid_layout()
    if not layout:
        return "⚠️ 网格布局计算失败", None

    all_emb, all_lbl = [], []
    patch_labels = {}
    rng = np.random.RandomState(42)
    class_map = {0: "Non-building", 1: "Building"}
    colors = {0: (100, 100, 100), 1: (250, 0, 0)}

    for i, pid in enumerate(ids):
        if i >= emb_maps.shape[0]:
            break
        wc_dir = RAW_DIR / "worldcover" / pid
        if not wc_dir.exists():
            continue
        tifs = sorted(wc_dir.glob("*.tif"))
        if not tifs:
            continue
        with rasterio.open(str(tifs[0])) as src:
            wc = src.read(1)
        binary = np.full_like(wc, fill_value=-1, dtype=np.int32)
        binary[wc == 50] = 1
        binary[(wc >= 0) & (wc != 50)] = 0

        D, H, W = emb_maps[i].shape
        if binary.shape[0] != H or binary.shape[1] != W:
            from PIL import Image as PILImage
            lbl_pil = PILImage.fromarray(binary.astype(np.int32))
            lbl_pil = lbl_pil.resize((W, H), PILImage.NEAREST)
            binary = np.array(lbl_pil, dtype=np.int32)

        patch_labels[pid] = binary
        valid = binary >= 0
        if valid.sum() == 0:
            continue
        emb_map = emb_maps[i]
        flat_emb = emb_map[:, valid].T
        flat_lbl = binary[valid]
        n = flat_emb.shape[0]
        if n > 300:
            idx = rng.choice(n, 300, replace=False)
            flat_emb = flat_emb[idx]
            flat_lbl = flat_lbl[idx]
        all_emb.append(flat_emb)
        all_lbl.append(flat_lbl)

    if not all_emb:
        return "⚠️ 无有效建筑物标签数据", None
    X = np.concatenate(all_emb)
    y = np.concatenate(all_lbl)
    if len(X) < 50:
        return f"⚠️ 有效样本不足 ({len(X)} 个)", None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_s, y)

    n_total = len(X)
    n_train = int(n_total * 0.7)
    split_idx = rng.permutation(n_total)
    lr_eval = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr_eval.fit(X_s[split_idx[:n_train]], y[split_idx[:n_train]])
    y_pred = lr_eval.predict(X_s[split_idx[n_train:]])
    bacc = balanced_accuracy_score(y[split_idx[n_train:]], y_pred)
    f1 = f1_score(y[split_idx[n_train:]], y_pred, average="binary")
    iou = jaccard_score(y[split_idx[n_train:]], y_pred, average="binary")

    D, H, W = emb_maps.shape[1], emb_maps.shape[2], emb_maps.shape[3]
    pred_patches = {}
    for pid, gt_map in patch_labels.items():
        if pid not in ids:
            continue
        idx_p = ids.index(pid)
        if idx_p >= emb_maps.shape[0]:
            continue
        flat = emb_maps[idx_p].reshape(D, -1).T
        flat_s = scaler.transform(flat)
        pred = lr.predict(flat_s).reshape(H, W)
        pred[gt_map < 0] = -1
        pred_patches[pid] = pred

    def _render_rgb(cls_map):
        rgb = np.full((*cls_map.shape, 3), 30, dtype=np.uint8)
        rgb[cls_map == 0] = colors[0]
        rgb[cls_map == 1] = colors[1]
        return rgb

    gt_rgb = {pid: _render_rgb(lbl) for pid, lbl in patch_labels.items()}
    pred_rgb = {pid: _render_rgb(p) for pid, p in pred_patches.items()}
    gt_canvas = _stitch_grid(gt_rgb, layout, n_rows, n_cols, fill=30)
    pred_canvas = _stitch_grid(pred_rgb, layout, n_rows, n_cols, fill=30)

    fig_w = max(12, n_cols * 0.3)
    fig_h = max(5, n_rows * 0.15)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h), dpi=120)
    axes[0].imshow(gt_canvas)
    axes[0].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(pred_canvas)
    axes[1].set_title("Prediction (Linear Probe)", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(colors[0]) / 255, label="Non-building"),
        Patch(facecolor=np.array(colors[1]) / 255, label="Building"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9, framealpha=0.9)
    fig.suptitle(f"Building Extraction ({version})\nBalanced Acc: {bacc:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    img = fig_to_pil(fig)
    plt.close(fig)

    report = (
        f"### Building Extraction\n\n"
        f"- **Model**: {version}\n"
        f"- **Classifier**: Linear Probe\n"
        f"- **Total pixels**: {len(X):,}\n"
        f"- **Balanced Accuracy**: {bacc:.4f}\n"
        f"- **F1**: {f1:.4f}\n"
        f"- **IoU**: {iou:.4f}\n"
        f"- **Patches**: {len(pred_patches)} / {len(ids)}\n"
    )
    return report, img


def _run_spatial_prediction(version: str, patch_id: str):
    """单 Patch 空间预测：S2 RGB | WorldCover GT | Prediction."""
    import rasterio
    if not patch_id or patch_id not in cache.patch_ids:
        return "❌ 无效的 Patch ID", None

    emb_map = cache.get_embedding_map(version, patch_id)
    if emb_map is None:
        return f"⚠️ {version} / {patch_id} 无 embedding map", None

    # 训练数据：从所有 patch 的 WorldCover 采样
    ids = cache.embedding_map_patch_ids.get(version, cache.patch_ids)
    all_emb, all_lbl = [], []
    rng = np.random.RandomState(42)
    CLASS_NAMES = ["Tree", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare"]
    WC_COLORS_RGB = {
        0: (0, 100, 0), 1: (255, 187, 34), 2: (136, 176, 83),
        3: (255, 187, 34), 4: (250, 0, 0), 5: (180, 180, 180),
    }
    CLASS_MAP = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}

    for i, pid in enumerate(ids):
        if i >= cache.embedding_maps[version].shape[0]:
            break
        wc_dir = RAW_DIR / "worldcover" / pid
        if not wc_dir.exists():
            continue
        tifs = sorted(wc_dir.glob("*.tif"))
        if not tifs:
            continue
        with rasterio.open(str(tifs[0])) as src:
            wc = src.read(1)
        mapped = np.full_like(wc, fill_value=-1, dtype=np.int32)
        for orig, new in CLASS_MAP.items():
            mapped[wc == orig] = new

        D, H, W = cache.embedding_maps[version][i].shape
        if mapped.shape[0] != H or mapped.shape[1] != W:
            from PIL import Image as PILImage
            lbl_pil = PILImage.fromarray(mapped.astype(np.int32))
            lbl_pil = lbl_pil.resize((W, H), PILImage.NEAREST)
            mapped = np.array(lbl_pil, dtype=np.int32)

        valid = mapped >= 0
        if valid.sum() == 0:
            continue
        emb = cache.embedding_maps[version][i]
        flat_emb = emb[:, valid].T
        flat_lbl = mapped[valid]
        n = flat_emb.shape[0]
        if n > 300:
            idx = rng.choice(n, 300, replace=False)
            flat_emb = flat_emb[idx]
            flat_lbl = flat_lbl[idx]
        all_emb.append(flat_emb)
        all_lbl.append(flat_lbl)

    if not all_emb:
        return "⚠️ 无有效训练数据", None
    X = np.concatenate(all_emb)
    y = np.concatenate(all_lbl)
    if len(X) == 0:
        return "⚠️ 无有效训练样本", None

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_s, y)

    # 预测当前 patch
    D, H, W = emb_map.shape
    flat = emb_map.transpose(1, 2, 0).reshape(-1, D)
    flat_s = scaler.transform(flat)
    pred_flat = lr.predict(flat_s)
    pred_map = pred_flat.reshape(H, W)

    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx, color in WC_COLORS_RGB.items():
        pred_rgb[pred_map == cls_idx] = color

    # 加载 S2 RGB
    patch_dir = cache.get_patch_dir(patch_id)
    s2_dir = patch_dir / "s2"
    s2_rgb = None
    if s2_dir.exists():
        s2_files = sorted(s2_dir.glob("*.tif"))
        if s2_files:
            try:
                with rasterio.open(str(s2_files[0])) as src:
                    data = src.read()
                if data.shape[0] >= 3:
                    rgb = data[[2, 1, 0]].astype(np.float32)
                    valid = rgb[rgb > 0]
                    if len(valid) > 0:
                        p2, p98 = np.percentile(valid, [2, 98])
                        if p98 > p2:
                            rgb = (rgb - p2) / (p98 - p2)
                    s2_rgb = np.clip(rgb, 0, 1).transpose(1, 2, 0)
            except Exception:
                pass

    # 加载 WorldCover GT
    wc_dir = RAW_DIR / "worldcover" / patch_id
    gt_rgb = None
    if wc_dir.exists():
        wc_files = sorted(wc_dir.glob("*.tif"))
        if wc_files:
            try:
                with rasterio.open(str(wc_files[0])) as src:
                    wc = src.read(1)
                gt_rgb = colorize_worldcover(wc)
            except Exception:
                pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    if s2_rgb is not None:
        axes[0].imshow(s2_rgb)
    else:
        axes[0].text(0.5, 0.5, "No S2", ha="center", va="center", fontsize=12)
    axes[0].set_title("Sentinel-2 RGB", fontsize=11)
    axes[0].axis("off")

    if gt_rgb is not None:
        axes[1].imshow(gt_rgb)
    else:
        axes[1].text(0.5, 0.5, "No GT", ha="center", va="center", fontsize=12)
    axes[1].set_title("WorldCover GT", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title(f"Prediction ({version})", fontsize=11)
    axes[2].axis("off")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(c) / 255, label=CLASS_NAMES[i])
        for i, c in WC_COLORS_RGB.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(CLASS_NAMES), fontsize=8, framealpha=0.9)
    fig.suptitle(f"Spatial Prediction — {patch_id}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    img = fig_to_pil(fig)
    plt.close(fig)

    report = (
        f"**Patch**: {patch_id}  |  **Model**: {version}\n\n"
        f"Embedding map: {D}D x {H}x{W}  |  Training pixels: {len(X):,}"
    )
    return report, img


def build_downstream_tab() -> None:
    """构建 Tab 4: 下游任务."""
    gr.Markdown("## Downstream Tasks\n基于预训练 embedding 的像素级分类下游任务评估。")

    available_versions = list(cache.embedding_maps.keys())
    default_version = available_versions[0] if available_versions else "v1"

    version_select = gr.Dropdown(choices=available_versions, value=default_version, label="模型版本")

    with gr.Tabs():
        with gr.Tab("WorldCover Classification"):
            btn_wc = gr.Button("Run WorldCover Evaluation", variant="primary")
            with gr.Row():
                wc_report = gr.Markdown()
                wc_img = gr.Image(label="WorldCover Full-Area Prediction", height=600)
            btn_wc.click(fn=_run_worldcover_eval, inputs=[version_select], outputs=[wc_report, wc_img])

        with gr.Tab("Dynamic World Classification"):
            btn_dw = gr.Button("Run Dynamic World Evaluation", variant="primary")
            with gr.Row():
                dw_report = gr.Markdown()
                dw_img = gr.Image(label="Dynamic World Full-Area Prediction", height=600)
            btn_dw.click(fn=_run_dynamic_world_eval, inputs=[version_select], outputs=[dw_report, dw_img])

        with gr.Tab("JRC Water Extraction"):
            btn_jrc = gr.Button("Run JRC Water Evaluation", variant="primary")
            with gr.Row():
                jrc_report = gr.Markdown()
                jrc_img = gr.Image(label="JRC Water Full-Area Prediction", height=600)
            btn_jrc.click(fn=_run_jrc_water_eval, inputs=[version_select], outputs=[jrc_report, jrc_img])

        with gr.Tab("Building Extraction"):
            btn_building = gr.Button("Run Building Extraction", variant="primary")
            with gr.Row():
                building_report = gr.Markdown()
                building_img = gr.Image(label="Building Extraction Full-Area Prediction", height=600)
            btn_building.click(fn=_run_building_extraction_eval, inputs=[version_select], outputs=[building_report, building_img])

        with gr.Tab("Spatial Prediction (Single Patch)"):
            with gr.Row():
                with gr.Column(scale=1):
                    sp_patch = gr.Textbox(label="Patch ID", placeholder="e.g. patch_000123")
                    btn_sp = gr.Button("Run Spatial Prediction", variant="primary")
                with gr.Column(scale=3):
                    sp_report = gr.Markdown()
                    sp_img = gr.Image(label="S2 RGB | GT | Prediction", height=500)
            btn_sp.click(fn=_run_spatial_prediction, inputs=[version_select, sp_patch], outputs=[sp_report, sp_img])
