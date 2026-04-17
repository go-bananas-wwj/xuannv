"""Tab: Spatial Anomaly Detection.

每个像素与 patch 内均值 embedding 的 cosine distance。
不是变化检测：只需选择单个 patch + 单个时间窗口。
"""
from __future__ import annotations

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from demo_v2.cache_manager import cache
from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.utils.constants import RAW_DIR, SOURCE_DISPLAY_NAMES
from demo_v2.utils.map_utils import render_folium_map
from demo_v2.utils.visualization import fig_to_pil


def _load_s2_rgb(patch_id: str) -> Image.Image | None:
    """加载该 patch 的代表性 S2 RGB 影像."""
    import rasterio
    from pathlib import Path
    from datetime import datetime

    s2_dir = RAW_DIR / "s2" / patch_id
    if not s2_dir.exists():
        return None
    files = sorted(s2_dir.glob("*.tif"))
    if not files:
        return None

    # 取中间日期的文件作为代表
    mid_idx = len(files) // 2
    frames = []
    for f in files[max(0, mid_idx - 2):mid_idx + 3]:
        try:
            with rasterio.open(str(f)) as ds:
                data = ds.read()
            if data.shape[0] >= 3:
                rgb = data[[2, 1, 0]].astype(np.float32)
                frames.append(rgb)
        except Exception:
            pass

    if not frames:
        return None

    img = np.median(frames, axis=0)
    valid = img[img > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        if p98 > p2:
            img = (img - p2) / (p98 - p2)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img.transpose(1, 2, 0))


def _load_worldcover(patch_id: str) -> Image.Image | None:
    """加载 WorldCover 彩色图."""
    import rasterio
    from demo_v2.utils.visualization import colorize_worldcover

    wc_dir = RAW_DIR / "worldcover" / patch_id
    if not wc_dir.exists():
        return None
    tifs = sorted(wc_dir.glob("*.tif"))
    if not tifs:
        return None
    try:
        with rasterio.open(str(tifs[0])) as ds:
            wc = ds.read(1)
        rgb = colorize_worldcover(wc)
        return Image.fromarray(rgb)
    except Exception:
        return None


def _run_anomaly_detection(patch_id: str, version: str, time_preset: str):
    """运行空间异常检测."""
    from demo_v2.utils.constants import TIME_WINDOWS

    if patch_id not in cache.patch_ids:
        return [None] * 3 + ["❌ Invalid Patch ID"]

    window = TIME_WINDOWS.get(time_preset)
    if window is None:
        return [None] * 3 + ["❌ Invalid time window"]

    # 提取 embedding
    engine = ChangeDetectionEngine(version)
    emb = engine.get_embedding(patch_id, window[0], window[1], use_precomputed=True)
    if emb is None:
        return [None] * 3 + ["❌ Embedding extraction failed"]

    D, H, W = emb.shape
    # L2 normalize per pixel
    flat = emb.reshape(D, -1)
    norms = np.linalg.norm(flat, axis=0, keepdims=True)
    flat_norm = flat / np.maximum(norms, 1e-8)

    # patch mean
    mean_vec = flat_norm.mean(axis=1, keepdims=True)
    mean_norm = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

    # cosine similarity -> anomaly score
    cos_sim = np.sum(flat_norm * mean_norm, axis=0).reshape(H, W)
    anomaly = (1.0 - cos_sim) / 2.0  # [0, 1]

    # 渲染热力图（带 colorbar）
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=120)
    vmax = 0.5  # 固定上限，与旧 demo 一致
    im = ax.imshow(anomaly, cmap="hot", vmin=0.0, vmax=vmax)
    ax.set_title("Spatial Anomaly Heatmap", fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap = fig_to_pil(fig)
    plt.close(fig)

    s2_img = _load_s2_rgb(patch_id)
    wc_img = _load_worldcover(patch_id)

    stats = (
        f"### Spatial Anomaly — {patch_id}\n\n"
        f"| 参数 | 值 |\n|------|-----|\n"
        f"| Model | {version} |\n"
        f"| Time Window | {time_preset} |\n"
        f"| Mean Anomaly | {anomaly.mean():.4f} |\n"
        f"| Max Anomaly | {anomaly.max():.4f} |\n"
        f"| Anomalous Pixels (>0.3) | {int((anomaly > 0.3).sum())} |\n\n"
        f"**说明**: 每个像素表示其与 patch 平均 embedding 的偏离程度。"
        f"亮色区域 = 空间异常（与周围环境差异大）。"
    )

    return [s2_img, heatmap, wc_img, stats]


def build_spatial_anomaly_tab() -> None:
    """构建 Spatial Anomaly Detection Tab."""
    gr.Markdown(
        "## 🔍 Spatial Anomaly Detection\n"
        "检测单个 patch 在某一时间窗口内的空间异常像素。"
        "只需选择 **一个** 时间窗口，无需 before/after 对比。"
    )

    from demo_v2.utils.constants import TIME_WINDOWS
    time_presets = list(TIME_WINDOWS.keys())
    versions = ["v1", "v2", "v3"]

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Interactive Map  \n*Click a patch rectangle to auto-fill Patch ID*")
            folium_html = gr.HTML(value=render_folium_map(cache.patch_metas))

            patch_select = gr.Textbox(
                label="Patch ID",
                placeholder="Click a patch on the map above...",
                value="",
            )
            version_select = gr.Dropdown(
                choices=versions,
                value="v2",
                label="Model Version",
            )
            time_select = gr.Dropdown(
                choices=time_presets,
                value="2025 全年",
                label="Time Window",
            )
            btn_run = gr.Button("🚀 Run Anomaly Detection", variant="primary")

        with gr.Column(scale=3):
            with gr.Row():
                s2_img = gr.Image(label="S2 RGB", height=300)
                anomaly_img = gr.Image(label="Spatial Anomaly Heatmap", height=300)
                wc_img = gr.Image(label="WorldCover", height=300)
            stats_md = gr.Markdown()

    btn_run.click(
        fn=_run_anomaly_detection,
        inputs=[patch_select, version_select, time_select],
        outputs=[s2_img, anomaly_img, wc_img, stats_md],
    )

    # Timer: auto-sync Folium click to Patch ID (client-side only)
    _sa_timer = gr.Timer(0.5, active=True)
    _sa_timer.tick(
        fn=None,
        inputs=[patch_select],
        outputs=[patch_select],
        js="(x) => { var p = window._aef_pending_patch; if (p) { window._aef_pending_patch = null; return p; } return x; }",
    )
