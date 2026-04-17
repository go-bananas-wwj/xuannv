"""Tab 3: Change Detection — per-patch real-time + global precomputed."""
from __future__ import annotations

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from demo_v2.cache_manager import cache
from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.engines.patch_image_loader import (
    load_patch_source_rgb,
    load_patch_source_raw,
    compute_ndvi_from_s2,
    _find_best_tif,
)
from demo_v2.utils.constants import TIME_WINDOWS, RAW_DIR
from demo_v2.utils.map_utils import render_folium_map
from demo_v2.utils.visualization import (
    change_heatmap_fig,
    overlay_rgb_heatmap,
    binary_change_map,
    ndvi_delta_map,
    fig_to_pil,
)

# 预计算数据目录
PRECOMPUTE_DIR = __file__.replace("components/change_detection_tab.py", "precomputed_cd")
PRECOMPUTE_DIR = PRECOMPUTE_DIR.replace("components\\change_detection_tab.py", "precomputed_cd")
from pathlib import Path
PRECOMPUTE_BASE = Path(PRECOMPUTE_DIR)

# 常用组合定义 (name, before_key, after_key)
COMMON_PAIRS = [
    ("2023 Full Year vs 2024 Full Year", "2023 全年", "2024 全年"),
    ("2024 Full Year vs 2025 Full Year", "2024 全年", "2025 全年"),
    ("2023 Q1-Q2 vs 2024 Q1-Q2", "2023 Q1-Q2", "2024 Q1-Q2"),
    ("2023 Q3-Q4 vs 2024 Q3-Q4", "2023 Q3-Q4", "2024 Q3-Q4"),
    ("2024 Q1-Q2 vs 2025 Q1-Q2", "2024 Q1-Q2", "2025 Q1-Q2"),
    ("2024 Q3-Q4 vs 2025 Q3-Q4", "2024 Q3-Q4", "2025 Q3-Q4"),
    ("2023-06 vs 2024-06", "2023-06", "2024-06"),
    ("2023-10 vs 2024-10", "2023-10", "2024-10"),
    ("2024-04 vs 2025-04", "2024-04", "2025-04"),
    ("2024-08 vs 2025-08", "2024-08", "2025-08"),
    ("2024-10 vs 2025-10", "2024-10", "2025-10"),
]


def _safe_filename(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def _load_precomputed(version: str, pair_name: str) -> dict | None:
    path = PRECOMPUTE_BASE / version / f"{_safe_filename(pair_name)}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "change_scores": data["change_scores"],
        "computed_patches": int(data["computed_patches"]),
        "mean_score": float(data["mean_score"]),
        "max_score": float(data["max_score"]),
        "pair_name": str(data["pair_name"]),
    }


def _apply_threshold(change_scores: np.ndarray, threshold_ratio: float, max_score: float) -> Image.Image:
    vmax = max(float(max_score), 0.001)
    threshold = threshold_ratio * vmax
    masked = change_scores.copy()
    masked[masked < threshold] = 0.0
    rgb = change_heatmap_fig(masked, vmin=0.0, vmax=vmax)
    mask = change_scores < threshold
    rgb_arr = np.array(rgb)
    rgb_arr[mask] = [20, 20, 20]
    return Image.fromarray(rgb_arr)


def _on_pair_change(version: str, pair_name: str, threshold_ratio: float):
    data = _load_precomputed(version, pair_name)
    if data is None:
        return None, f"⚠️ {version} / {pair_name} 尚未预计算。请选择其他组合或联系管理员预计算。"
    img = _apply_threshold(data["change_scores"], threshold_ratio, data["max_score"])
    stats = (
        f"### 全域变化检测统计\n\n"
        f"| 指标 | 值 |\n|------|-----|\n"
        f"| Model | {version} |\n"
        f"| Pair | {pair_name} |\n"
        f"| 计算 Patch 数 | {data['computed_patches']} |\n"
        f"| 平均变化强度 | {data['mean_score']:.4f} |\n"
        f"| 最大变化强度 | {data['max_score']:.4f} |\n"
        f"| 当前阈值 | {threshold_ratio * data['max_score']:.4f} ({threshold_ratio*100:.0f}%) |\n\n"
        f"拖动下方阈值滑块，实时过滤低强度变化区域。"
    )
    return img, stats


def _on_threshold_change(version: str, pair_name: str, threshold_ratio: float):
    data = _load_precomputed(version, pair_name)
    if data is None:
        return None
    return _apply_threshold(data["change_scores"], threshold_ratio, data["max_score"])


def _get_available_time_windows(patch_id: str) -> list[str]:
    """返回该 patch 至少有一个影像（s2/s2_hr/landsat）的时间窗口列表."""
    available = []
    sources = ["s2", "s2_hr", "landsat"]
    for key, (start_ms, end_ms) in TIME_WINDOWS.items():
        for src in sources:
            source_dir = RAW_DIR / src / patch_id
            if _find_best_tif(source_dir, start_ms, end_ms) is not None:
                available.append(key)
                break
    return available


def _on_patch_change(patch_id: str):
    """当 Patch ID 变化时，更新 Before/After 下拉框为可用时间窗口."""
    if not patch_id or patch_id not in cache.patch_ids:
        choices = list(TIME_WINDOWS.keys())
        return (
            gr.update(choices=choices, value="2024-10"),
            gr.update(choices=choices, value="2025-10"),
        )
    available = _get_available_time_windows(patch_id)
    if not available:
        choices = list(TIME_WINDOWS.keys())
        return (
            gr.update(choices=choices, value="2024-10"),
            gr.update(choices=choices, value="2025-10"),
        )
    before_default = "2024-10" if "2024-10" in available else available[0]
    after_default = "2025-10" if "2025-10" in available else available[-1]
    return (
        gr.update(choices=available, value=before_default),
        gr.update(choices=available, value=after_default),
    )


def _load_before_after_rgb(patch_id: str, before_window: tuple[float, float], after_window: tuple[float, float]):
    before_img = None
    after_img = None
    for source in ("s2", "s2_hr", "landsat"):
        if before_img is None:
            before_img = load_patch_source_rgb(patch_id, source, before_window)
        if after_img is None:
            after_img = load_patch_source_rgb(patch_id, source, after_window)
        if before_img is not None and after_img is not None:
            break
    return before_img, after_img


def _compute_pixel_change_score(
    patch_id: str,
    before_window: tuple[float, float],
    after_window: tuple[float, float],
    before_img: np.ndarray | None,
    after_img: np.ndarray | None,
) -> tuple[np.ndarray | None, str]:
    """基于像素级差异计算变化强度图（embedding 不可用时的回退方案）.

    优先级：NDVI 差异 > RGB LAB-like 差异 > 简单灰度差异
    """
    # 1) 尝试 NDVI 差异（需要 S2 原始 4 波段以上数据）
    for source in ("s2", "s2_hr"):
        before_raw = load_patch_source_raw(patch_id, source, before_window)
        after_raw = load_patch_source_raw(patch_id, source, after_window)
        if before_raw is not None and after_raw is not None and before_raw.shape[0] >= 4 and after_raw.shape[0] >= 4:
            ndvi_before = compute_ndvi_from_s2(before_raw)
            ndvi_after = compute_ndvi_from_s2(after_raw)
            # Resize to same shape if needed
            if ndvi_before.shape != ndvi_after.shape:
                from skimage.transform import resize
                ndvi_after = resize(ndvi_after, ndvi_before.shape, preserve_range=True, anti_aliasing=True)
            score = ndvi_delta_map(ndvi_before, ndvi_after)
            # Normalize
            smax = float(np.percentile(score, 99))
            if smax > 0:
                score = np.clip(score / smax, 0, 1)
            return score, "NDVI Delta (像素级回退)"

    # 2) RGB 像素 L2 距离回退
    if before_img is not None and after_img is not None:
        # Resize to same shape
        if before_img.shape != after_img.shape:
            try:
                from PIL import Image
                after_pil = Image.fromarray(after_img)
                after_pil = after_pil.resize((before_img.shape[1], before_img.shape[0]), Image.LANCZOS)
                after_img = np.array(after_pil)
            except Exception:
                pass
        if before_img.shape == after_img.shape:
            diff = np.abs(after_img.astype(np.float32) - before_img.astype(np.float32))
            # Per-pixel L2 over RGB
            score = np.linalg.norm(diff, axis=-1)
            smax = float(np.percentile(score, 99))
            if smax > 0:
                score = np.clip(score / smax, 0, 1)
            return score, "RGB Pixel Diff (像素级回退)"

    return None, ""


def _compute_patch_change(version: str, patch_id: str, before_key: str, after_key: str, threshold_ratio: float):
    if not patch_id or patch_id not in cache.patch_ids:
        return [None] * 6 + ["❌ 无效的 Patch ID"]
    before_window = TIME_WINDOWS.get(before_key)
    after_window = TIME_WINDOWS.get(after_key)
    if before_window is None or after_window is None:
        return [None] * 6 + ["❌ 无效的时间窗口"]

    before_img, after_img = _load_before_after_rgb(patch_id, before_window, after_window)

    engine = ChangeDetectionEngine(version)
    emb_before = engine.get_embedding(patch_id, before_window[0], before_window[1], use_precomputed=True)
    emb_after = engine.get_embedding(patch_id, after_window[0], after_window[1], use_precomputed=True)

    score = None
    note = ""
    use_embedding = False

    if emb_before is not None and emb_after is not None:
        # 检查是否是预计算模式（同一个 embedding）
        is_precomputed_mode = np.array_equal(emb_before, emb_after)
        if not is_precomputed_mode:
            # 实时推理模式：计算 cosine distance
            D, H, W = emb_before.shape
            fb = emb_before.reshape(D, -1)
            fa = emb_after.reshape(D, -1)
            nb = np.linalg.norm(fb, axis=0, keepdims=True)
            na = np.linalg.norm(fa, axis=0, keepdims=True)
            fb = fb / np.maximum(nb, 1e-8)
            fa = fa / np.maximum(na, 1e-8)
            cos_sim = np.sum(fb * fa, axis=0)
            score = ((1.0 - cos_sim) / 2.0).reshape(H, W)
            # 如果 embedding 变化信号极弱（max < 0.05），则回退到像素级差异
            if score.max() >= 0.05:
                use_embedding = True
                note = "基于 embedding cosine distance"
            else:
                score = None

    if score is None:
        # 回退到像素级差异
        score, pixel_note = _compute_pixel_change_score(patch_id, before_window, after_window, before_img, after_img)
        if score is None:
            return [before_img, after_img, None, None, None, None, f"❌ 无法计算 {patch_id} 的变化分数（无有效影像或 embedding）"]
        note = pixel_note

    # vmax 用 p95 增强对比度
    vmax = max(float(np.percentile(score, 95)), 0.001)

    # 热力图（matplotlib figure + colorbar）
    pil_heatmap = change_heatmap_fig(score, title="变化热力图", vmin=0.0, vmax=vmax)

    # 二值化图（matplotlib figure + 统计标题）
    threshold = threshold_ratio * vmax
    binary = binary_change_map(score, threshold)
    fig_bin, ax_bin = plt.subplots(figsize=(5.5, 4.8))
    ax_bin.imshow(binary)
    n_change = int(binary[:, :, 0].sum() // 255)
    ax_bin.set_title(f"二值化变化区域 (阈值={threshold:.3f})", fontsize=12, fontweight="bold")
    ax_bin.axis("off")
    fig_bin.tight_layout()
    pil_binary = fig_to_pil(fig_bin)

    # 叠加视图
    overlay_base = after_img if after_img is not None else before_img
    pil_overlay = None
    if overlay_base is not None:
        pil_overlay = overlay_rgb_heatmap(
            overlay_base, score, alpha=0.5, vmin=0.0, vmax=vmax, title="变化叠加视图"
        )

    # NDVI delta（如有 S2 原始数据）
    pil_ndvi = None
    for source in ("s2", "s2_hr"):
        before_raw = load_patch_source_raw(patch_id, source, before_window)
        after_raw = load_patch_source_raw(patch_id, source, after_window)
        if before_raw is not None and after_raw is not None and before_raw.shape[0] >= 4 and after_raw.shape[0] >= 4:
            ndvi_before = compute_ndvi_from_s2(before_raw)
            ndvi_after = compute_ndvi_from_s2(after_raw)
            delta = ndvi_delta_map(ndvi_before, ndvi_after)
            vmax_ndvi = max(abs(float(np.percentile(delta, 99))), 0.01)
            fig_ndvi, ax_ndvi = plt.subplots(figsize=(5.5, 4.8))
            im = ax_ndvi.imshow(delta, cmap="RdYlGn_r", vmin=-vmax_ndvi, vmax=vmax_ndvi)
            ax_ndvi.set_title("NDVI 差异", fontsize=12, fontweight="bold")
            ax_ndvi.axis("off")
            fig_ndvi.colorbar(im, ax=ax_ndvi, fraction=0.046, pad=0.04)
            fig_ndvi.tight_layout()
            pil_ndvi = fig_to_pil(fig_ndvi)
            break

    stats = (
        f"### Patch 变化检测详情\n\n"
        f"| 指标 | 值 |\n|------|-----|\n"
        f"| Patch ID | {patch_id} |\n"
        f"| Model | {version} |\n"
        f"| Before | {before_key} |\n"
        f"| After | {after_key} |\n"
        f"| 计算方法 | {note} |\n"
        f"| 平均变化强度 | {score.mean():.4f} |\n"
        f"| 最大变化强度 | {score.max():.4f} |\n"
        f"| 变化像素数 (>阈值) | {(score >= threshold).sum()} |\n"
        f"| 当前阈值 | {threshold:.4f} ({threshold_ratio*100:.0f}%) |\n\n"
        f"🔴 暖色 = 高变化 | ⚫ 冷色 = 低变化"
    )
    return [before_img, after_img, pil_ndvi, pil_heatmap, pil_binary, pil_overlay, stats]


def build_change_detection_tab():
    """构建 Tab 3: 变化检测。"""
    gr.Markdown(
        "## Change Detection\n"
        "选择 Patch（建议在 **Data & Embedding Field** 页面地图中点击选择后，将 Patch ID 复制到下方）、"
        "变化前后时间窗口，实时生成变化热力图与二值化结果。"
    )

    pair_choices = [p[0] for p in COMMON_PAIRS]
    time_keys = list(TIME_WINDOWS.keys())

    with gr.Tabs():
        with gr.Tab("🔎 Patch Change Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Interactive Map  \n*Click a patch rectangle to auto-fill Patch ID*")
                    folium_html = gr.HTML(value=render_folium_map(cache.patch_metas))

                    detail_version = gr.Dropdown(
                        choices=["v1", "v2", "v3"],
                        value="v2",
                        label="模型版本",
                    )
                    detail_patch = gr.Textbox(
                        label="Patch ID",
                        placeholder="Click a patch on the map above...",
                        value="",
                    )
                    detail_before = gr.Dropdown(
                        choices=time_keys,
                        value="2024-10",
                        label="Before 时间窗口",
                    )
                    detail_after = gr.Dropdown(
                        choices=time_keys,
                        value="2025-10",
                        label="After 时间窗口",
                    )
                    threshold_slider = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.01,
                        label="变化强度阈值 (相对最大值的比率)",
                    )
                    btn_compute_detail = gr.Button("🚀 生成变化检测", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        img_before = gr.Image(label="Before 影像", height=320)
                        img_after = gr.Image(label="After 影像", height=320)
                    with gr.Row():
                        img_ndvi = gr.Image(label="NDVI 差异", height=320)
                    with gr.Row():
                        img_heatmap = gr.Image(label="变化热力图", height=320)
                        img_binary = gr.Image(label="二值化变化区域", height=320)
                        img_overlay = gr.Image(label="变化叠加视图", height=320)
                    detail_stats = gr.Markdown()

            btn_compute_detail.click(
                fn=_compute_patch_change,
                inputs=[detail_version, detail_patch, detail_before, detail_after, threshold_slider],
                outputs=[img_before, img_after, img_ndvi, img_heatmap, img_binary, img_overlay, detail_stats],
            )

            # Timer: auto-sync Folium click to Patch ID (client-side only)
            _cd_timer = gr.Timer(0.5, active=True)
            _cd_timer.tick(
                fn=None,
                inputs=[detail_patch],
                outputs=[detail_patch],
                js="(x) => { var p = window._aef_pending_patch; if (p) { window._aef_pending_patch = null; return p; } return x; }",
            )

            detail_patch.change(
                fn=_on_patch_change,
                inputs=[detail_patch],
                outputs=[detail_before, detail_after],
            )

        with gr.Tab("🗺️ Global Precomputed Map"):
            with gr.Row():
                with gr.Column(scale=1):
                    version_select = gr.Dropdown(
                        choices=["v1", "v2", "v3"],
                        value="v2",
                        label="模型版本",
                    )
                    pair_select = gr.Dropdown(
                        choices=pair_choices,
                        value=pair_choices[3] if len(pair_choices) > 3 else pair_choices[0],
                        label="时间组合",
                    )
                    global_threshold_slider = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.01,
                        label="Change Intensity Threshold (ratio of max score)",
                    )
                    btn_load = gr.Button("Load Precomputed Map", variant="primary")

                with gr.Column(scale=3):
                    img_global = gr.Image(label="Global Change Intensity Map", height=650)
                    global_stats = gr.Markdown()

            btn_load.click(
                fn=_on_pair_change,
                inputs=[version_select, pair_select, global_threshold_slider],
                outputs=[img_global, global_stats],
            )
            global_threshold_slider.change(
                fn=_on_threshold_change,
                inputs=[version_select, pair_select, global_threshold_slider],
                outputs=[img_global],
            )
