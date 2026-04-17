"""Tab: 哈尔滨新区 Few-Shot 变化检测."""
from __future__ import annotations

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from demo_v2.engines.fewshot_engine import FewShotEngine
from demo_v2.utils.visualization import fig_to_pil

_ENGINE = FewShotEngine(version="v2")
_ANNOTATED_PATCHES = _ENGINE.get_annotated_patches()


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy [H, W, 3] uint8 -> PIL."""
    if arr is None:
        return None
    return Image.fromarray(arr)


def _render_heatmap_with_colorbar(prob_map: np.ndarray, title: str = "Predicted Heatmap") -> Image.Image:
    """渲染带 colorbar 的热力图，参考用户提供的样式."""
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=120)
    vmax = max(float(prob_map.max()), 0.001)
    im = ax.imshow(prob_map, cmap="hot", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    pil_img = fig_to_pil(fig)
    plt.close(fig)
    return pil_img


def _render_gt_mask(gt_mask: np.ndarray, target_size: tuple[int, int] = (128, 128)) -> Image.Image:
    """GT mask: 变化区域红色，背景黑色，resize 到目标尺寸."""
    rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    rgb[gt_mask == 1] = [255, 0, 0]
    rgb[gt_mask == 0] = [20, 20, 20]
    pil_img = Image.fromarray(rgb)
    pil_img = pil_img.resize(target_size, Image.NEAREST)
    return pil_img


def _run_single_detection(patch_id: str, shot_count: int):
    """单 patch 实时检测."""
    if patch_id not in _ANNOTATED_PATCHES:
        return [None] * 4 + ["❌ 该 patch 无变化标注"]

    res = _ENGINE.detect_single_patch(patch_id, shot_count)
    if "error" in res:
        return [None] * 4 + [f"❌ {res['error']}"]

    before_img = _np_to_pil(res.get("before_rgb"))
    after_img = _np_to_pil(res.get("after_rgb"))
    gt_img = _render_gt_mask(res["gt_mask"], target_size=(128, 128))
    heatmap_img = _render_heatmap_with_colorbar(res["prob_map"], title=f"Predicted Heatmap (shot={shot_count})")

    metrics = res.get("metrics")
    period = res.get("period", "Unknown")
    stats = (
        f"### Patch {patch_id} | {period} | Shot={shot_count}\n\n"
        f"| 指标 | 值 |\n|------|-----|\n"
    )
    if metrics:
        stats += (
            f"| AUC | {metrics['auc']:.4f} |\n"
            f"| BA  | {metrics['ba']:.4f} |\n"
            f"| F1  | {metrics['f1']:.4f} |\n"
        )
    else:
        stats += "| AUC | N/A |\n| BA | N/A |\n| F1 | N/A |\n"

    stats += f"\n变化像素数: {int(res['gt_mask'].sum())} / {res['gt_mask'].size}"
    return [before_img, after_img, gt_img, heatmap_img, stats]


def _run_shot_comparison(patch_id: str):
    """生成 1/10/50/100/500-shot 对比图，顶部加 Before/After RGB 参考."""
    if patch_id not in _ANNOTATED_PATCHES:
        return None, "❌ 该 patch 无变化标注"

    benchmark = _ENGINE.evaluate_benchmark(patch_id)
    if "error" in benchmark:
        return None, f"❌ {benchmark['error']}"

    prob_maps = benchmark.get("prob_maps", {})
    gt_mask = benchmark.get("gt_mask")
    before_rgb = benchmark.get("before_rgb")
    after_rgb = benchmark.get("after_rgb")
    results = benchmark.get("results", {})

    shots = [1, 10, 50, 100, 500]
    valid_shots = [s for s in shots if s in prob_maps]
    if not valid_shots:
        return None, "❌ 所有 shot 评估均失败"

    # 布局: 第一行 [Before] [After] [GT]
    #       第二行 [1-shot] [10-shot] [50-shot] [100-shot] [500-shot]
    n_shots = len(valid_shots)
    n_cols = max(n_shots, 3)
    fig = plt.figure(figsize=(n_cols * 2.8, 6.5), dpi=120)
    gs = fig.add_gridspec(2, n_cols, hspace=0.35, wspace=0.25)

    def _add_image(row, col, img_arr, title):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    # 第一行: Before, After, GT
    if before_rgb is not None:
        _add_image(0, 0, before_rgb, "Before (S2 RGB)")
    if after_rgb is not None:
        _add_image(0, 1, after_rgb, "After (S2 RGB)")
    if gt_mask is not None:
        gt_vis = _render_gt_mask(gt_mask)
        _add_image(0, 2, np.array(gt_vis), "GT Mask")
    # 隐藏第一行剩余空位
    for c in range(3, n_cols):
        ax = fig.add_subplot(gs[0, c])
        ax.axis("off")

    # 第二行: shot 对比
    for i, shot in enumerate(valid_shots):
        ax = fig.add_subplot(gs[1, i])
        prob = prob_maps[shot]
        vmax = max(float(prob.max()), 0.001)
        im = ax.imshow(prob, cmap="hot", vmin=0.0, vmax=vmax)
        ax.set_title(f"{shot}-shot", fontsize=10, fontweight="bold")
        ax.axis("off")
        # 右侧加小 colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        metrics = results[shot].get("metrics")
        if metrics:
            ax.text(0.5, -0.08, f"AUC={metrics['auc']:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=9, color="black")

    fig.suptitle(f"Few-Shot Change Detection — {patch_id}", fontsize=14, fontweight="bold", y=0.98)
    comp_img = fig_to_pil(fig)
    plt.close(fig)

    stats = f"### {patch_id} Few-Shot 能力对比\n\n"
    for shot in valid_shots:
        metrics = results[shot].get("metrics")
        if metrics:
            stats += f"- **{shot}-shot**: AUC={metrics['auc']:.3f}, BA={metrics['ba']:.3f}, F1={metrics['f1']:.3f}\n"
        else:
            stats += f"- **{shot}-shot**: metrics unavailable\n"

    return comp_img, stats


def build_fewshot_harbin_tab() -> None:
    """构建哈尔滨新区 Few-Shot 变化检测 Tab."""
    gr.Markdown(
        "## 🌆 Harbin Few-Shot Change Detection\n"
        "Frozen V2 Backbone + kNN/Linear head. Using real 2025 change annotations.\n\n"
        "Layout: Before (S2 RGB) | After (S2 RGB) | GT Mask (Red=Change) | Predicted Heatmap"
    )

    with gr.Tabs():
        with gr.Tab("单 Patch 实时检测"):
            with gr.Row():
                with gr.Column(scale=1):
                    default_patch = "patch_000229" if "patch_000229" in _ANNOTATED_PATCHES else (_ANNOTATED_PATCHES[0] if _ANNOTATED_PATCHES else None)
                    patch_select = gr.Dropdown(
                        choices=_ANNOTATED_PATCHES,
                        value=default_patch,
                        label="Patch ID",
                    )
                    shot_slider = gr.Slider(
                        minimum=1, maximum=1000, value=500, step=1,
                        label="Shot Count (per class)",
                    )
                    btn_detect = gr.Button("🚀 运行检测", variant="primary")

                with gr.Column(scale=3):
                    with gr.Row():
                        before_img = gr.Image(label="Before (S2 RGB)", height=280)
                        after_img = gr.Image(label="After (S2 RGB)", height=280)
                        gt_img = gr.Image(label="GT Mask", height=280)
                        heatmap_img = gr.Image(label="Predicted Heatmap", height=280)
                    detect_stats = gr.Markdown()

            btn_detect.click(
                fn=_run_single_detection,
                inputs=[patch_select, shot_slider],
                outputs=[before_img, after_img, gt_img, heatmap_img, detect_stats],
            )

        with gr.Tab("Few-Shot 能力对比"):
            with gr.Row():
                with gr.Column(scale=1):
                    comp_patch = gr.Dropdown(
                        choices=_ANNOTATED_PATCHES,
                        value=default_patch,
                        label="Patch ID",
                    )
                    btn_compare = gr.Button("🚀 生成对比图", variant="primary")

                with gr.Column(scale=3):
                    comp_img = gr.Image(label="1 / 10 / 50 / 100 / 500-shot 对比", height=500)
                    comp_stats = gr.Markdown()

            btn_compare.click(
                fn=_run_shot_comparison,
                inputs=[comp_patch],
                outputs=[comp_img, comp_stats],
            )
