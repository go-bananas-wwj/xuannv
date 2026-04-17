#!/usr/bin/env python3
"""TaskHead 效果展示 Demo — 运行在 7991，专注 Raw vs Head 对比."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "/workspace/xuannv")

from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.engines.fewshot_engine import FewShotEngine
from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
    PERIODS,
)
from demo_v2.utils.visualization import fig_to_pil

# ── 常量 ──
BENCH_DIR = Path("/workspace/outputs/aef_qwen_v2_taskheads/benchmark_all")
VISUALS_DIR = BENCH_DIR / "visuals"
SUMMARY_PATH = BENCH_DIR / "summary.json"

with open(SUMMARY_PATH) as f:
    BENCH_DATA = json.load(f)

RECORDS = BENCH_DATA["records"]
PATCH_IDS = [r["patch_id"] for r in RECORDS]

# 预计算排序
RECORDS_BY_IMPROVEMENT = sorted(
    RECORDS, key=lambda x: x["head"]["auc"] - x["raw"]["auc"], reverse=True
)
RECORDS_BY_HEAD_AUC = sorted(RECORDS, key=lambda x: x["head"]["auc"], reverse=True)

_CD_ENGINE = ChangeDetectionEngine("v2", "cuda:0")
_FS_ENGINE = FewShotEngine("v2", "cuda:0")


# ── 辅助函数 ──
def _load_s2_rgb(pid: str, period: str):
    if period not in PERIODS:
        return None, None
    bs, be = PERIODS[period]
    mid = (bs + be) / 2.0
    before = _FS_ENGINE._load_s2_rgb_for_period(pid, bs, mid)
    after = _FS_ENGINE._load_s2_rgb_for_period(pid, mid, be)
    return before, after


def _np_to_pil(arr):
    if arr is None:
        return None
    return Image.fromarray(arr)


def _render_heatmap(score: np.ndarray, title: str = "") -> Image.Image:
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    vmax = max(float(score.max()), 0.001)
    im = ax.imshow(score, cmap="hot", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    pil_img = fig_to_pil(fig)
    plt.close(fig)
    return pil_img


def _render_gt_mask(gt_mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    rgb[gt_mask == 1] = [255, 0, 0]
    rgb[gt_mask == 0] = [20, 20, 20]
    return Image.fromarray(rgb)


# ── Tab 1: 全局指标 ──
def build_global_metrics_tab():
    raw = BENCH_DATA["raw"]
    head = BENCH_DATA["head"]

    # 1. 数字卡片
    gr.Markdown(
        f"""
        ## 📊 TaskHead 全局提升指标

        | 指标 | Raw Cosine | TaskHead CD | 提升 |
        |------|-----------|-------------|------|
        | AUC Mean | **{raw['auc_mean']:.4f}** | **{head['auc_mean']:.4f}** | +{head['auc_mean'] - raw['auc_mean']:.4f} |
        | AUC Median | {raw['auc_median']:.4f} | {head['auc_median']:.4f} | +{head['auc_median'] - raw['auc_median']:.4f} |
        | 改善 Patch 数 | — | **{BENCH_DATA['improved_patches']} / {BENCH_DATA['n_patches']}** | — |
        """
    )

    # 2. 箱线图 + 散点图
    with gr.Row():
        with gr.Column():
            box_plot = gr.Image(label="AUC 分布对比 (Boxplot)", height=320)
        with gr.Column():
            scatter_plot = gr.Image(label="Raw AUC vs Head AUC 散点图", height=320)

    def _gen_plots():
        raw_aucs = [r["raw"]["auc"] for r in RECORDS]
        head_aucs = [r["head"]["auc"] for r in RECORDS]

        # Boxplot
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
        bp = ax.boxplot([raw_aucs, head_aucs], labels=["Raw Cosine", "TaskHead CD"], patch_artist=True)
        colors = ["#ffcccc", "#ccffcc"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title("AUC Distribution Comparison", fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()
        box_img = fig_to_pil(fig)
        plt.close(fig)

        # Scatter
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
        ax.scatter(raw_aucs, head_aucs, alpha=0.6, edgecolors="black", s=60)
        ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y=x (no change)")
        ax.set_xlabel("Raw Cosine AUC", fontsize=12)
        ax.set_ylabel("TaskHead CD AUC", fontsize=12)
        ax.set_title("Per-Patch AUC Improvement", fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        fig.tight_layout()
        scatter_img = fig_to_pil(fig)
        plt.close(fig)

        return box_img, scatter_img

    # 页面加载时自动生成
    box_plot.value, scatter_plot.value = _gen_plots()


# ── Tab 2: 单 Patch 对比 ──
def build_single_patch_tab():
    gr.Markdown("## 🔍 单 Patch 变化检测对比")

    with gr.Row():
        patch_select = gr.Dropdown(choices=PATCH_IDS, value=PATCH_IDS[0], label="选择 Patch")
        btn_run = gr.Button("🚀 生成对比", variant="primary")

    with gr.Row():
        before_img = gr.Image(label="Before S2 RGB", height=280)
        after_img = gr.Image(label="After S2 RGB", height=280)
        gt_img = gr.Image(label="GT Mask", height=280)

    with gr.Row():
        raw_img = gr.Image(label="Raw Cosine Distance", height=280)
        head_img = gr.Image(label="TaskHead CD", height=280)

    stats_md = gr.Markdown()

    def _run(patch_id: str):
        rec = next((r for r in RECORDS if r["patch_id"] == patch_id), None)
        if rec is None:
            return [None] * 5 + ["❌ Patch not found"]

        period = rec["period"]
        before_rgb, after_rgb = _load_s2_rgb(patch_id, period)
        gt_mask, _ = rasterize_patch_changes(patch_id, grid_size=64)

        # 获取 score maps
        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0
        score_raw = _CD_ENGINE.compute_change_score(patch_id, (bs, mid), (mid, be), use_precomputed=True, use_task_head=False)
        score_head = _CD_ENGINE.compute_change_score(patch_id, (bs, mid), (mid, be), use_precomputed=True, use_task_head=True)

        out = [
            _np_to_pil(before_rgb),
            _np_to_pil(after_rgb),
            _render_gt_mask(gt_mask),
            _render_heatmap(score_raw, "Raw Cosine") if score_raw is not None else None,
            _render_heatmap(score_head, "TaskHead CD") if score_head is not None else None,
        ]

        md = (
            f"### {patch_id} | {period}\n\n"
            f"| 方法 | AUC | BA | F1 |\n"
            f"|------|-----|-----|-----|\n"
            f"| Raw Cosine | {rec['raw']['auc']:.4f} | {rec['raw']['ba']:.4f} | {rec['raw']['f1']:.4f} |\n"
            f"| TaskHead CD | **{rec['head']['auc']:.4f}** | {rec['head']['ba']:.4f} | {rec['head']['f1']:.4f} |\n"
            f"| 提升 | +{rec['head']['auc'] - rec['raw']['auc']:.4f} | — | — |"
        )
        out.append(md)
        return out

    btn_run.click(
        fn=_run,
        inputs=[patch_select],
        outputs=[before_img, after_img, gt_img, raw_img, head_img, stats_md],
    )
    # 自动加载默认值
    before_img.value, after_img.value, gt_img.value, raw_img.value, head_img.value, stats_md.value = _run(PATCH_IDS[0])


# ── Tab 3: Top 提升案例 ──
def build_top_improvements_tab():
    gr.Markdown("## 🏆 Top 10 提升最大案例")
    gr.Markdown("直接从全量 benchmark 可视化结果中读取。")

    top_imgs = []
    top_labels = []
    for r in RECORDS_BY_IMPROVEMENT[:10]:
        pid = r["patch_id"]
        diff = r["head"]["auc"] - r["raw"]["auc"]
        path = VISUALS_DIR / f"{pid}_comparison.png"
        if path.exists():
            top_imgs.append(str(path))
            top_labels.append(f"{pid}  |  +{diff:.3f}")

    if top_imgs:
        gr.Gallery(
            value=list(zip(top_imgs, top_labels)),
            label="Top Improvements (Raw vs TaskHead vs GT)",
            columns=5,
            height="auto",
        )
    else:
        gr.Markdown("⚠️ 无可视化文件")


# ── Tab 4: Few-Shot 500-shot 对比 ──
def build_fewshot_compare_tab():
    gr.Markdown("## 🎯 Few-Shot 500-shot 对比")
    gr.Markdown(
        "左侧：原始 embedding + sklearn 500-shot（Legacy）；"
        "右侧：TaskHead 直接输出（500-shot 模式自动触发）"
    )

    with gr.Row():
        patch_select = gr.Dropdown(choices=PATCH_IDS, value=PATCH_IDS[0], label="选择 Patch")
        btn_run = gr.Button("🚀 运行对比", variant="primary")

    with gr.Row():
        raw_fs_img = gr.Image(label="Legacy 500-shot (sklearn)", height=320)
        head_fs_img = gr.Image(label="TaskHead 500-shot", height=320)
        gt_fs_img = gr.Image(label="GT Mask", height=320)

    stats_md = gr.Markdown()

    def _run_fs(patch_id: str):
        period = get_period_for_patch(patch_id)
        if period is None or period not in PERIODS:
            return [None] * 3 + ["❌ 无有效 period"]

        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0
        gt_mask, _ = rasterize_patch_changes(patch_id, grid_size=64)

        # TaskHead 500-shot（走 FewShotEngine，shot>=50 自动切 TaskHead）
        res_head = _FS_ENGINE.detect_single_patch(patch_id, 500)
        prob_head = res_head.get("prob_map")

        # Legacy：强制用原始 sklearn 方法
        # 临时 hack：直接调用内部 sklearn 逻辑（复制一段代码）
        emb_before = _CD_ENGINE.get_embedding(patch_id, bs, mid, use_precomputed=True)
        emb_after = _CD_ENGINE.get_embedding(patch_id, mid, be, use_precomputed=True)
        D, H, W = emb_before.shape
        features = np.zeros((H * W, D * 2), dtype=np.float32)
        for px in range(H):
            for py in range(W):
                idx = px * W + py
                features[idx, :D] = emb_before[:, px, py]
                features[idx, D:] = emb_after[:, px, py]
        labels = gt_mask.flatten()
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_neg_sample = min(len(pos_idx) * 10, len(neg_idx))
        neg_sample = np.random.choice(neg_idx, n_neg_sample, replace=False)
        sampled_idx = np.concatenate([pos_idx, neg_sample])
        np.random.shuffle(sampled_idx)
        X = features[sampled_idx]
        y = labels[sampled_idx]
        rng = np.random.RandomState(42)
        train_samples = []
        for c in np.unique(y):
            c_idx = np.where(y == c)[0]
            n_sample = min(500, len(c_idx))
            if n_sample > 0:
                train_samples.extend(rng.choice(c_idx, n_sample, replace=False).tolist())
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_samples])
        X_all_s = scaler.transform(features)
        clf.fit(X_train_s, y[train_samples])
        prob_raw = clf.predict_proba(X_all_s)[:, 1].reshape(H, W)

        out = [
            _render_heatmap(prob_raw, "Legacy 500-shot"),
            _render_heatmap(prob_head, "TaskHead 500-shot") if prob_head is not None else None,
            _render_gt_mask(gt_mask),
        ]

        # metrics
        def _calc_metrics(probs, labels_flat):
            try:
                auc = roc_auc_score(labels_flat, probs)
            except Exception:
                auc = 0.5
            preds = (probs > 0.5).astype(int)
            ba = balanced_accuracy_score(labels_flat, preds)
            f1 = f1_score(labels_flat, preds, zero_division=0)
            return auc, ba, f1

        auc_raw, ba_raw, f1_raw = _calc_metrics(prob_raw.flatten(), labels)
        auc_head, ba_head, f1_head = _calc_metrics(prob_head.flatten(), labels) if prob_head is not None else (0, 0, 0)

        md = (
            f"### {patch_id} | 500-shot 对比\n\n"
            f"| 方法 | AUC | BA | F1 |\n"
            f"|------|-----|-----|-----|\n"
            f"| Legacy (sklearn) | {auc_raw:.4f} | {ba_raw:.4f} | {f1_raw:.4f} |\n"
            f"| TaskHead | **{auc_head:.4f}** | {ba_head:.4f} | {f1_head:.4f} |\n"
        )
        out.append(md)
        return out

    btn_run.click(
        fn=_run_fs,
        inputs=[patch_select],
        outputs=[raw_fs_img, head_fs_img, gt_fs_img, stats_md],
    )


# ── 主入口 ──
_CSS = """
.gradio-container { max-width: 1400px !important; }
"""

with gr.Blocks(title="TaskHead Benchmark", css=_CSS) as app:
    gr.Markdown(
        "# 🚀 TaskHead vs Raw Cosine 效果对比\n"
        "基于 V2 backbone + 轻量 ChangeDetectionHead 的全量 70 patch benchmark 结果展示"
    )

    with gr.Tabs():
        with gr.Tab("📊 全局指标"):
            build_global_metrics_tab()
        with gr.Tab("🔍 单 Patch 对比"):
            build_single_patch_tab()
        with gr.Tab("🏆 Top 提升案例"):
            build_top_improvements_tab()
        with gr.Tab("🎯 Few-Shot 对比"):
            build_fewshot_compare_tab()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7991)
