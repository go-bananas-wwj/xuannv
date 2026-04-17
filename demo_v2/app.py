#!/usr/bin/env python3
"""玄女底座 Demo V2 — Unified Visualization Platform.

Tabs:
1. Project Introduction
2. Data & Embedding Field
3. Change Detection
4. Three-Type Change Detection (建筑工地 / 房屋拆除 / 非农非粮)
5. Downstream Tasks
6. Model Performance Analysis
7. Model Comparison

Launch:
    python demo_v2/app.py [--port 7990] [--share]
"""
from __future__ import annotations

import sys

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import gradio as gr
import matplotlib
matplotlib.use("Agg")

from demo_v2.cache_manager import cache
from demo_v2.config import list_available_models

from demo_v2.components.project_intro import build_project_intro_tab
from demo_v2.components.data_browser import build_data_browser_tab
from demo_v2.components.spatial_anomaly_tab import build_spatial_anomaly_tab
from demo_v2.components.change_detection_tab import build_change_detection_tab
from demo_v2.components.three_type_cd_tab import build_three_type_cd_tab
from demo_v2.components.downstream_tab import build_downstream_tab
from demo_v2.components.performance_tab import build_performance_tab
from demo_v2.components.model_comparison_tab import build_model_comparison_tab
from demo_v2.components.alphaearth_tab import build_alphaearth_tab

_CSS = """
.gradio-container {
    max-width: 1800px !important;
    margin: auto !important;
}

/* Time x Source Matrix 横向滚动 */
#time_source_matrix_img {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: nowrap !important;
    width: 100% !important;
}
#time_source_matrix_img img,
#time_source_matrix_img .image-container,
#time_source_matrix_img .image-frame,
#time_source_matrix_img .image-preview {
    max-width: none !important;
    width: auto !important;
    max-height: 700px !important;
    display: inline-block !important;
}
"""

_JS_FOLIUM_BRIDGE = """
() => {
    window._aef_pending_patch = null;
}
"""


def create_app() -> gr.Blocks:
    """Create Gradio application."""
    print("[App] Loading cache...")
    cache.load()
    models = list_available_models()
    emb_info = ", ".join(
        f"{k}: {v['has_embeddings'] and 'Ready' or 'Missing'}"
        for k, v in models.items()
    )

    with gr.Blocks(title="玄女底座 Visualization Platform") as app:
        gr.Markdown(
            f"""# 🌍 玄女底座 Visualization Platform — Harbin {cache.num_patches} patches

| Item | Details |
|------|---------|
| **Best Model** | V2 (S2+S1+Landsat+S2-HR+S1-HR → 128-dim embedding, skip L2 + raw_uniformity) |
| **Training** | 500 epochs, temporal contrastive loss |
| **Input** | 5 temporal sources (S2, S1, Landsat, S2-HR, S1-HR) |
| **Downstream** | kNN-5 / Linear Probe / MLP 下游变化检测 |
| **Embeddings** | {emb_info} |
"""
        )

        with gr.Tabs():
            with gr.Tab("📖 Project Intro"):
                build_project_intro_tab()
            with gr.Tab("🗺️ Data & Embedding Field"):
                folium_html, load_map_fn = build_data_browser_tab()
            with gr.Tab("🔍 Spatial Anomaly"):
                build_spatial_anomaly_tab()
            with gr.Tab("🔥 Change Detection"):
                build_change_detection_tab()
            with gr.Tab("🌆 Three-Type Change Detection"):
                build_three_type_cd_tab()
            with gr.Tab("🌊 Downstream Tasks"):
                build_downstream_tab()
            with gr.Tab("📈 Performance"):
                build_performance_tab()
            with gr.Tab("⚖️ Model Comparison"):
                build_model_comparison_tab()
            with gr.Tab("🌍 AlphaEarth Official"):
                build_alphaearth_tab()

        gr.Markdown(
            "<div style='text-align:center; color:#888; margin-top:20px;'>"
            "玄女底座 Visualization Demo V2 | Powered by Gradio</div>"
        )

        # 启动后加载 Folium 地图
        app.load(fn=load_map_fn, outputs=[folium_html])
        # 初始化 Folium JS bridge 全局变量
        app.load(fn=None, inputs=None, outputs=None, js=_JS_FOLIUM_BRIDGE)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="玄女底座 Visualization Platform")
    parser.add_argument("--port", type=int, default=7990)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
