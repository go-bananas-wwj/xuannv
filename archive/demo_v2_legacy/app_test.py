#!/usr/bin/env python3
import sys
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
from demo_v2.components.mlp_cd_tab import build_mlp_cd_tab
from demo_v2.components.three_type_cd_tab import build_three_type_cd_tab
from demo_v2.components.downstream_tab import build_downstream_tab
from demo_v2.components.performance_tab import build_performance_tab
from demo_v2.components.model_comparison_tab import build_model_comparison_tab

_CSS = """
.gradio-container {
    max-width: 1800px !important;
    margin: auto !important;
}
"""

_JS_FOLIUM_BRIDGE = """
() => {
    window._aef_pending_patch = null;
}
"""

def create_app() -> gr.Blocks:
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
| **Best Model** | V2 ... |
| **Embeddings** | {emb_info} |
"""
        )

        shared_patch_id = gr.State("")

        with gr.Tabs():
            with gr.Tab("📖 Project Intro"):
                build_project_intro_tab()
            with gr.Tab("🗺️ Data & Embedding Field"):
                folium_html, load_map_fn, patch_id_box = build_data_browser_tab(shared_patch_id)
            with gr.Tab("🔍 Spatial Anomaly"):
                build_spatial_anomaly_tab()
            with gr.Tab("🔥 Change Detection"):
                build_change_detection_tab(shared_patch_id)
            with gr.Tab("🏗️ MLP Change Detection"):
                build_mlp_cd_tab()
            with gr.Tab("🌆 Three-Type Change Detection"):
                build_three_type_cd_tab()
            with gr.Tab("🌊 Downstream Tasks"):
                build_downstream_tab()
            with gr.Tab("📈 Performance"):
                build_performance_tab()
            with gr.Tab("⚖️ Model Comparison"):
                build_model_comparison_tab()

        app.load(fn=load_map_fn, outputs=[folium_html])

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7991, css=_CSS, js=_JS_FOLIUM_BRIDGE)
