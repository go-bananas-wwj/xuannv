"""Tab: MLP Change Detection 结果浏览."""
from __future__ import annotations

import gradio as gr

from demo_v2.engines.mlp_engine import MLPEngine

_engine = MLPEngine()

_CATEGORIES = {
    "june": "2025-06 月度综合",
    "aug": "2025-08 月度综合",
    "September": "2025-09 月度综合",
    "October": "2025-10 月度综合",
    "SAR建筑工地": "建筑工地专题",
    "SAR房屋拆除": "房屋拆除专题",
    "SAR疑似违建": "疑似违建专题",
    "SAR非农非粮": "非农非粮专题",
}

_CATEGORY_METRIC_KEY = {
    "SAR建筑工地": "construction",
    "SAR房屋拆除": "demolition",
    "SAR非农非粮": "farmland",
    "SAR疑似违建": "construction",
}


def _display_name(cat: str) -> str:
    return _CATEGORIES.get(cat, cat)


def _inv_display_name(display: str) -> str:
    for k, v in _CATEGORIES.items():
        if v == display:
            return k
    return display


def _format_metrics(data: dict) -> str:
    if not data:
        return "*暂无该类别的 MLP 指标数据。*"
    lines = [
        "### MLP 下游指标",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
    ]
    for key in ["category", "n_patches", "mean_auc", "std_auc", "mean_f1", "std_f1", "mean_ap", "std_ap"]:
        if key in data:
            val = data[key]
            if isinstance(val, float):
                val = f"{val:.4f}"
            lines.append(f"| {key} | {val} |")
    if "fold_results" in data and isinstance(data["fold_results"], list):
        lines.append(f"| folds | {len(data['fold_results'])} |")
    lines.append("")
    return "\n".join(lines)


def _get_total_pages(version: str, folder: str) -> int:
    return len(_engine.list_pages(version, folder, ""))


def _load_page(version: str, folder_display: str, page_str: str):
    folder = _inv_display_name(folder_display)
    total = _get_total_pages(version, folder)
    if total == 0:
        return None, "未找到结果图片。", ""
    try:
        page_index = int(page_str)
    except Exception:
        page_index = 1
    page_index = max(1, min(page_index, total))
    img = _engine.load_page_image(version, folder, "", page_index)
    status = f"第 {page_index} / {total} 页"
    metric_key = _CATEGORY_METRIC_KEY.get(folder)
    metrics_md = ""
    if metric_key:
        m = _engine.get_metrics(version)
        cat_metrics = m.get(metric_key, {})
        metrics_md = _format_metrics(cat_metrics)
    else:
        metrics_md = "*该目录暂无汇总指标。*"
    return img, status, metrics_md


def build_mlp_cd_tab():
    """构建 MLP Change Detection Tab — 纯按钮触发，无自动级联事件."""
    gr.Markdown(
        "## MLP Change Detection\n"
        "浏览 MLP 下游变化检测结果（概率图 / SHP 叠加图 / 专题图）。"
    )

    versions = _engine.list_available_versions()
    default_version = versions[0] if versions else None
    default_cats = _engine.list_categories(default_version) if default_version else []
    default_folder = _display_name(default_cats[0]) if default_cats else None

    with gr.Row():
        with gr.Column(scale=1):
            version_select = gr.Dropdown(
                choices=versions,
                value=default_version,
                label="模型版本",
            )
            folder_select = gr.Dropdown(
                choices=[_display_name(c) for c in default_cats],
                value=default_folder,
                label="输出类别 / 月度",
            )
            page_input = gr.Number(
                value=1,
                label="页码",
                minimum=1,
                step=1,
                precision=0,
            )
            btn_load = gr.Button("🖼️ 加载结果", variant="primary")
            page_status = gr.Textbox(label="页码状态", interactive=False)
            metrics_md = gr.Markdown()

        with gr.Column(scale=3):
            img_display = gr.Image(label="MLP 结果图", height=750)

    version_select.change(
        fn=lambda v: gr.update(choices=[_display_name(c) for c in _engine.list_categories(v)]),
        inputs=[version_select],
        outputs=[folder_select],
    )

    btn_load.click(
        fn=_load_page,
        inputs=[version_select, folder_select, page_input],
        outputs=[img_display, page_status, metrics_md],
    )
