"""Tab: 三类变化监测展示 (建筑工地 / 房屋拆除 / 非农非粮)."""
from __future__ import annotations

import gradio as gr

from demo_v2.engines.mlp_engine import MLPEngine

_engine = MLPEngine()

_CATEGORIES = {
    "建筑工地": "SAR建筑工地",
    "房屋拆除": "SAR房屋拆除",
    "非农非粮": "SAR非农非粮",
}

_CATEGORY_METRIC_KEY = {
    "SAR建筑工地": "construction",
    "SAR房屋拆除": "demolition",
    "SAR非农非粮": "farmland",
}


def _format_metrics(data: dict) -> str:
    if not data:
        return "*暂无该类别的 MLP 指标数据。*"
    lines = [
        "### 下游指标",
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


def _get_total_pages(folder: str) -> int:
    return len(_engine.list_pages("v2", folder, ""))


def _on_category_change(category: str):
    folder = _CATEGORIES.get(category, "")
    total = _get_total_pages(folder)
    m = _engine.get_metrics("v2")
    metric_key = _CATEGORY_METRIC_KEY.get(folder)
    metrics_md = _format_metrics(m.get(metric_key, {})) if metric_key else ""
    img, status = _load_page(folder, 1)
    return (
        gr.update(value=1, maximum=max(1, total)),
        gr.update(value=img),
        gr.update(value=status),
        gr.update(value=metrics_md),
    )


def _load_page(folder: str, page_num: float):
    total = _get_total_pages(folder)
    if total == 0:
        return None, "未找到结果图片。"
    page_index = max(1, min(int(page_num), total))
    img = _engine.load_page_image("v2", folder, "", page_index)
    status = f"第 {page_index} / {total} 页"
    return img, status


def _on_prev(category: str, page_num: float):
    folder = _CATEGORIES.get(category, "")
    new_page = max(1, int(page_num) - 1)
    img, status = _load_page(folder, new_page)
    total = _get_total_pages(folder)
    return (
        gr.update(value=img),
        gr.update(value=status),
        gr.update(value=new_page),
        gr.update(interactive=new_page > 1),
        gr.update(interactive=new_page < total),
    )


def _on_next(category: str, page_num: float):
    folder = _CATEGORIES.get(category, "")
    total = _get_total_pages(folder)
    new_page = min(total, int(page_num) + 1)
    img, status = _load_page(folder, new_page)
    return (
        gr.update(value=img),
        gr.update(value=status),
        gr.update(value=new_page),
        gr.update(interactive=new_page > 1),
        gr.update(interactive=new_page < total),
    )


def build_three_type_cd_tab() -> None:
    """构建三类变化监测展示 Tab."""
    gr.Markdown(
        "## 🌆 三类变化监测展示\n"
        "基于 MLP 下游模型的专题变化监测结果。选择变化类型后查看监测效果。\n\n"
        "> **下游指标说明**：本页面展示的是 MLP（多层感知机）在二分类变化监测任务上的交叉验证表现。"
        "`n_patches` 表示参与训练/测试的样本数；`mean_auc` 与 `std_auc` 分别是 5 折交叉验证中 AUC-ROC 的均值与标准差，"
        "AUC 衡量模型区分正负样本的能力，越接近 1 越好；`mean_f1` / `std_f1` 是 F1 分数的均值与标准差，"
        "综合了精确率（Precision）与召回率（Recall），F1 越高说明模型在查全与查准之间越平衡；"
        "`mean_ap` / `std_ap` 是 Average Precision（PR 曲线下面积）的均值与标准差，AP 对类别不平衡更敏感，"
        "AP 越高代表模型对少量变化样本的识别能力越强。"
    )

    default_category = list(_CATEGORIES.keys())[0]
    default_folder = _CATEGORIES[default_category]
    default_total = _get_total_pages(default_folder)
    m = _engine.get_metrics("v2")
    default_metric_key = _CATEGORY_METRIC_KEY.get(default_folder)
    default_metrics_md = _format_metrics(m.get(default_metric_key, {})) if default_metric_key else ""

    with gr.Row():
        with gr.Column(scale=1):
            category_select = gr.Dropdown(
                choices=list(_CATEGORIES.keys()),
                value=default_category,
                label="选择变化类型",
            )
            metrics_md = gr.Markdown(value=default_metrics_md)
            total_md = gr.Markdown(value=f"*共 {default_total} 张结果图*")
            with gr.Row():
                prev_btn = gr.Button("◀ 上一页", interactive=False)
                next_btn = gr.Button("下一页 ▶", interactive=default_total > 1)
            page_input = gr.Number(
                value=1,
                label="页码",
                minimum=1,
                maximum=max(1, default_total),
                step=1,
                precision=0,
            )
            page_status = gr.Textbox(label="页码状态", interactive=False)

        with gr.Column(scale=3):
            img_display = gr.Image(label="监测结果", height=750)

    category_select.change(
        fn=_on_category_change,
        inputs=[category_select],
        outputs=[page_input, img_display, page_status, metrics_md],
    )
    prev_btn.click(
        fn=_on_prev,
        inputs=[category_select, page_input],
        outputs=[img_display, page_status, page_input, prev_btn, next_btn],
    )
    next_btn.click(
        fn=_on_next,
        inputs=[category_select, page_input],
        outputs=[img_display, page_status, page_input, prev_btn, next_btn],
    )
    page_input.change(
        fn=_load_page,
        inputs=[gr.State(default_folder), page_input],
        outputs=[img_display, page_status],
    )

    # 初始加载默认类别第一页
    _on_category_change(default_category)
