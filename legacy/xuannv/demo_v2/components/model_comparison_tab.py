"""Tab: Model Comparison — 多版本模型指标横向对比."""
from __future__ import annotations

import gradio as gr

from demo_v2.utils.constants import MODEL_REGISTRY
from demo_v2.engines.mlp_engine import MLPEngine

_engine = MLPEngine()


def _build_comparison_table() -> str:
    lines = [
        "### 模型版本对比",
        "",
        "| 版本 | 显示名 | Epochs | 描述 | 检查点存在 |",
        "|------|--------|--------|------|-----------|",
    ]
    for key, info in MODEL_REGISTRY.items():
        ckpt_exists = "✅" if info["checkpoint"].exists() else "❌"
        lines.append(
            f"| {key} | {info['display_name']} | {info['epochs']} | {info['desc']} | {ckpt_exists} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "### MLP 下游指标对比（已有结果）",
        "",
    ])

    versions = _engine.list_available_versions()
    if not versions:
        lines.append("*暂无任何版本的 MLP 下游结果。*")
        return "\n".join(lines)

    # 收集所有指标字段
    all_keys = set()
    version_metrics = {}
    for v in versions:
        m = _engine.get_metrics(v)
        version_metrics[v] = m
        for cat_data in m.values():
            if isinstance(cat_data, dict):
                all_keys.update(cat_data.keys())

    display_keys = [k for k in ["category", "n_patches", "mean_auc", "std_auc", "mean_f1", "std_f1", "mean_ap", "std_ap"] if k in all_keys]

    for v in versions:
        lines.append(f"#### {v}")
        lines.append("")
        m = version_metrics.get(v, {})
        if not m:
            lines.append("*无指标数据。*")
            continue
        header = "| 类别 | " + " | ".join(display_keys) + " |"
        sep = "|------|" + "|".join(["-----"] * len(display_keys)) + "|"
        lines.append(header)
        lines.append(sep)
        for cat_key, cat_data in sorted(m.items()):
            if not isinstance(cat_data, dict):
                continue
            row_vals = []
            for dk in display_keys:
                val = cat_data.get(dk, "-")
                if isinstance(val, float):
                    val = f"{val:.4f}"
                row_vals.append(str(val))
            lines.append(f"| {cat_key} | " + " | ".join(row_vals) + " |")
        lines.append("")

    return "\n".join(lines)


def build_model_comparison_tab() -> None:
    """构建模型对比 Tab."""
    gr.Markdown(
        "## Model Comparison\n"
        "横向对比各模型版本的配置、检查点状态以及 MLP 下游指标。"
    )
    comp_md = gr.Markdown(_build_comparison_table())
    refresh_btn = gr.Button("🔄 刷新指标", variant="secondary")
    refresh_btn.click(fn=lambda: _build_comparison_table(), outputs=[comp_md])
