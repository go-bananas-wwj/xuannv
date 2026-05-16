#!/usr/bin/env python3
"""批量评估结果对比报告生成器.

用法:
    python scripts/eval/generate_comparison_report.py \
        --input-dir /workspace/outputs \
        --pattern "round8_single_exp*" \
        --output /workspace/outputs/round8_comparison.md
"""
from __future__ import annotations

import sys, json, glob, argparse, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/workspace/outputs", help="包含实验输出目录的根目录")
    parser.add_argument("--pattern", default="round8_single_exp*", help="匹配实验目录的 glob 模式")
    parser.add_argument("--output", default="/workspace/outputs/round8_comparison.md", help="输出 Markdown 报告路径")
    return parser.parse_args()


def load_all_results(input_dir, pattern):
    """加载所有匹配的 eval_results.json."""
    dirs = sorted(glob.glob(f"{input_dir}/{pattern}"))
    results = []
    for d in dirs:
        json_path = os.path.join(d, "eval_results.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                exp_name = os.path.basename(d)
                data["_exp_name"] = exp_name
                data["_dir"] = d
                results.append(data)
            except Exception as e:
                print(f"  跳过 {json_path}: {e}")
    return results


def extract_key_metrics(data):
    """从单个实验结果中提取关键指标."""
    m = {}
    emb = data.get("embedding", {})
    cd = data.get("change_detection", {})
    ds = data.get("downstream", {})

    # Embedding 质量
    m["rankme"] = emb.get("rankme", float('nan'))
    m["stable_rank"] = emb.get("stable_rank", float('nan'))
    m["active_dims_t0.10"] = emb.get("active_dims_t0.1", float('nan'))
    m["raw_uniformity"] = emb.get("raw_uniformity", float('nan'))
    m["temporal_disc"] = emb.get("temporal_discriminability", float('nan'))

    # 变化检测
    bare = cd.get("bare_auc", {})
    m["bare_auc"] = bare.get("mean", float('nan')) if isinstance(bare, dict) else float('nan')
    cdh = cd.get("cdhead_auc", {})
    m["cdhead_k1"] = cdh.get("k1", {}).get("mean", float('nan')) if isinstance(cdh, dict) else float('nan')
    m["cdhead_k5"] = cdh.get("k5", {}).get("mean", float('nan')) if isinstance(cdh, dict) else float('nan')

    # 下游任务
    wc = ds.get("worldcover", {})
    m["worldcover_miou"] = wc.get("miou", float('nan')) if isinstance(wc, dict) else float('nan')
    m["worldcover_bacc"] = wc.get("balanced_accuracy", float('nan')) if isinstance(wc, dict) else float('nan')

    dw = ds.get("dynamic_world", {})
    m["dynamic_world_miou"] = dw.get("miou", float('nan')) if isinstance(dw, dict) else float('nan')
    m["dynamic_world_bacc"] = dw.get("balanced_accuracy", float('nan')) if isinstance(dw, dict) else float('nan')

    jrc = ds.get("jrc_water", {})
    m["jrc_iou"] = jrc.get("iou", float('nan')) if isinstance(jrc, dict) else float('nan')
    m["jrc_auc"] = jrc.get("auc", float('nan')) if isinstance(jrc, dict) else float('nan')

    osm = ds.get("osm_buildings", {})
    m["osm_iou"] = osm.get("iou", float('nan')) if isinstance(osm, dict) else float('nan')
    m["osm_auc"] = osm.get("auc", float('nan')) if isinstance(osm, dict) else float('nan')

    return m


def format_value(v, fmt=".4f"):
    if v is None or v != v:  # NaN check
        return "N/A"
    return f"{v:{fmt}}"


def rank_column(values, higher_is_better=True):
    """为每列返回排名 (1=best)."""
    # values: list of (exp_name, value)
    valid = [(name, v) for name, v in values if v == v]  # filter NaN
    if not valid:
        return {}
    sorted_items = sorted(valid, key=lambda x: x[1], reverse=higher_is_better)
    ranks = {}
    for rank, (name, _) in enumerate(sorted_items, 1):
        ranks[name] = rank
    return ranks


def generate_markdown(all_data, all_metrics):
    lines = []
    lines.append("# Round 8 实验评估对比报告")
    lines.append(f"\n生成时间: {datetime.now().isoformat()}")
    lines.append(f"实验数: {len(all_data)}\n")

    # 配置摘要
    lines.append("## 1. 实验配置摘要\n")
    lines.append("| 实验 | Config | Checkpoint |")
    lines.append("|------|--------|------------|")
    for d in all_data:
        lines.append(f"| {d['_exp_name']} | {os.path.basename(d['config'])} | {os.path.basename(d['checkpoint'])} |")
    lines.append("")

    # Embedding 质量
    lines.append("## 2. Embedding 质量分析\n")
    lines.append("| 实验 | RankMe | StableRank | ActiveDims | RawUnif | TempDisc |")
    lines.append("|------|--------|------------|------------|---------|----------|")
    for d, m in zip(all_data, all_metrics):
        lines.append(f"| {d['_exp_name']} | {format_value(m['rankme'])} | {format_value(m['stable_rank'], '.2f')} | {format_value(m['active_dims_t0.10'], '.0f')} | {format_value(m['raw_uniformity'])} | {format_value(m['temporal_disc'])} |")
    lines.append("")

    # 变化检测
    lines.append("## 3. 变化检测评估\n")
    lines.append("| 实验 | Bare AUC | CDHead K=1 | CDHead K=5 |")
    lines.append("|------|----------|------------|------------|")
    for d, m in zip(all_data, all_metrics):
        lines.append(f"| {d['_exp_name']} | {format_value(m['bare_auc'])} | {format_value(m['cdhead_k1'])} | {format_value(m['cdhead_k5'])} |")
    lines.append("")

    # 下游任务
    lines.append("## 4. 语义分割评估 (Linear Probe)\n")
    lines.append("| 实验 | WorldCover mIoU | WorldCover BAcc | DynamicWorld mIoU | DynamicWorld BAcc |")
    lines.append("|------|-----------------|-----------------|-------------------|-------------------|")
    for d, m in zip(all_data, all_metrics):
        lines.append(f"| {d['_exp_name']} | {format_value(m['worldcover_miou'])} | {format_value(m['worldcover_bacc'])} | {format_value(m['dynamic_world_miou'])} | {format_value(m['dynamic_world_bacc'])} |")
    lines.append("")

    lines.append("## 5. 二值分割评估 (Linear Probe)\n")
    lines.append("| 实验 | JRC Water IoU | JRC Water AUC | OSM Buildings IoU | OSM Buildings AUC |")
    lines.append("|------|---------------|---------------|-------------------|-------------------|")
    for d, m in zip(all_data, all_metrics):
        lines.append(f"| {d['_exp_name']} | {format_value(m['jrc_iou'])} | {format_value(m['jrc_auc'])} | {format_value(m['osm_iou'])} | {format_value(m['osm_auc'])} |")
    lines.append("")

    # 综合排名
    lines.append("## 6. 综合排名\n")

    metric_configs = [
        ("bare_auc", True, "Bare AUC"),
        ("cdhead_k1", True, "CDHead K=1 AUC"),
        ("worldcover_miou", True, "WorldCover mIoU"),
        ("dynamic_world_miou", True, "DynamicWorld mIoU"),
        ("jrc_auc", True, "JRC Water AUC"),
        ("osm_auc", True, "OSM Buildings AUC"),
        ("rankme", True, "RankMe"),
        ("temporal_disc", True, "Temporal Discriminability"),
    ]

    # 计算每个指标下各实验的排名
    rank_matrix = {d['_exp_name']: {} for d in all_data}
    for key, higher, label in metric_configs:
        values = [(d['_exp_name'], m[key]) for d, m in zip(all_data, all_metrics)]
        ranks = rank_column(values, higher)
        for exp, rank in ranks.items():
            rank_matrix[exp][label] = rank

    # 平均排名
    lines.append("| 实验 | " + " | ".join([label for _, _, label in metric_configs]) + " | 平均排名 |")
    lines.append("|------|" + "|".join(["--------"] * len(metric_configs)) + "|----------|")
    for d in all_data:
        exp = d['_exp_name']
        ranks = [rank_matrix[exp].get(label, '-') for _, _, label in metric_configs]
        valid_ranks = [r for r in ranks if isinstance(r, int)]
        avg_rank = f"{sum(valid_ranks)/len(valid_ranks):.1f}" if valid_ranks else "N/A"
        rank_str = " | ".join([str(r) for r in ranks])
        lines.append(f"| {exp} | {rank_str} | {avg_rank} |")
    lines.append("")

    # 最佳实验
    lines.append("## 7. 最佳实验分析\n")
    best_by_metric = {}
    for key, higher, label in metric_configs:
        values = [(d['_exp_name'], m[key]) for d, m in zip(all_data, all_metrics) if m[key] == m[key]]
        if values:
            if higher:
                best = max(values, key=lambda x: x[1])
            else:
                best = min(values, key=lambda x: x[1])
            best_by_metric[label] = best

    for label, (exp, val) in sorted(best_by_metric.items()):
        lines.append(f"- **{label}**: `{exp}` = {format_value(val)}")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    print(f"扫描: {args.input_dir}/{args.pattern}")

    all_data = load_all_results(args.input_dir, args.pattern)
    if not all_data:
        print("  未找到任何评估结果!")
        return

    print(f"  加载: {len(all_data)} 个实验")

    all_metrics = [extract_key_metrics(d) for d in all_data]

    # 保存 JSON
    json_output = args.output.replace(".md", ".json")
    combined = {
        "experiments": [{**{"name": d['_exp_name']}, **m} for d, m in zip(all_data, all_metrics)],
        "generated_at": datetime.now().isoformat(),
    }
    with open(json_output, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  JSON: {json_output}")

    # 生成 Markdown
    md = generate_markdown(all_data, all_metrics)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"  Markdown: {args.output}")


if __name__ == "__main__":
    main()
