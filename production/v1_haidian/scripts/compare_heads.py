#!/usr/bin/env python3
"""海淀 6 任务下游头横向对比实验.

依次运行 linear / mlp_torch / unet 等 head，汇总每个任务的 AUC、F1、IoU 等指标，
输出 CSV、JSON 和对比图。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import backbone, haidian_tasks


HEAD_CHOICES = ["linear", "mlp_torch", "mlp_torch_v2", "unet"]
TASK_NAMES = haidian_tasks.CLASS_NAMES + list(haidian_tasks.MERGED_TASKS.keys())
TASK_NAMES_CN = {
    **haidian_tasks.CLASS_NAMES_CN,
    "shigongjiandu": "施工工地监测",
}


def parse_args():
    parser = argparse.ArgumentParser(description="海淀下游头横向对比")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    parser.add_argument("--output-dir", default="outputs/head_ablation")
    parser.add_argument("--cache", default=None, help="embedding 缓存路径（默认 output-dir/.cache/embeddings.npz）")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", default="bitemporal", choices=["single", "bitemporal"])
    parser.add_argument(
        "--heads",
        default=",".join(HEAD_CHOICES),
        help="逗号分隔的 head 列表，例如 linear,mlp_torch,unet",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="只运行指定任务（例如 shigongjiandu），不指定则运行全部 6 个原始任务",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_metrics(head_out_dir: Path) -> dict[str, dict]:
    path = head_out_dir / "metrics_all.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _build_table(results: dict[str, dict[str, dict]]) -> list[dict]:
    rows: list[dict] = []
    if not results:
        return rows
    # 从第一个 head 的结果中推导实际运行的任务列表
    first_head = next(iter(results.values()))
    actual_tasks = list(first_head.keys())
    for task in actual_tasks:
        cn = TASK_NAMES_CN.get(task, task)
        for head, metrics_by_task in results.items():
            m = metrics_by_task.get(task, {})
            if not m or m.get("skipped"):
                rows.append(
                    {
                        "task": task,
                        "task_cn": cn,
                        "head": head,
                        "n_train": None,
                        "n_test": None,
                        "pos_ratio": None,
                        "accuracy": None,
                        "balanced_accuracy": None,
                        "f1": None,
                        "iou": None,
                        "auc": None,
                    }
                )
                continue
            rows.append(
                {
                    "task": task,
                    "task_cn": cn,
                    "head": head,
                    "n_train": m.get("n_train_patches"),
                    "n_test": m.get("n_test_patches"),
                    "pos_ratio": m.get("pos_ratio"),
                    "accuracy": m.get("accuracy"),
                    "balanced_accuracy": m.get("balanced_accuracy"),
                    "f1": m.get("f1"),
                    "iou": m.get("iou"),
                    "auc": m.get("auc"),
                }
            )
    return rows


def _plot_comparison(rows: list[dict], out_dir: Path) -> None:
    heads = sorted({r["head"] for r in rows})
    tasks = sorted({r["task"] for r in rows}, key=lambda t: list(TASK_NAMES_CN.keys()).index(t) if t in TASK_NAMES_CN else 999)
    x = np.arange(len(tasks))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("auc", "AUC", 0, 0),
        ("f1", "F1", 0, 1),
        ("iou", "IoU", 1, 0),
        ("balanced_accuracy", "Balanced Accuracy", 1, 1),
    ]
    for key, title, ri, ci in metrics:
        ax = axes[ri, ci]
        for i, head in enumerate(heads):
            vals = [
                next(
                    (r[key] for r in rows if r["task"] == t and r["head"] == head),
                    None,
                )
                for t in tasks
            ]
            vals = [v if v is not None else 0.0 for v in vals]
            ax.bar(x + i * width, vals, width, label=head)
        ax.set_xticks(x + width * (len(heads) - 1) / 2)
        ax.set_xticklabels([TASK_NAMES_CN.get(t, t) for t in tasks], rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"各任务 {title} 对比")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "head_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_or_extract_embeddings(
    model: Any,
    dataset: Any,
    label_dir: Path,
    cache_path: Path,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """只提取一次 embedding 并缓存，避免每个 head/每个任务重复推理。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[compare_heads] 加载 embedding 缓存: {cache_path}")
        data = np.load(cache_path, allow_pickle=False)
        pids = [str(p) for p in data["patch_ids"]]
        emb_dec = {pid: data["emb_dec"][i] for i, pid in enumerate(pids)}
        emb_apr = {pid: data["emb_apr"][i] for i, pid in enumerate(pids)}
        return emb_dec, emb_apr

    print("[compare_heads] 提取全部标注 patch 的 embedding ...")
    all_pids = haidian_tasks.discover_labeled_patches(label_dir)
    all_pids = [
        p
        for p in all_pids
        if (label_dir / f"{p}_20260430_rgb_uint8.json").exists()
    ]
    emb_dec = backbone.extract_embeddings_for_patches(
        model, dataset, all_pids, 2025, 12, device
    )
    emb_apr = backbone.extract_embeddings_for_patches(
        model, dataset, all_pids, 2026, 4, device
    )
    # 只保留前后两期都成功提取的 patch
    common_pids = sorted(set(emb_dec.keys()) & set(emb_apr.keys()))
    dec_arr = np.stack([emb_dec[p] for p in common_pids], axis=0)
    apr_arr = np.stack([emb_apr[p] for p in common_pids], axis=0)
    np.savez_compressed(
        cache_path,
        patch_ids=np.array(common_pids),
        emb_dec=dec_arr,
        emb_apr=apr_arr,
    )
    print(f"[compare_heads] 已缓存 {len(common_pids)} 个 patch 的 embedding")
    return emb_dec, emb_apr


def main() -> int:
    args = parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    unknown = [h for h in heads if h not in HEAD_CHOICES]
    if unknown:
        raise ValueError(f"未知 head: {unknown}，可选: {HEAD_CHOICES}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[str] | None = None
    if args.task:
        if args.task not in TASK_NAMES:
            raise ValueError(f"未知任务: {args.task}，可选: {TASK_NAMES}")
        tasks = [args.task]

    print("[compare_heads] 加载生产 backbone ...")
    model, dataset, _ = backbone.load_production_model(args.model_dir, device=args.device)

    label_dir = Path(args.label_dir)
    cache_path = Path(args.cache) if args.cache else out_dir / ".cache" / "embeddings.npz"
    emb_dec, emb_apr = _load_or_extract_embeddings(
        model, dataset, label_dir, cache_path, args.device
    )

    results: dict[str, dict[str, dict]] = {}
    for head in heads:
        head_out = out_dir / head
        head_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[compare_heads] ===== 运行 head: {head} =====")
        summary = haidian_tasks.run_all_tasks(
            model_dir=args.model_dir,
            label_dir=str(label_dir),
            output_dir=str(head_out),
            device=args.device,
            mode=args.mode,
            classifier=head,
            model=model,
            dataset=dataset,
            emb_dec=emb_dec,
            emb_apr=emb_apr,
            tasks=tasks,
        )
        results[head] = summary

    rows = _build_table(results)

    # CSV
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    json_path = out_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # Markdown table
    md_lines = [
        "| 任务 | Head | AUC | F1 | IoU | Balanced Acc | 正样本比例 |",
        "|------|------|-----|----|----|-------------|-----------|",
    ]
    for r in rows:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "-"
        f1 = f"{r['f1']:.3f}" if r["f1"] is not None else "-"
        iou = f"{r['iou']:.3f}" if r["iou"] is not None else "-"
        bacc = f"{r['balanced_accuracy']:.3f}" if r["balanced_accuracy"] is not None else "-"
        pos = f"{r['pos_ratio']*100:.2f}%" if r["pos_ratio"] is not None else "-"
        md_lines.append(
            f"| {r['task_cn']} ({r['task']}) | {r['head']} | {auc} | {f1} | {iou} | {bacc} | {pos} |"
        )
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    _plot_comparison(rows, out_dir)

    print(f"\n[compare_heads] 对比结果已保存到: {out_dir}")
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
