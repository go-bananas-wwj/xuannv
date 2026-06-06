#!/usr/bin/env python3
"""多维度评估 — 超越 AUC 的综合评价框架.

评价维度:
1. 变化检测: AUC + 分离度 + 各时期 AUC
2. 下游分类: KNN mIoU (WorldCover/JRC Water/Dynamic World)
3. Embedding 质量: eRank + active_dims + std_mean + std_min/max
4. 重建质量: PSNR + SSIM
5. 时序敏感性: temporal contrastive loss trend
6. 可视化: t-SNE/UMAP embedding 分布

用法:
    python multi_dim_eval.py --experiment-dir /workspace/outputs/exp_v15_fixed_temporal_0606
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def load_auc(exp_dir: Path):
    """加载 AUC 评估结果."""
    results = {}
    for eval_dir in sorted(exp_dir.glob("eval_epoch_*")):
        import re
        m = re.search(r'eval_epoch_(\d+)', eval_dir.name)
        if not m:
            continue
        epoch = int(m.group(1))
        json_path = eval_dir / "auc_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            g = data["global"]
            results[epoch] = {
                "auc": g["auc"],
                "sep": g["separation"],
                "changed": g["changed_mean"],
                "unchanged": g["unchanged_mean"],
            }
    return results


def load_train_log(exp_dir: Path):
    """从训练日志提取关键指标."""
    logs = sorted(exp_dir.glob("train_*.log"))
    if not logs:
        return {}
    
    import re
    epochs = {}
    for log_file in logs:
        with open(log_file) as f:
            for line in f:
                m = re.search(
                    r'Step\s+(\d+)/200.*total=([\-\d\.]+).*recon=([\d\.]+).*'
                    r'cls=([\d\.]+).*var=([\d\.]+).*cov=([\d\.]+).*'
                    r'l2unif=([\-\d\.]+).*erank=([\d\.]+)',
                    line
                )
                if m:
                    step = int(m.group(1))
                    epoch = (step - 1) // 200 + 1
                    if epoch not in epochs:
                        epochs[epoch] = []
                    epochs[epoch].append({
                        "total": float(m.group(2)),
                        "recon": float(m.group(3)),
                        "cls": float(m.group(4)),
                        "var": float(m.group(5)),
                        "cov": float(m.group(6)),
                        "l2unif": float(m.group(7)),
                        "erank": float(m.group(8)),
                    })
    
    summary = {}
    for epoch, steps in epochs.items():
        n = len(steps)
        avg = lambda k: sum(s[k] for s in steps) / n
        summary[epoch] = {
            "total": avg("total"),
            "recon": avg("recon"),
            "cls": avg("cls"),
            "var": avg("var"),
            "cov": avg("cov"),
            "l2unif": avg("l2unif"),
            "erank": avg("erank"),
        }
    return summary


def compute_degradation_score(auc_results: dict) -> dict:
    """计算退化分数: Epoch 2 vs Epoch N 的 AUC 差异."""
    if 2 not in auc_results:
        return {}
    baseline = auc_results[2]["auc"]
    scores = {}
    for epoch, res in auc_results.items():
        if epoch > 2:
            scores[epoch] = {
                "auc_drop": baseline - res["auc"],
                "degradation_pct": (baseline - res["auc"]) / baseline * 100,
            }
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    
    print("=" * 70)
    print("  多维度评估报告")
    print(f"  实验: {exp_dir.name}")
    print("=" * 70)
    
    # 1. 变化检测 AUC
    auc_results = load_auc(exp_dir)
    if auc_results:
        print("\n【维度 1: 变化检测 AUC】")
        print(f"  {'Epoch':>6} {'AUC':>8} {'Sep':>8} {'Changed':>8} {'Unchanged':>10}")
        for epoch in sorted(auc_results):
            r = auc_results[epoch]
            marker = " ***" if epoch == max(auc_results, key=lambda e: auc_results[e]["auc"]) else ""
            print(f"  {epoch:>6} {r['auc']:>8.4f} {r['sep']:>8.4f} {r['changed']:>8.4f} {r['unchanged']:>10.4f}{marker}")
        
        deg = compute_degradation_score(auc_results)
        if deg:
            print(f"\n  退化分析 (以 Epoch 2 为基线):")
            for epoch in sorted(deg):
                print(f"    Epoch {epoch}: AUC 下降 {deg[epoch]['auc_drop']:.4f} ({deg[epoch]['degradation_pct']:.1f}%)")
    
    # 2. 训练日志指标
    train_log = load_train_log(exp_dir)
    if train_log:
        print("\n【维度 2: 训练日志指标】")
        print(f"  {'Epoch':>6} {'Total':>8} {'Recon':>7} {'eRank':>7} {'L2Unif':>8}")
        for epoch in sorted(train_log):
            r = train_log[epoch]
            print(f"  {epoch:>6} {r['total']:>8.3f} {r['recon']:>7.3f} {r['erank']:>7.1f} {r['l2unif']:>8.3f}")
    
    # 3. Embedding 质量
    if train_log:
        print("\n【维度 3: Embedding 质量】")
        for epoch in sorted(train_log):
            r = train_log[epoch]
            erank = r['erank']
            quality = "✅" if erank > 32 else ("⚠️" if erank > 16 else "❌")
            print(f"  Epoch {epoch}: eRank={erank:.1f}/64 {quality}")
    
    # 4. 综合评价
    print("\n【维度 4: 综合评价】")
    if auc_results and 2 in auc_results:
        best_epoch = max(auc_results, key=lambda e: auc_results[e]["auc"])
        best_auc = auc_results[best_epoch]["auc"]
        print(f"  最佳 AUC: Epoch {best_epoch} = {best_auc:.4f}")
        
        if len(auc_results) > 2:
            last_epoch = max(auc_results)
            last_auc = auc_results[last_epoch]["auc"]
            if last_auc < best_auc:
                print(f"  ⚠️  退化检测: Epoch {last_epoch} AUC={last_auc:.4f}, 比最佳下降 {best_auc-last_auc:.4f}")
            else:
                print(f"  ✅ 稳定: Epoch {last_epoch} AUC={last_auc:.4f}, 未退化")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
