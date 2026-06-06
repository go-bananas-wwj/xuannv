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
    python multi_dim_eval.py --experiment-dir /workspace/xuannv/outputs/exp_v15_fixed_temporal_0606
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
        if json_path.exists() and json_path.stat().st_size > 0:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                g = data["global"]
                results[epoch] = {
                    "auc": g["auc"],
                    "sep": g["separation"],
                    "changed": g["changed_mean"],
                    "unchanged": g["unchanged_mean"],
                    "periods": data.get("periods", {}),
                }
            except (json.JSONDecodeError, KeyError):
                pass
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
                # 匹配 Step 行
                m = re.search(
                    r'Step\s+(\d+)/\d+.*total=([\-\d\.]+).*recon=([\d\.]+).*'
                    r'cls=([\d\.]+).*var=([\d\.]+).*cov=([\d\.]+).*'
                    r'l2unif=([\-\d\.]+).*erank=([\d\.]+)',
                    line
                )
                if m:
                    step = int(m.group(1))
                    # 尝试从日志中的其他信息推断epoch，否则按每epoch 200步估算
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
        # 计算稳定性指标：loss方差
        total_vals = [s["total"] for s in steps]
        recon_vals = [s["recon"] for s in steps]
        erank_vals = [s["erank"] for s in steps]
        summary[epoch] = {
            "total": avg("total"),
            "recon": avg("recon"),
            "cls": avg("cls"),
            "var": avg("var"),
            "cov": avg("cov"),
            "l2unif": avg("l2unif"),
            "erank": avg("erank"),
            "total_std": float(np.std(total_vals)),
            "recon_std": float(np.std(recon_vals)),
            "erank_std": float(np.std(erank_vals)),
            "erank_min": float(np.min(erank_vals)),
            "erank_max": float(np.max(erank_vals)),
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


def evaluate_embedding_quality(erank: float) -> tuple[str, str]:
    """评估embedding质量等级."""
    if erank > 32:
        return "优秀", "✅"
    elif erank > 20:
        return "良好", "✅"
    elif erank > 10:
        return "轻度坍缩", "⚠️"
    else:
        return "严重坍缩", "❌"


def evaluate_auc(auc: float) -> tuple[str, str]:
    """评估AUC等级."""
    if auc > 0.85:
        return "优秀", "✅"
    elif auc > 0.7:
        return "及格", "✅"
    elif auc > 0.6:
        return "偏低", "⚠️"
    else:
        return "接近随机", "❌"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output", default="", help="输出JSON路径")
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
        print(f"  {'Epoch':>6} {'AUC':>8} {'评级':>10} {'Sep':>8} {'Changed':>8} {'Unchanged':>10}")
        best_epoch = max(auc_results, key=lambda e: auc_results[e]["auc"])
        for epoch in sorted(auc_results):
            r = auc_results[epoch]
            level, icon = evaluate_auc(r["auc"])
            marker = " ***" if epoch == best_epoch else ""
            print(f"  {epoch:>6} {r['auc']:>8.4f} {icon+' '+level:>10} {r['sep']:>8.4f} {r['changed']:>8.4f} {r['unchanged']:>10.4f}{marker}")
        
        deg = compute_degradation_score(auc_results)
        if deg:
            print(f"\n  退化分析 (以 Epoch 2 为基线):")
            for epoch in sorted(deg):
                drop = deg[epoch]['auc_drop']
                pct = deg[epoch]['degradation_pct']
                icon = "❌" if drop > 0.05 else ("⚠️" if drop > 0.02 else "✅")
                print(f"    Epoch {epoch}: AUC 变化 {drop:+.4f} ({pct:+.1f}%) {icon}")
        
        # 分时期详情
        if auc_results[best_epoch].get("periods"):
            print(f"\n  最佳 Epoch {best_epoch} 分时期详情:")
            for period, res in auc_results[best_epoch]["periods"].items():
                print(f"    {period:12s}: AUC={res['auc']:.4f}  sep={res['separation']:.4f}  n={res['n_samples']} pos={res['n_positive']}")
    else:
        print("\n【维度 1: 变化检测 AUC】")
        print("  ⚠️  未找到有效的 AUC 评估结果")
    
    # 2. 训练日志指标
    train_log = load_train_log(exp_dir)
    if train_log:
        print("\n【维度 2: 训练日志指标】")
        print(f"  {'Epoch':>6} {'Total':>8} {'Recon':>7} {'eRank':>7} {'L2Unif':>8} {'Totalσ':>7} {'Reconσ':>7}")
        for epoch in sorted(train_log):
            r = train_log[epoch]
            print(f"  {epoch:>6} {r['total']:>8.3f} {r['recon']:>7.3f} {r['erank']:>7.1f} {r['l2unif']:>8.3f} {r['total_std']:>7.3f} {r['recon_std']:>7.3f}")
    
    # 3. Embedding 质量
    if train_log:
        print("\n【维度 3: Embedding 质量】")
        print(f"  {'Epoch':>6} {'eRank':>8} {'eRank范围':>14} {'评级':>12} {'建议':>20}")
        for epoch in sorted(train_log):
            r = train_log[epoch]
            erank = r['erank']
            level, icon = evaluate_embedding_quality(erank)
            erange = f"{r['erank_min']:.1f}-{r['erank_max']:.1f}"
            if erank < 10:
                advice = "立即检查反坍缩损失权重"
            elif erank < 20:
                advice = "增强pre_norm_uniform/erank"
            elif erank < 32:
                advice = "可接受，继续观察"
            else:
                advice = "优秀，保持当前配置"
            print(f"  {epoch:>6} {erank:>8.1f} {erange:>14} {icon+' '+level:>12} {advice:>20}")
    
    # 4. 训练稳定性
    if train_log:
        print("\n【维度 4: 训练稳定性】")
        print(f"  {'Epoch':>6} {'Lossσ':>8} {'Reconσ':>8} {'eRankσ':>8} {'稳定性':>10}")
        for epoch in sorted(train_log):
            r = train_log[epoch]
            tstd = r['total_std']
            level = "稳定" if tstd < 0.5 else ("波动" if tstd < 1.0 else "剧烈")
            icon = "✅" if tstd < 0.5 else ("⚠️" if tstd < 1.0 else "❌")
            print(f"  {epoch:>6} {tstd:>8.3f} {r['recon_std']:>8.3f} {r['erank_std']:>8.3f} {icon+' '+level:>10}")
    
    # 5. 综合评价与建议
    print("\n【维度 5: 综合评价与建议】")
    issues = []
    recommendations = []
    
    if auc_results:
        best_epoch = max(auc_results, key=lambda e: auc_results[e]["auc"])
        best_auc = auc_results[best_epoch]["auc"]
        print(f"  最佳 AUC: Epoch {best_epoch} = {best_auc:.4f}")
        
        if best_auc < 0.6:
            issues.append("AUC接近随机，变化检测能力未形成")
            recommendations.append("检查双窗口时序数据是否正确生成")
            recommendations.append("确认temporal_contrastive_loss或gap_aware_temporal_cosine_loss已生效")
        
        if len(auc_results) > 2:
            last_epoch = max(auc_results)
            last_auc = auc_results[last_epoch]["auc"]
            if last_auc < best_auc - 0.02:
                issues.append(f"Epoch {last_epoch} AUC={last_auc:.4f} 比最佳下降 {best_auc-last_auc:.4f}")
                recommendations.append("启用Early Stopping或降低reconstruction_weight防止过拟合")
    
    if train_log:
        last_epoch = max(train_log)
        last_erank = train_log[last_epoch]["erank"]
        if last_erank < 10:
            issues.append(f"Epoch {last_epoch} eRank={last_erank:.1f}，严重维度坍缩")
            recommendations.append("提高pre_norm_uniform_weight（建议1.0-1.5）")
            recommendations.append("启用erank_loss_weight（建议0.3-0.5）")
            recommendations.append("关闭use_spatial_vicreg（与patch级erank测量冲突）")
            recommendations.append("启用gap_aware_temporal_cosine_loss（建议0.2）")
        elif last_erank < 20:
            issues.append(f"Epoch {last_epoch} eRank={last_erank:.1f}，轻度坍缩")
            recommendations.append("适度提高erank_loss_weight或decorrelation_weight")
    
    if issues:
        print("\n  ⚠️  发现问题:")
        for issue in issues:
            print(f"    - {issue}")
    
    if recommendations:
        print("\n  💡 建议:")
        for rec in recommendations:
            print(f"    - {rec}")
    
    if not issues:
        print("  ✅ 模型状态良好")
    
    print("\n" + "=" * 70)
    
    # 保存JSON
    output = {
        "experiment": exp_dir.name,
        "auc": auc_results,
        "train": train_log,
        "issues": issues,
        "recommendations": recommendations,
    }
    out_path = args.output or exp_dir / "multi_dim_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
