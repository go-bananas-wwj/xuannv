#!/usr/bin/env python3
"""
compare_with_aef.py — 将 v14 评估结果与 AEF baseline 对比
生成对比表格和结论
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# AEF baseline 指标（来自历史实验）
AEF_BASELINE = {
    "knn_worldcover_miou": 0.42,
    "knn_jrc_water_miou": 0.65,
    "knn_dynamic_world_miou": 0.38,
    "change_detection_auc": 0.72,
    "erank": 8.5,
    "active_dims": 45,
}

def load_eval_result(eval_dir: Path) -> dict:
    """加载评估结果"""
    result = {}
    # KNN eval
    knn_file = eval_dir / "knn" / "summary.json"
    if knn_file.exists():
        with open(knn_file) as f:
            result["knn"] = json.load(f)
    # AUC eval
    for sub in eval_dir.iterdir():
        if sub.is_dir() and (sub / "auc_results.json").exists():
            with open(sub / "auc_results.json") as f:
                result["auc"] = json.load(f)
            break
    return result

def compare(eval_dir: str):
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        print(f"[错误] 评估目录不存在: {eval_dir}")
        sys.exit(1)
    
    result = load_eval_result(eval_path)
    
    print("=" * 60)
    print("  v14 (anti-collapse) vs AEF Baseline 对比")
    print("=" * 60)
    
    # KNN 对比
    if "knn" in result:
        print("\n--- KNN 下游任务 mIoU ---")
        print(f"{'任务':<25} {'AEF':>8} {'v14':>8} {'Δ':>8}")
        print("-" * 55)
        for task, baseline in [
            ("WorldCover", AEF_BASELINE["knn_worldcover_miou"]),
            ("JRC Water", AEF_BASELINE["knn_jrc_water_miou"]),
            ("Dynamic World", AEF_BASELINE["knn_dynamic_world_miou"]),
        ]:
            key = task.lower().replace(" ", "_")
            v14_val = result["knn"].get(key, 0.0)
            delta = v14_val - baseline
            print(f"{task:<25} {baseline:>8.3f} {v14_val:>8.3f} {delta:>+8.3f}")
    
    # AUC 对比
    if "auc" in result:
        print("\n--- 变化检测 AUC ---")
        v14_auc = result["auc"].get("mean_auc", 0.0)
        baseline_auc = AEF_BASELINE["change_detection_auc"]
        delta = v14_auc - baseline_auc
        print(f"{'Metric':<25} {'AEF':>8} {'v14':>8} {'Δ':>8}")
        print("-" * 55)
        print(f"{'Mean AUC':<25} {baseline_auc:>8.3f} {v14_auc:>8.3f} {delta:>+8.3f}")
    
    # Embedding 质量
    print("\n--- Embedding 质量 ---")
    print(f"{'Metric':<25} {'AEF':>8} {'v14':>8} {'Δ':>8}")
    print("-" * 55)
    # erank 需要从训练日志中提取，这里暂时用占位
    print(f"{'eRank (target >10)':<25} {AEF_BASELINE['erank']:>8.1f} {'?':>8} {'?':>8}")
    print(f"{'Active Dims (target 64)':<25} {AEF_BASELINE['active_dims']:>8} {'?':>8} {'?':>8}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python compare_with_aef.py <eval_dir>")
        print("示例: python compare_with_aef.py /workspace/xuannv/outputs/exp_v14_anti_collapse_0605/eval_epoch_10")
        sys.exit(1)
    compare(sys.argv[1])
