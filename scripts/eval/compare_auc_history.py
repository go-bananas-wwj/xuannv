#!/usr/bin/env python3
"""对比多个实验的 AUC 评估历史"""
import json, glob, sys
from pathlib import Path

def load_auc(experiment_dir):
    results = {}
    for eval_dir in sorted(Path(experiment_dir).glob("eval_epoch_*")):
        epoch = int(eval_dir.name.split("_")[-1])
        json_path = eval_dir / "auc_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            results[epoch] = {
                "auc": data["global"]["auc"],
                "sep": data["global"]["separation"],
            }
    return results

def main():
    experiments = {
        "v14 (broken)": "/workspace/outputs/exp_v14_anti_collapse_0605",
        "v15 (fix)": "/workspace/outputs/exp_v15_fixed_temporal_0606",
    }
    
    print("=" * 60)
    print("  AUC 评估历史对比")
    print("=" * 60)
    
    for name, path in experiments.items():
        results = load_auc(path)
        if not results:
            print(f"\n{name}: 暂无评估数据")
            continue
        print(f"\n{name}:")
        print(f"  {'Epoch':>6} {'AUC':>8} {'Sep':>8}")
        print(f"  {'-'*6} {'-'*8} {'-'*8}")
        best_epoch, best_auc = max(results.items(), key=lambda x: x[1]["auc"])
        for epoch in sorted(results):
            marker = " ***" if epoch == best_epoch else ""
            print(f"  {epoch:>6} {results[epoch]['auc']:>8.4f} {results[epoch]['sep']:>8.4f}{marker}")
        print(f"  Best: Epoch {best_epoch} AUC={best_auc['auc']:.4f}")

if __name__ == "__main__":
    main()
