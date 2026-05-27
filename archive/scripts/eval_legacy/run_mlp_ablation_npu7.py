#!/usr/bin/env python
"""MLP 下游分类消融实验 — NPU7 串行批量运行"""
from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

EMB = "/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation/embeddings/patch_embeddings.npz"
OUT = Path("/workspace/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation/mlp_ablation")
OUT.mkdir(parents=True, exist_ok=True)

PYTHON = "/root/miniconda3/envs/xuannv/bin/python"
SCRIPT = "/workspace/xuannv/scripts/eval/evaluate_mlp_v2.py"

EXPERIMENTS = [
    ("baseline",      ["--head-type", "mlp", "--epochs", "50"]),
    ("class_weight",  ["--head-type", "mlp", "--epochs", "50", "--use-class-weight"]),
    ("class_focal",   ["--head-type", "mlp", "--epochs", "50", "--use-class-weight", "--use-focal"]),
    ("mlpv2",         ["--head-type", "mlpv2", "--epochs", "50", "--hidden-dim", "512", "--use-class-weight"]),
]

results = {}

for name, extra_args in EXPERIMENTS:
    exp_out = OUT / name
    log_file = OUT / f"{name}.log"
    
    print(f"\n{'='*60}")
    print(f"[Experiment] {name}")
    print(f"{'='*60}")
    
    cmd = [
        PYTHON, SCRIPT,
        "--embedding-file", EMB,
        "--output-dir", str(exp_out),
        "--device", "npu:0",
    ] + extra_args
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")
    
    env = {"ASCEND_RT_VISIBLE_DEVICES": "7", "PYTHONUNBUFFERED": "1"}
    
    with open(log_file, "w") as f:
        f.write(f"# Experiment: {name}\n")
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.write(f"# {'='*60}\n\n")
        f.flush()
        
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**env},
            cwd="/workspace/xuannv"
        )
        proc.wait()
        returncode = proc.returncode
    
    # 读取结果
    summary_file = exp_out / "mlp_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results[name] = json.load(f)
        print(f"✅ {name}: {results[name]}")
    else:
        print(f"❌ {name}: 失败或没有输出 (returncode={returncode})")

# 汇总
print(f"\n{'='*60}")
print("汇总结果")
print(f"{'='*60}")
for name, res in results.items():
    print(f"\n--- {name} ---")
    for task, metrics in res.items():
        acc = metrics.get('accuracy', 0)
        miou = metrics.get('mean_iou', 0)
        print(f"  {task:20s}: Acc={acc:.4f}, mIoU={miou:.4f}")

# 保存汇总
with open(OUT / "ablation_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 全部完成! 输出目录: {OUT}")
