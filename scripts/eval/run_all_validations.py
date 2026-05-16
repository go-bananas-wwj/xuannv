#!/usr/bin/env python3
"""并行运行8个实验的 Bare AUC 验证."""
import subprocess, sys, time
from pathlib import Path

EXPERIMENTS = [
    ("aef_baseline",     0),
    ("aef_high_consist", 1),
    ("aef_no_static",    2),
    ("aef_skip_l2",      3),
    ("aef_diff_recon",   4),
    ("aef_high_kappa",   5),
    ("aef_cyclic_unif",  6),
    ("aef_no_uniform",   7),
]

processes = []
print("="*60)
print("  Launching 8 parallel validation jobs")
print("="*60)

for exp_name, gpu_idx in EXPERIMENTS:
    ckpt = f"/workspace/outputs/{exp_name}/epoch_best_epoch"
    # 查找实际的 checkpoint 文件
    ckpt_dir = Path(f"/workspace/outputs/{exp_name}")
    ckpt_files = list(ckpt_dir.glob("epoch_best_epoch*.pt"))
    if not ckpt_files:
        print(f"  [SKIP] {exp_name}: no checkpoint found")
        continue
    ckpt_path = str(ckpt_files[0])
    output_dir = f"/workspace/outputs/{exp_name}/eval"

    cmd = [
        "python3", "scripts/eval/validate_aef_bare.py",
        "--config", f"configs/{exp_name}.yaml",
        "--checkpoint", ckpt_path,
        "--device", f"npu:{gpu_idx}",
        "--output-dir", output_dir,
    ]

    log_path = Path(output_dir) / "validate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{exp_name}] NPU {gpu_idx} → {ckpt_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd="/workspace/xuannv",
        env={**dict(subprocess.os.environ), "ASCEND_RT_VISIBLE_DEVICES": str(gpu_idx)},
    )
    processes.append((exp_name, proc))

print(f"\n  Launched {len(processes)} jobs. Waiting...")

# 等待所有进程完成
for exp_name, proc in processes:
    proc.wait()
    status = "✅" if proc.returncode == 0 else "❌"
    print(f"  {status} {exp_name} (exit={proc.returncode})")

print("\n  All validations complete!")
