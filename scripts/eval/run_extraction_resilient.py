#!/usr/bin/env python
""" resilient embedding 提取调度器：崩溃自动重启，断点续传。

用法:
    python run_extraction_resilient.py \
        --exp-name round4_full_vicreg_baseline \
        --config configs/round4_8gpu/round4_full_vicreg_baseline.yaml \
        --checkpoint /workspace/outputs/xuannv_round2/round4_full_vicreg_baseline/epoch_best_epoch20.pt \
        --device npu:0

提取完成后自动调用下游评估:
    --run-downstream  (默认开启)
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


EXPERIMENT_ROOT = Path("/workspace/outputs/xuannv_round2")
EVAL_SCRIPT = Path("/workspace/xuannv/scripts/eval/extract_embeddings_v2.py")
MAX_RETRIES = 20
RETRY_DELAY = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True, help="实验名称")
    p.add_argument("--config", required=True, help="配置文件路径")
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    p.add_argument("--device", default="npu:0", help="NPU 设备")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--run-downstream", type=bool, default=True, help="提取完成后是否运行下游评估")
    return p.parse_args()


def run_extraction(exp_name: str, config: str, checkpoint: str, device: str,
                   batch_size: int, save_every: int) -> bool:
    """运行提取，崩溃时自动重启。返回是否成功。"""
    output_dir = EXPERIMENT_ROOT / exp_name / "evaluation" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_file = output_dir / "patch_embeddings_partial.npz"

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--config", config,
        "--checkpoint", checkpoint,
        "--output-dir", str(output_dir),
        "--device", device,
        "--batch-size", str(batch_size),
        "--save-every", str(save_every),
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*60}")
        print(f"[Attempt {attempt}/{MAX_RETRIES}] {exp_name} on {device}")
        print(f"{'='*60}", flush=True)

        if partial_file.exists():
            data = dict(np.load(partial_file))
            processed = int(data.get("processed", [0])[0])
            print(f"[INFO] 发现断点文件，已处理 {processed}/5088 个样本")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 实时打印输出
        for line in proc.stdout:
            print(line, end="", flush=True)

        proc.wait()
        exit_code = proc.returncode

        # 检查是否成功（patch_embeddings.npz 存在且 partial 不存在）
        final_npz = output_dir / "patch_embeddings.npz"
        if final_npz.exists() and not partial_file.exists():
            print(f"\n[SUCCESS] {exp_name} 提取完成!")
            return True

        if attempt < MAX_RETRIES:
            print(f"\n[RETRY] 退出码 {exit_code}，{RETRY_DELAY}秒后重试...")
            time.sleep(RETRY_DELAY)
        else:
            print(f"\n[FAIL] {exp_name} 达到最大重试次数，放弃。")
            return False

    return False


def run_downstream_eval(exp_name: str, device: str):
    """提取完成后运行下游评估。"""
    eval_dir = EXPERIMENT_ROOT / exp_name / "evaluation"
    embeddings_dir = eval_dir / "embeddings"
    log_file = eval_dir / "downstream.log"

    scripts = [
        ("KNN", "/workspace/xuannv/scripts/eval/evaluate_knn_v2.py",
         ["--embedding-file", str(embeddings_dir / "patch_embeddings.npz"),
          "--output-dir", str(eval_dir / "downstream"), "--device", device]),
        ("MLP", "/workspace/xuannv/scripts/eval/evaluate_mlp_v2.py",
         ["--embedding-file", str(embeddings_dir / "patch_embeddings.npz"),
          "--output-dir", str(eval_dir / "downstream"), "--device", device]),
        ("CD", "/workspace/xuannv/scripts/eval/evaluate_cd_v2.py",
         ["--embedding-file", str(embeddings_dir / "patch_embeddings.npz"),
          "--output-dir", str(eval_dir / "change_detection")]),
    ]

    with open(log_file, "w") as log_fh:
        for name, script_path, args in scripts:
            print(f"\n[DOWNSTREAM] Running {name}...")
            cmd = [sys.executable, script_path] + args
            proc = subprocess.run(cmd, capture_output=True, text=True)
            log_fh.write(f"\n=== {name} ===\n")
            log_fh.write(proc.stdout)
            log_fh.write(proc.stderr)
            log_fh.flush()
            print(f"[DOWNSTREAM] {name} exit code: {proc.returncode}")


def main():
    args = parse_args()

    success = run_extraction(
        args.exp_name, args.config, args.checkpoint,
        args.device, args.batch_size, args.save_every
    )

    if success and args.run_downstream:
        run_downstream_eval(args.exp_name, args.device)
    elif not success:
        print(f"\n[FATAL] {args.exp_name} 提取失败，跳过后续评估。")
        sys.exit(1)


if __name__ == "__main__":
    main()
