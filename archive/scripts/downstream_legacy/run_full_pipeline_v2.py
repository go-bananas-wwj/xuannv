#!/usr/bin/env python3
"""
Master Pipeline V2 — 自动化 Few-Shot 下游评估流程。

流程:
  1. 等待backbone训练完成
  2. 提取所有patch、月份的embedding
  3. 运行统一few-shot评估 (变化检测 + 水体检测 + 土地利用分割)
  4. 汇总结果

用法:
  python scripts/downstream/run_full_pipeline_v2.py \
    --experiment-name v2_vicreg_recon \
    --config configs/v2_vicreg_recon.yaml \
    --checkpoint-dir /workspace/outputs/v2_vicreg_recon_10ep \
    --device npu:0
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, "/workspace/xuannv")

import json
import time
import subprocess
from pathlib import Path


EXPERIMENTS = [
    "v2_vicreg_recon",
    "v2_vicreg_recon_batch",
    "v2_high_kappa",
    "v2_vicreg_25",
    "v2_baseline",
    "v2_vicreg_uniform",
    "v2_vicreg_bilateral",
    "v2_original",
]


def wait_for_checkpoint(checkpoint_dir: Path, target_epochs: int = 10, poll_interval: int = 60) -> Path | None:
    """等待训练完成并返回最佳checkpoint路径"""
    print(f"⏳ 等待训练完成: {checkpoint_dir}")

    while True:
        ckpt_files = list(checkpoint_dir.glob("epoch_best_*.pt"))
        if ckpt_files:
            best_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
            print(f"✅ 发现best checkpoint: {best_ckpt.name}")
            return best_ckpt

        epoch_files = list(checkpoint_dir.glob("epoch_*.pt"))
        epoch_numbers = []
        for f in epoch_files:
            try:
                num = int(f.stem.split("_")[1])
                epoch_numbers.append((num, f))
            except (ValueError, IndexError):
                continue

        if epoch_numbers:
            max_epoch, max_ckpt = max(epoch_numbers, key=lambda x: x[0])
            if max_epoch >= target_epochs:
                print(f"✅ 发现epoch {max_epoch} checkpoint: {max_ckpt.name}")
                return max_ckpt

        log_file = checkpoint_dir / "train.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                lines = f.readlines()
                if lines and "ERROR" in lines[-1]:
                    print(f"❌ 训练可能已失败")
                    return None

        time.sleep(poll_interval)


def run_command(cmd: list[str], desc: str) -> bool:
    """运行命令并返回是否成功"""
    print(f"\n{'='*60}")
    print(f"▶ {desc}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ 失败 (exit code {result.returncode})")
        if result.stderr:
            print(f"   stderr: {result.stderr[:500]}")
        return False

    print(f"✅ 成功")
    return True


def run_pipeline_for_experiment(
    exp_name: str,
    config_path: str,
    checkpoint_dir: Path,
    device: str,
    target_epochs: int = 10,
) -> dict:
    """为单个实验运行完整pipeline"""

    print(f"\n{'#'*70}")
    print(f"# 实验: {exp_name}")
    print(f"{'#'*70}")

    results = {"experiment": exp_name, "status": "running", "steps": {}}

    # Step 1: 等待训练完成
    ckpt_path = wait_for_checkpoint(checkpoint_dir, target_epochs)
    if ckpt_path is None:
        results["status"] = "failed"
        results["error"] = "等待checkpoint超时或训练失败"
        return results

    results["steps"]["checkpoint"] = str(ckpt_path)

    # Step 2: 提取embedding
    embedding_dir = Path("/workspace/xuannv/data/embeddings") / exp_name
    embed_cmd = [
        "python", "scripts/downstream/extract_embeddings.py",
        "--checkpoint", str(ckpt_path),
        "--config", config_path,
        "--output-dir", str(embedding_dir),
        "--device", device,
    ]
    if not run_command(embed_cmd, f"提取Embedding ({exp_name})"):
        results["status"] = "failed"
        results["error"] = "embedding提取失败"
        return results

    results["steps"]["embedding_dir"] = str(embedding_dir)

    # Step 3: 统一Few-Shot评估 (3个任务)
    eval_output = checkpoint_dir / "fewshot_results.json"
    eval_cmd = [
        "python", "scripts/downstream/fewshot_eval_unified.py",
        "--config", config_path,
        "--checkpoint", str(ckpt_path),
        "--embedding-dir", str(embedding_dir),
        "--tasks", "cd,water,landuse",
        "--k-shots", "5,10,20,50",
        "--n-splits", "5",
        "--device", device,
        "--output", str(eval_output),
    ]
    if not run_command(eval_cmd, f"Few-Shot评估 ({exp_name})"):
        results["status"] = "failed"
        results["error"] = "few-shot评估失败"
        return results

    results["steps"]["eval_output"] = str(eval_output)
    results["status"] = "success"

    # 读取评估结果摘要
    if eval_output.exists():
        with open(eval_output) as f:
            eval_data = json.load(f)
        results["eval_summary"] = eval_data.get("results", {})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", help="单个实验名称（可选）")
    parser.add_argument("--device-prefix", default="npu:")
    parser.add_argument("--target-epochs", type=int, default=10)
    args = parser.parse_args()

    all_results = []

    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = EXPERIMENTS

    for i, exp_name in enumerate(experiments):
        device = f"{args.device_prefix}{i % 8}"
        config_path = f"configs/{exp_name}.yaml"
        checkpoint_dir = Path(f"/workspace/outputs/{exp_name}_10ep")

        result = run_pipeline_for_experiment(
            exp_name, config_path, checkpoint_dir, device, args.target_epochs
        )
        all_results.append(result)

        # 保存中间结果
        with open("/workspace/xuannv/data/pipeline_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # 最终汇总
    print(f"\n{'#'*70}")
    print(f"# 全部实验完成!")
    print(f"{'#'*70}")
    for r in all_results:
        status_emoji = "✅" if r["status"] == "success" else "❌"
        print(f"{status_emoji} {r['experiment']}: {r['status']}")


if __name__ == "__main__":
    main()
