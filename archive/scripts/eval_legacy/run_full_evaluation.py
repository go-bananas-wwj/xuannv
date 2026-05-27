#!/usr/bin/env python
"""一键运行完整评估流程。

用法:
    python run_full_evaluation.py \
        --config configs/round4_8gpu/round4_full_vicreg_baseline.yaml \
        --checkpoint /workspace/outputs/.../epoch_best_xxx.pt \
        --device npu:0
"""
from __future__ import annotations

import sys
import json
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"[STEP] {desc}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {desc} failed with code {result.returncode}")
        return False
    print(f"[OK] {desc} completed")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="训练配置文件")
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    p.add_argument("--device", default="npu:0", help="NPU 设备")
    p.add_argument("--skip-extract", action="store_true", help="跳过 embedding 提取（如果已存在）")
    p.add_argument("--skip-viz", action="store_true", help="跳过可视化（节省时间）")
    args = p.parse_args()
    
    cfg = load_config(args.config)
    exp_name = cfg.experiment.name
    
    # 确定输出目录
    output_base = Path(f"/workspace/outputs/{exp_name}")
    eval_dir = output_base / "evaluation"
    emb_dir = eval_dir / "embeddings"
    downstream_dir = eval_dir / "downstream"
    cd_dir = eval_dir / "change_detection"
    viz_dir = eval_dir / "visualizations"
    
    emb_file = emb_dir / "patch_embeddings.npz"
    
    print(f"\n{'#'*60}")
    print(f"# 完整评估: {exp_name}")
    print(f"# Checkpoint: {args.checkpoint}")
    print(f"# Device: {args.device}")
    print(f"{'#'*60}")
    
    # Step 1: 提取 embedding
    if not args.skip_extract or not emb_file.exists():
        success = run_cmd(
            f"cd /workspace/xuannv && python scripts/eval/extract_embeddings_v2.py "
            f"--config {args.config} "
            f"--checkpoint {args.checkpoint} "
            f"--output-dir {emb_dir} "
            f"--device {args.device} "
            f"--batch-size 16",
            "提取 Embedding"
        )
        if not success:
            print("[FATAL] Embedding 提取失败，终止评估")
            return
    else:
        print(f"[SKIP] Embedding 已存在: {emb_file}")
    
    # Step 2: KNN 评估
    run_cmd(
        f"cd /workspace/xuannv && python scripts/eval/evaluate_knn_v2.py "
        f"--embedding-file {emb_file} "
        f"--output-dir {downstream_dir} "
        f"--device {args.device} "
        f"--k 5",
        "KNN 下游评估"
    )
    
    # Step 3: MLP 评估
    run_cmd(
        f"cd /workspace/xuannv && python scripts/eval/evaluate_mlp_v2.py "
        f"--embedding-file {emb_file} "
        f"--output-dir {downstream_dir} "
        f"--device {args.device} "
        f"--epochs 50",
        "MLP 下游评估"
    )
    
    # Step 4: 变化检测 AUC
    run_cmd(
        f"cd /workspace/xuannv && python scripts/eval/evaluate_cd_v2.py "
        f"--embedding-file {emb_file} "
        f"--output-dir {cd_dir}",
        "变化检测 AUC 评估"
    )
    
    # Step 5: CD Head 训练
    run_cmd(
        f"cd /workspace/xuannv && python scripts/eval/train_cd_head.py "
        f"--embedding-file {emb_file} "
        f"--output-dir {cd_dir} "
        f"--device {args.device} "
        f"--epochs 30",
        "CD Head 训练"
    )
    
    # Step 6: 可视化
    if not args.skip_viz:
        run_cmd(
            f"cd /workspace/xuannv && python scripts/eval/visualize_cd.py "
            f"--embedding-file {emb_file} "
            f"--output-dir {viz_dir / 'cd_before_after'} "
            f"--n-jobs 8",
            "变化检测可视化"
        )
    
    # Step 7: 生成汇总报告
    print(f"\n{'='*60}")
    print("[STEP] 生成汇总报告")
    print(f"{'='*60}")
    
    summary = {"experiment": exp_name, "checkpoint": args.checkpoint}
    
    # 读取下游结果
    for task in ["worldcover", "dynamic_world", "jrc_water"]:
        knn_path = downstream_dir / f"knn_{task}.json"
        mlp_path = downstream_dir / f"mlp_{task}.json"
        if knn_path.exists():
            with open(knn_path) as f:
                summary[f"knn_{task}"] = json.load(f)
        if mlp_path.exists():
            with open(mlp_path) as f:
                summary[f"mlp_{task}"] = json.load(f)
    
    # 读取 CD 结果
    cd_auc_path = cd_dir / "auc_results.json"
    if cd_auc_path.exists():
        with open(cd_auc_path) as f:
            summary["cd_auc"] = json.load(f)
    
    cd_head_path = cd_dir / "cd_head_results.json"
    if cd_head_path.exists():
        with open(cd_head_path) as f:
            summary["cd_head"] = json.load(f)
    
    with open(eval_dir / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 汇总报告已保存: {eval_dir / 'summary_report.json'}")
    print(f"\n{'#'*60}")
    print(f"# 评估完成: {exp_name}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
