#!/usr/bin/env python3
"""标准实验流水线: 训练 → 提取 embedding → KNN 评估 → 可视化 → 汇总报告.

用法:
    cd /workspace/xuannv
    # 单卡快速验证
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval/run_full_pipeline.py --config configs/aef_high_kappa.yaml --gpus 1

    # 4 卡训练 + 评估
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python scripts/eval/run_full_pipeline.py --config configs/aef_high_kappa.yaml --gpus 4

    # 仅评估（跳过训练）
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/eval/run_full_pipeline.py --config configs/aef_high_kappa.yaml --skip-train
"""
import sys, os, argparse, subprocess, json, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")

def run(cmd, desc, timeout=None):
    """运行 shell 命令，打印输出."""
    print(f"\n{'='*60}")
    print(f"  [{desc}]")
    print(f"  Command: {cmd}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True, timeout=timeout)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] {desc} failed (exit={result.returncode}) after {dt:.0f}s")
        return False
    print(f"  [OK] {desc} completed in {dt:.0f}s")
    return True

def extract_exp_name(config_path):
    """从 config 文件名提取实验名."""
    return Path(config_path).stem

def get_output_dir(config_path):
    """从 config 读取 output_dir."""
    from src.config import load_config
    cfg = load_config(config_path)
    return Path(cfg.experiment.output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="训练配置 YAML")
    parser.add_argument("--gpus", type=int, default=1, help="训练用的 GPU/NPU 数")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，只做评估")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--skip-viz", action="store_true", help="跳过可视化")
    parser.add_argument("--device", default="npu:0", help="评估用的设备")
    args = parser.parse_args()

    exp_name = extract_exp_name(args.config)
    output_dir = get_output_dir(args.config)
    log_dir = output_dir / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "pipeline_report.json"

    pipeline_log = []
    overall_t0 = time.time()

    # ── Phase 1: 训练 ────────────────────────────────
    if not args.skip_train:
        train_log = log_dir / "train.log"
        cmd = (
            f"cd /workspace/xuannv && "
            f"torchrun --nproc_per_node={args.gpus} "
            f"scripts/train/train_unified.py --config {args.config} "
            f"> {train_log} 2>&1"
        )
        ok = run(cmd, f"Training {exp_name} ({args.gpus} GPUs)", timeout=7200)
        pipeline_log.append({"phase": "train", "status": "ok" if ok else "failed", "log": str(train_log)})
        if not ok:
            print("[PIPELINE] Training failed, aborting.")
            sys.exit(1)
    else:
        print(f"[PIPELINE] Skipping training (--skip-train)")
        pipeline_log.append({"phase": "train", "status": "skipped"})

    # ── Phase 2: 提取 Embedding ───────────────────────
    if not args.skip_eval:
        extract_log = log_dir / "extract_embeddings.log"
        cmd = (
            f"cd /workspace/xuannv && "
            f"ASCEND_RT_VISIBLE_DEVICES={args.device.split(':')[-1]} "
            f"/root/miniconda3/envs/xuannv/bin/python scripts/eval/extract_embeddings_for_knn.py "
            f"--experiment {exp_name} --device {args.device} "
            f"> {extract_log} 2>&1"
        )
        ok = run(cmd, f"Extracting embeddings for {exp_name}", timeout=1800)
        pipeline_log.append({"phase": "extract", "status": "ok" if ok else "failed", "log": str(extract_log)})
        if not ok:
            print("[PIPELINE] Embedding extraction failed, aborting.")
            sys.exit(1)

        # ── Phase 3: KNN 评估 ───────────────────────────
        knn_log = log_dir / "knn_eval.log"
        cmd = (
            f"cd /workspace/xuannv && "
            f"ASCEND_RT_VISIBLE_DEVICES={args.device.split(':')[-1]} "
            f"/root/miniconda3/envs/xuannv/bin/python scripts/eval/run_knn_npu_fast.py "
            f"--experiment {exp_name} --device {args.device} "
            f"> {knn_log} 2>&1"
        )
        ok = run(cmd, f"KNN evaluation for {exp_name}", timeout=1800)
        pipeline_log.append({"phase": "knn", "status": "ok" if ok else "failed", "log": str(knn_log)})

        # 读取 KNN 结果
        results_path = output_dir / "downstream_knn" / "results.json"
        knn_results = {}
        if results_path.exists():
            with open(results_path) as f:
                knn_results = json.load(f)
    else:
        print(f"[PIPELINE] Skipping evaluation (--skip-eval)")
        pipeline_log.append({"phase": "extract", "status": "skipped"})
        pipeline_log.append({"phase": "knn", "status": "skipped"})
        knn_results = {}

    # ── Phase 4: 可视化 ───────────────────────────────
    if not args.skip_viz and not args.skip_eval:
        viz_log = log_dir / "viz.log"
        # 为单个实验生成可视化（复用已有的生成脚本）
        cmd = (
            f"cd /workspace/xuannv && "
            f"/root/miniconda3/envs/xuannv/bin/python scripts/eval/generate_comparison_viz.py "
            f"> {viz_log} 2>&1"
        )
        ok = run(cmd, "Generating comparison visualizations", timeout=600)
        pipeline_log.append({"phase": "viz", "status": "ok" if ok else "failed", "log": str(viz_log)})
    else:
        pipeline_log.append({"phase": "viz", "status": "skipped"})

    # ── 汇总报告 ─────────────────────────────────────
    overall_dt = time.time() - overall_t0
    report = {
        "experiment": exp_name,
        "config": args.config,
        "output_dir": str(output_dir),
        "gpus": args.gpus,
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": int(overall_dt),
        "phases": pipeline_log,
        "knn_results": knn_results,
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  [PIPELINE COMPLETE]")
    print(f"  Experiment: {exp_name}")
    print(f"  Total time: {overall_dt/60:.1f} min")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")

    # 打印 KNN 结果摘要
    if knn_results:
        print(f"\n  Downstream KNN Results (K=5):")
        for task, vals in knn_results.items():
            k5 = vals.get("k5", {})
            acc = k5.get("accuracy", 0) * 100
            miou = k5.get("mean_iou", 0) * 100
            print(f"    {task}: Acc={acc:.2f}%  mIoU={miou:.2f}%")

if __name__ == "__main__":
    main()
