#!/usr/bin/env python3
"""批量 AUC 评估 — 一次性评估多个 checkpoint"""
import argparse, json, subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    for epoch in args.epochs:
        ckpt = exp_dir / f"epoch_{epoch}.pt"
        if not ckpt.exists():
            print(f"Skip epoch {epoch}: {ckpt} not found")
            continue
        outdir = exp_dir / f"eval_epoch_{epoch}"
        outdir.mkdir(exist_ok=True)
        if (outdir / "auc_results.json").exists():
            print(f"Skip epoch {epoch}: already evaluated")
            continue
        print(f"Evaluating epoch {epoch}...")
        subprocess.run([
            "python", "scripts/eval/auc_eval.py",
            "--config", args.config,
            "--checkpoint", str(ckpt),
            "--device", args.device,
            "--output", str(outdir / "auc_results.json")
        ])
        print()

if __name__ == "__main__":
    main()
