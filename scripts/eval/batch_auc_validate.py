"""Round7 8实验 AUC 批量验证脚本.

对每个实验的 E50 checkpoint 运行 pre-norm 和 l2-norm 两种模式的 AUC 验证.

用法:
    python scripts/eval/batch_auc_validate.py \
        --experiments 1,2,3,4,5,6,7,8 \
        --n-samples 200 \
        --output /workspace/outputs/round7_auc_results.json

输出:
    - JSON: 每个实验的 pre-norm AUC 和 l2-norm AUC
    - 文本表格: 8 实验 AUC 对比
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")


def find_checkpoint(exp_id: int) -> tuple[str, str] | None:
    """查找实验目录、config 和 checkpoint."""
    # 查找目录
    dirs = glob.glob(f"/workspace/outputs/round7_exp{exp_id}_*")
    if not dirs:
        return None
    exp_dir = dirs[0]

    # 查找 config
    configs = glob.glob(f"/workspace/xuannv/configs/round7_8gpu/exp{exp_id}_*.yaml")
    if not configs:
        return None
    config_path = configs[0]

    # 查找 checkpoint（取最新的 epoch_*.pt 或 epoch_best_*.pt）
    checkpoints = (
        glob.glob(f"{exp_dir}/epoch_*.pt") +
        glob.glob(f"{exp_dir}/epoch_best_*.pt")
    )
    if not checkpoints:
        return None
    # 按修改时间排序取最新
    checkpoint_path = max(checkpoints, key=lambda p: Path(p).stat().st_mtime)

    return config_path, checkpoint_path


def run_auc_validation(config_path: str, checkpoint_path: str, mode: str, n_samples: int, device: str, output_path: str) -> dict:
    """运行单个 AUC 验证."""
    cmd = [
        "python", "scripts/eval/quick_eval_checkpoint.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--mode", mode,
        "--n-samples", str(n_samples),
        "--device", device,
        "--output", output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
            cwd="/workspace/xuannv",
        )
        output = result.stdout + result.stderr

        # 解析输出
        auc = None
        separation = None
        for line in output.split("\n"):
            if "AUC" in line and "=" in line:
                try:
                    auc = float(line.split("=")[-1].strip().split()[0])
                except:
                    pass
            if "separation" in line.lower() and "=" in line:
                try:
                    separation = float(line.split("=")[-1].strip().split()[0])
                except:
                    pass

        return {
            "success": result.returncode == 0 and auc is not None,
            "auc": auc,
            "separation": separation,
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "output": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "output": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="/workspace/outputs/round7_auc_results.json")
    parser.add_argument("--skip-l2", action="store_true", help="跳过 l2-norm 验证（节省 NPU）")
    args = parser.parse_args()

    exp_ids = [int(x.strip()) for x in args.experiments.split(",")]
    results = []

    print("=" * 80)
    print(" Round7 AUC 批量验证")
    print("=" * 80)

    for exp_id in exp_ids:
        found = find_checkpoint(exp_id)
        if not found:
            print(f"\n⚠️ exp{exp_id}: config/checkpoint 未找到，跳过")
            results.append({"exp_id": exp_id, "error": "not found"})
            continue

        config_path, checkpoint_path = found
        print(f"\n{'='*60}")
        print(f"exp{exp_id}: {Path(checkpoint_path).name}")
        print(f"{'='*60}")

        result = {
            "exp_id": exp_id,
            "config": config_path,
            "checkpoint": checkpoint_path,
        }

        # Pre-norm 验证
        print(f"  运行 pre-norm AUC 验证...")
        output_dir = Path(args.output).parent
        pre_norm_output = str(output_dir / f"exp{exp_id}_pre_norm_auc.json")
        pre_norm_result = run_auc_validation(
            config_path, checkpoint_path, "pre-norm",
            args.n_samples, args.device, pre_norm_output,
        )
        result["pre_norm"] = pre_norm_result
        if pre_norm_result["success"]:
            print(f"  ✅ Pre-norm AUC = {pre_norm_result['auc']:.4f}")
        else:
            print(f"  ❌ Pre-norm 失败")
            print(f"     {pre_norm_result.get('error', 'unknown error')}")

        # L2-norm 验证
        if not args.skip_l2:
            print(f"  运行 l2-norm AUC 验证...")
            output_dir = Path(args.output).parent
            l2_output = str(output_dir / f"exp{exp_id}_l2_norm_auc.json")
            l2_result = run_auc_validation(
                config_path, checkpoint_path, "l2-norm",
                args.n_samples, args.device, l2_output,
            )
            result["l2_norm"] = l2_result
            if l2_result["success"]:
                print(f"  ✅ L2-norm AUC = {l2_result['auc']:.4f}")
            else:
                print(f"  ❌ L2-norm 失败")
                print(f"     {l2_result.get('error', 'unknown error')}")

        results.append(result)

    # 打印对比表
    print("\n" + "=" * 80)
    print(" AUC 结果对比表")
    print("=" * 80)
    print(f"{'Exp':<6} {'Pre-norm AUC':<14} {'L2-norm AUC':<14} {'Delta':<10} {'Separation':<12}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            continue
        pre_auc = r["pre_norm"].get("auc", None)
        l2_auc = r["l2_norm"].get("auc", None) if "l2_norm" in r else None
        sep = r["pre_norm"].get("separation", None)

        pre_str = f"{pre_auc:.4f}" if pre_auc else "N/A"
        l2_str = f"{l2_auc:.4f}" if l2_auc else "N/A"
        delta = f"{pre_auc - l2_auc:+.4f}" if (pre_auc and l2_auc) else "N/A"
        sep_str = f"{sep:.4f}" if sep else "N/A"

        print(f"exp{r['exp_id']:<2} {pre_str:<14} {l2_str:<14} {delta:<10} {sep_str:<12}")

    print("=" * 80)

    # 保存 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 AUC 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
