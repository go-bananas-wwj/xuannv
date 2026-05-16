#!/usr/bin/env python3
"""等待所有评估完成并生成报告."""
import os, sys, time, subprocess

sys.path.insert(0, "/workspace/xuannv")

def check_status():
    status = []
    all_done = True
    for i in range(1, 9):
        d = f'/workspace/outputs/round8_single_exp{i}'
        has_pre = os.path.exists(f'{d}/precomputed_embeddings.pt')
        has_eval = os.path.exists(f'{d}/eval_results.json')
        status.append(f'exp{i}:pre={has_pre},eval={has_eval}')
        if not has_eval:
            all_done = False
    return all_done, status

if __name__ == "__main__":
    print("等待评估完成...")
    for iteration in range(120):  # 最多 120 分钟
        all_done, status = check_status()
        print(time.strftime('%H:%M:%S'), ' '.join(status))
        if all_done:
            print("ALL_DONE")
            # 生成报告
            result = subprocess.run([
                sys.executable, "scripts/eval/generate_comparison_report.py",
                "--input-dir", "/workspace/outputs",
                "--pattern", "round8_single_exp*",
                "--output", "/workspace/outputs/round8_comparison.md",
            ], cwd="/workspace/xuannv")
            print(f"报告生成: {result.returncode}")
            break
        time.sleep(60)
    else:
        print("TIMEOUT")
