#!/usr/bin/env xuannv/bin/python3
"""后台等待所有评估完成并自动生成报告."""
import os, sys, time, subprocess

sys.path.insert(0, "/workspace/xuannv")

def check():
    for i in range(1, 9):
        if not os.path.exists(f'/workspace/outputs/round8_single_exp{i}/eval_results.json'):
            return False
    return True

if __name__ == "__main__":
    print("Waiting for all evaluations to complete...")
    for iteration in range(120):
        if check():
            print("All evaluations completed! Generating report...")
            result = subprocess.run([
                sys.executable, "scripts/eval/generate_comparison_report.py",
                "--input-dir", "/workspace/outputs",
                "--pattern", "round8_single_exp*",
                "--output", "/workspace/outputs/round8_comparison.md",
            ], cwd="/workspace/xuannv", capture_output=True, text=True)
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
            break
        time.sleep(60)
    else:
        print("Timeout waiting for evaluations")
