"""实时监控 Round7 8实验到 E50 的进度.

用法:
    python scripts/monitor_e50_progress.py

输出: 每 5 分钟刷新一次，显示各实验当前 epoch、ETA、关键指标
"""
from __future__ import annotations

import glob
import os
import re
import time
from datetime import datetime, timedelta


def parse_log_for_epoch(log_path: str) -> dict | None:
    """从日志文件解析最新 epoch 和关键指标."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except:
        return None

    # 从后往前找最新的 epoch 信息
    for line in reversed(lines):
        # 匹配 Epoch 行: "Epoch 40/50"
        epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)', line)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))

            # 提取关键指标
            recon = re.search(r'recon=([0-9.]+)', line)
            unif = re.search(r'(?:raw_unif|l2unif|pre_unif|uniform)=(-?[0-9.]+)', line)
            active = re.search(r'active_dims=(\d+)', line)

            return {
                'epoch': current,
                'total': total,
                'recon': float(recon.group(1)) if recon else None,
                'unif': float(unif.group(1)) if unif else None,
                'active': int(active.group(1)) if active else None,
            }
    return None


def get_epoch_from_tracker(exp_id: int) -> int | None:
    """从 round7_epoch_tracker.log 获取指定实验的 epoch."""
    tracker_path = "/workspace/outputs/round7_epoch_tracker.log"
    try:
        with open(tracker_path, 'r') as f:
            lines = f.readlines()
    except:
        return None

    # 从后往前找
    for line in reversed(lines):
        if f'exp{exp_id}_' in line:
            match = re.search(r'E(\d+)', line)
            if match:
                return int(match.group(1))
    return None


def get_experiment_status() -> list[dict]:
    """获取所有实验状态."""
    experiments = []
    output_dir = "/workspace/outputs"

    for i in range(1, 9):
        # 找到实验目录
        dirs = glob.glob(f"{output_dir}/round7_exp{i}_*")
        if not dirs:
            continue
        exp_dir = dirs[0]
        exp_name = os.path.basename(exp_dir)

        # 找到日志文件
        log_files = glob.glob(f"{exp_dir}/*.log")
        if not log_files:
            continue
        log_path = log_files[0]

        # 解析
        info = parse_log_for_epoch(log_path)
        if info:
            experiments.append({
                'id': i,
                'name': exp_name.replace('round7_exp', 'exp'),
                'epoch': info['epoch'],
                'total': info['total'],
                'recon': info['recon'],
                'unif': info['unif'],
                'active': info['active'],
            })
        else:
            # 回退到 tracker 日志
            epoch = get_epoch_from_tracker(i)
            if epoch:
                experiments.append({
                    'id': i,
                    'name': exp_name.replace('round7_exp', 'exp'),
                    'epoch': epoch,
                    'total': 50,
                    'recon': None,
                    'unif': None,
                    'active': None,
                })

    return experiments


def format_eta(remaining_epochs: int, avg_time_per_epoch_min: float = 10.0) -> str:
    """格式化 ETA."""
    total_min = remaining_epochs * avg_time_per_epoch_min
    eta = datetime.now() + timedelta(minutes=total_min)
    return f"~{total_min:.0f}min ({eta.strftime('%H:%M')})"


def print_status(experiments: list[dict]):
    """打印状态表."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*90}")
    print(f" Round7 E50 进度监控 | {now}")
    print(f"{'='*90}")
    print(f"{'Exp':<8} {'Name':<22} {'Epoch':<10} {'%':<8} {'ETA':<18} {'Recon':<10} {'Unif':<10} {'Active'}")
    print(f"{'-'*90}")

    all_done = True
    for exp in experiments:
        pct = exp['epoch'] / exp['total'] * 100
        remaining = exp['total'] - exp['epoch']
        eta_str = "✅ DONE" if remaining <= 0 else format_eta(remaining)
        if remaining > 0:
            all_done = False

        recon_str = f"{exp['recon']:.4f}" if exp['recon'] else "N/A"
        unif_str = f"{exp['unif']:.2f}" if exp['unif'] else "N/A"
        active_str = f"{exp['active']}/128" if exp['active'] else "N/A"

        print(f"exp{exp['id']:<2} {exp['name']:<22} {exp['epoch']}/{exp['total']:<4} {pct:<7.1f} {eta_str:<18} {recon_str:<10} {unif_str:<10} {active_str}")

    print(f"{'='*90}")

    if all_done:
        print("\n🎉 所有实验已完成 E50！准备启动 AUC 验证。")
        return True
    else:
        # 找出最慢的实验
        min_epoch = min(e['epoch'] for e in experiments)
        slowest = [e for e in experiments if e['epoch'] == min_epoch]
        eta_all = format_eta(50 - min_epoch)
        print(f"\n最慢: {', '.join([e['name'] for e in slowest])} @ E{min_epoch}")
        print(f"预计全部完成: {eta_all}")
        print(f"下次刷新: 5 分钟后...")
        return False


def main():
    import sys
    # 支持一次性运行或循环监控
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        experiments = get_experiment_status()
        print_status(experiments)
        return

    print("开始监控 Round7 E50 进度（每 5 分钟刷新，Ctrl+C 退出）...")
    try:
        while True:
            experiments = get_experiment_status()
            done = print_status(experiments)
            if done:
                break
            time.sleep(300)  # 5 分钟
    except KeyboardInterrupt:
        print("\n监控已停止。")


if __name__ == "__main__":
    main()
