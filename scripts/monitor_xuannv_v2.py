#!/usr/bin/env python3
"""玄女V2 训练后台监控脚本 — 每10分钟检查一次关键指标."""
from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")

EXPERIMENTS = [
    ("ExpA_skipL2", "/workspace/outputs/xuannv_v2_expA_skipL2/train.log"),
    ("ExpB_noSkipL2", "/workspace/outputs/xuannv_v2_expB_noSkipL2/train.log"),
]

MONITOR_LOG = "/workspace/outputs/xuannv_v2_monitor.log"


def parse_epoch_line(line: str) -> dict | None:
    """从 epoch 汇总行提取关键指标."""
    import re
    # 匹配: Epoch 001/050 | total=0.3886 recon=1.1580 consist=0.3861 ... active=2/128 std_mean=0.0252
    try:
        epoch_match = re.search(r'Epoch\s+(\d+)', line)
        if not epoch_match:
            return None
        epoch = int(epoch_match.group(1))

        def get_val(key: str):
            m = re.search(rf'{key}=([\d.]+)', line)
            return float(m.group(1)) if m else None

        def get_frac(key: str):
            m = re.search(rf'{key}=(\d+)/(\d+)', line)
            return (int(m.group(1)), int(m.group(2))) if m else (None, None)

        total = get_val('total')
        recon = get_val('recon')
        consist = get_val('consist')
        l2unif = get_val('l2unif')
        active_num, active_denom = get_frac('active')
        std_mean = get_val('std_mean')
        lr = get_val('lr')

        return {
            'epoch': epoch,
            'total': total,
            'recon': recon,
            'consist': consist,
            'l2unif': l2unif,
            'active': active_num,
            'active_denom': active_denom,
            'std_mean': std_mean,
            'lr': lr,
        }
    except Exception:
        return None


def check_experiment(name: str, log_path: str) -> str:
    """检查单个实验的最新状态."""
    path = Path(log_path)
    if not path.exists():
        return f"[{name}] 日志文件不存在"

    lines = path.read_text().splitlines()
    if not lines:
        return f"[{name}] 日志为空"

    # 找最新的 epoch 汇总行
    epoch_lines = [l for l in lines if 'Epoch' in l and 'active=' in l]
    step_lines = [l for l in lines if '[Step' in l]

    if not epoch_lines and not step_lines:
        return f"[{name}] 尚未开始训练"

    result = f"[{name}]"

    if epoch_lines:
        latest = parse_epoch_line(epoch_lines[-1])
        if latest:
            result += f"\n  Epoch {latest['epoch']:02d}:"
            result += f" total={latest['total']:.4f}" if latest['total'] else ""
            result += f" recon={latest['recon']:.4f}" if latest['recon'] else ""
            result += f" consist={latest['consist']:.4f}" if latest['consist'] else ""
            result += f" l2unif={latest['l2unif']:.4f}" if latest['l2unif'] else ""
            result += f" active={latest['active']}/{latest['active_denom']}" if latest['active'] is not None else ""
            result += f" std_mean={latest['std_mean']:.4f}" if latest['std_mean'] else ""

            # 预警
            alerts = []
            if latest['active'] is not None and latest['active_denom'] is not None:
                ratio = latest['active'] / latest['active_denom']
                if ratio < 0.2:
                    alerts.append(f"🚨 严重坍缩! active={latest['active']}/{latest['active_denom']}")
                elif ratio < 0.4:
                    alerts.append(f"⚠️ 轻度坍缩 active={latest['active']}/{latest['active_denom']}")
            if latest['std_mean'] is not None and latest['std_mean'] < 0.01:
                alerts.append(f"🚨 std_mean={latest['std_mean']:.4f} 过低")
            if latest['l2unif'] is not None and latest['l2unif'] > 0.95:
                alerts.append(f"⚠️ l2unif={latest['l2unif']:.4f} 接近坍缩")
            if alerts:
                result += "\n  " + " | ".join(alerts)

    if step_lines:
        latest_step = step_lines[-1]
        result += f"\n  Latest step: {latest_step.strip()[:120]}"

    return result


def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 玄女V2 监控启动")
    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report = f"\n{'='*60}\n[{timestamp}] 监控报告\n{'='*60}\n"

        for name, log_path in EXPERIMENTS:
            report += check_experiment(name, log_path) + "\n"

        # 写入日志
        with open(MONITOR_LOG, 'a') as f:
            f.write(report)

        print(report)

        # 每10分钟检查一次
        time.sleep(600)


if __name__ == "__main__":
    main()
