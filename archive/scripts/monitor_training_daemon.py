#!/usr/bin/env python3
"""训练监控守护进程 — 独立于对话持续运行.

每 5 分钟抓取 4 个实验的 tmux 日志，提取关键指标保存到监控日志.
"""
import subprocess
import time
import re
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    ("v13_r2_expD", "r2_expD_skip_l2"),
    ("v13_r2_expE", "r2_expE_skip_l2_prenorm"),
    ("v13_r2_expF", "r2_expF_skip_l2_ortho"),
    ("v13_r2_expG", "r2_expG_skip_l2_all"),
]

LOG_FILE = Path("/workspace/outputs/v13_fast_logs/monitor_history.log")
SUMMARY_FILE = Path("/workspace/outputs/v13_fast_logs/latest_summary.txt")


def capture_tmux(session: str) -> str:
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout
    except Exception as e:
        return f"ERROR: {e}"


def extract_metrics(text: str) -> dict:
    """从 tmux 输出中提取关键指标."""
    metrics = {}
    
    # Epoch 级别汇总
    epoch_match = re.search(
        r'Epoch (\d+).*?active=(\d+)/(\d+).*?std_mean=([\d.]+).*?recon=([\d.]+).*?l2unif=(-?[\d.]+)',
        text
    )
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
        metrics['active_dims'] = int(epoch_match.group(2))
        metrics['total_dims'] = int(epoch_match.group(3))
        metrics['std_mean'] = float(epoch_match.group(4))
        metrics['recon'] = float(epoch_match.group(5))
        metrics['l2unif'] = float(epoch_match.group(6))
    
    # Step 级别最新数据
    step_match = re.search(
        r'\[Step (\d+)\].*?recon=([\d.]+).*?var=([\d.]+).*?l2unif=(-?[\d.]+).*?active_dims=(\d+)',
        text
    )
    if step_match:
        metrics['latest_step'] = int(step_match.group(1))
        metrics['step_recon'] = float(step_match.group(2))
        metrics['step_var'] = float(step_match.group(3))
        metrics['step_l2unif'] = float(step_match.group(4))
        metrics['step_active'] = int(step_match.group(5))
    
    return metrics


def check_running(session: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True, text=True, timeout=5
        )
        return session in result.stdout
    except Exception:
        return False


def monitor_cycle():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n{'='*60}", f"[{now}] Monitoring Cycle", f"{'='*60}"]
    summary_lines = [f"[{now}] Latest Summary", "=" * 50]
    
    for session, name in EXPERIMENTS:
        lines.append(f"\n--- {session} ({name}) ---")
        
        if not check_running(session):
            lines.append("  STATUS: NOT RUNNING")
            summary_lines.append(f"{name}: NOT RUNNING")
            continue
        
        text = capture_tmux(session)
        metrics = extract_metrics(text)
        
        if metrics:
            epoch = metrics.get('epoch', '?')
            active = metrics.get('active_dims', '?')
            total = metrics.get('total_dims', 128)
            recon = metrics.get('recon', metrics.get('step_recon', '?'))
            l2unif = metrics.get('l2unif', metrics.get('step_l2unif', '?'))
            std_mean = metrics.get('std_mean', '?')
            
            lines.append(f"  Epoch: {epoch} | Active: {active}/{total} | Recon: {recon} | L2Unif: {l2unif} | StdMean: {std_mean}")
            
            # 判断状态
            if isinstance(active, int):
                if active >= 80:
                    status = "🌟 EXCELLENT"
                elif active >= 50:
                    status = "✅ GOOD"
                elif active >= 20:
                    status = "⚠️ WARNING"
                else:
                    status = "❌ COLLAPSED"
            else:
                status = "⏳ STARTING"
            
            summary_lines.append(f"{name}: Epoch={epoch} Active={active}/{total} Recon={recon:.4f} L2Unif={l2unif:.4f} [{status}]")
        else:
            lines.append("  STATUS: WARMING UP (no metrics yet)")
            summary_lines.append(f"{name}: WARMING UP")
    
    # 写入日志
    with open(LOG_FILE, "a") as f:
        f.write("\n".join(lines) + "\n")
    
    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    
    print(f"[{now}] Monitor cycle complete. Log: {LOG_FILE}")


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Monitor daemon started. Logging to {LOG_FILE}")
    print("Press Ctrl+C to stop.")
    
    while True:
        try:
            monitor_cycle()
            time.sleep(300)  # 5分钟检查一次
        except KeyboardInterrupt:
            print("\nMonitor daemon stopped.")
            break
        except Exception as e:
            print(f"Error in monitor cycle: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
