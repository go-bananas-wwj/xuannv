#!/usr/bin/env python3
"""
monitor_training_health.py -- Training health monitor
Checks training logs for abnormal metrics and alerts
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

def check_log(log_file: Path):
    """Check training log health"""
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return 1
    
    content = log_file.read_text()
    
    # Extract latest epoch metrics
    steps = re.findall(
        r"\[Step\s+(\d+)/(\d+)\]\s+total=([\-\d.]+)\s+recon=([\-\d.]+)\s+cls=([\-\d.]+)\s+var=([\-\d.]+)\s+cov=([\-\d.]+)\s+l2unif=([\-\d.]+)\s+erank=([\-\d.]+)",
        content
    )
    
    if not steps:
        print("[WARN] No training step data found")
        return 1
    
    latest = steps[-1]
    step, total_steps, total, recon, cls, var, cov, l2unif, erank = latest
    erank = float(erank)
    recon = float(recon)
    total = float(total)
    
    issues = []
    
    # Check eRank
    if erank < 8.0:
        issues.append(f"[CRITICAL] eRank={erank} < 8.0, dimensional collapse likely!")
    elif erank < 10.0:
        issues.append(f"[WARNING] eRank={erank} < 10.0, monitor closely")
    
    # Check NaN/Inf in numeric contexts only
    if re.search(r"=nan\b|=inf\b|loss=nan|loss=inf", content.lower()):
        issues.append("[CRITICAL] NaN/Inf detected in loss values!")
    
    # Check recon loss
    if recon > 2.0:
        issues.append(f"[WARNING] recon={recon} > 2.0, reconstruction loss high")
    
    # Check total loss
    if total > 0:
        issues.append(f"[WARNING] total={total} > 0, total loss is positive")
    
    print(f"=== Training Health Check ({log_file.name}) ===")
    print(f"Latest step: {step}/{total_steps}")
    print(f"eRank: {erank}")
    print(f"recon: {recon}")
    print(f"total: {total}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        return 1
    else:
        print("\nTraining healthy")
        return 0

if __name__ == "__main__":
    log_dir = Path("/workspace/outputs/exp_v14_anti_collapse_0605")
    if not log_dir.exists():
        print(f"[ERROR] Output directory not found: {log_dir}")
        sys.exit(1)
    
    # Find latest log file
    logs = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("[ERROR] No training logs found")
        sys.exit(1)
    
    sys.exit(check_log(logs[-1]))
