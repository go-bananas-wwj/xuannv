#!/usr/bin/env python3
"""V7 Minimal 训练监控脚本 — 后台运行，定期汇报指标."""
from __future__ import annotations

import time
import re
import sys
from pathlib import Path

LOG_PATH = Path("/workspace/outputs/aef_qwen_v7_minimal/train.log")
SUMMARY_PATH = Path("/workspace/outputs/aef_qwen_v7_minimal/monitor_summary.txt")

# 关键指标的正则
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
    r"Loss:\s+([\d.]+)\s+\|\s+"
    r"Recon:\s+([\d.]+)\s+\|\s+"
    r"VICReg:\s+([\d.]+)\s+\(inv=([\d.]+)\s+var=([\d.]+)\s+cov=([\d.]+)\)\s+\|\s+"
    r"KoLeo:\s+([\d.]+)\s+\|\s+"
    r"PreUnif:\s+([\d.-]+)\s+\|\s+"
    r"EncUnif:\s+([\d.-]+)\s+\|\s+"
    r"Consist:\s+([\d.-]+)\s+\|\s+"
    r"Cls:\s+([\d.]+)\s+\|\s+"
    r"Temporal:\s+([\d.]+)\s+\|\s+"
    r"LR:\s+([\de.-]+)\s+\|\s+"
    r"Time:\s+([\d.]+)s"
)

def parse_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    text = LOG_PATH.read_text(encoding='utf-8', errors='replace')
    results = []
    for m in EPOCH_RE.finditer(text):
        results.append({
            "epoch": int(m.group(1)),
            "total": float(m.group(3)),
            "recon": float(m.group(4)),
            "vicreg": float(m.group(5)),
            "vicreg_inv": float(m.group(6)),
            "vicreg_var": float(m.group(7)),
            "vicreg_cov": float(m.group(8)),
            "koleo": float(m.group(9)),
            "pre_unif": float(m.group(10)),
            "enc_unif": float(m.group(11)),
            "consist": float(m.group(12)),
            "cls": float(m.group(13)),
            "temporal": float(m.group(14)),
            "lr": m.group(15),
            "time": float(m.group(16)),
        })
    return results

def format_summary(epochs: list[dict]) -> str:
    if not epochs:
        return "No epoch data found yet."
    lines = []
    lines.append("=" * 100)
    lines.append(f"V7 Minimal Training Monitor — {len(epochs)} epochs recorded")
    lines.append("=" * 100)
    lines.append(f"{'Epoch':>5} | {'Recon':>7} | {'VICReg':>7} | {'V_var':>7} | {'V_cov':>7} | {'KoLeo':>7} | {'PreUnif':>8} | {'EncUnif':>8} | {'LR':>10} | {'Time':>6}")
    lines.append("-" * 100)
    for e in epochs:
        lines.append(
            f"{e['epoch']:>5} | {e['recon']:>7.4f} | {e['vicreg']:>7.4f} | "
            f"{e['vicreg_var']:>7.4f} | {e['vicreg_cov']:>7.4f} | {e['koleo']:>7.4f} | "
            f"{e['pre_unif']:>8.4f} | {e['enc_unif']:>8.4f} | {e['lr']:>10} | {e['time']:>6.1f}s"
        )
    lines.append("-" * 100)
    
    # 趋势分析
    if len(epochs) >= 2:
        last = epochs[-1]
        prev = epochs[-2]
        lines.append(f"\nLatest Trend (Epoch {prev['epoch']} → {last['epoch']}):")
        lines.append(f"  VICReg:  {prev['vicreg']:.4f} → {last['vicreg']:.4f} ({'↓' if last['vicreg'] < prev['vicreg'] else '↑'})")
        lines.append(f"  KoLeo:   {prev['koleo']:.4f} → {last['koleo']:.4f} ({'↓' if last['koleo'] < prev['koleo'] else '↑'})")
        lines.append(f"  PreUnif: {prev['pre_unif']:.4f} → {last['pre_unif']:.4f} ({'↓' if last['pre_unif'] < prev['pre_unif'] else '↑'})")
        lines.append(f"  Recon:   {prev['recon']:.4f} → {last['recon']:.4f} ({'↓' if last['recon'] < prev['recon'] else '↑'})")
    
    # 目标评估
    last = epochs[-1]
    lines.append(f"\nTarget Assessment (Epoch {last['epoch']}):")
    lines.append(f"  VICReg < 1.0:    {'✅ PASS' if last['vicreg'] < 1.0 else '❌ FAIL'} ({last['vicreg']:.4f})")
    lines.append(f"  KoLeo < 1.0:     {'✅ PASS' if last['koleo'] < 1.0 else '⚠️  PENDING'} ({last['koleo']:.4f})")
    lines.append(f"  PreUnif < -1.0:  {'✅ PASS' if last['pre_unif'] < -1.0 else '⚠️  PENDING'} ({last['pre_unif']:.4f})")
    lines.append(f"  Recon < 0.35:    {'✅ PASS' if last['recon'] < 0.35 else '⚠️  PENDING'} ({last['recon']:.4f})")
    
    lines.append(f"\nLast updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)

def main():
    print("[Monitor] Starting V7 training monitor...")
    print(f"[Monitor] Watching: {LOG_PATH}")
    print(f"[Monitor] Summary:  {SUMMARY_PATH}")
    print("[Monitor] Press Ctrl+C to stop")
    
    last_count = 0
    while True:
        epochs = parse_log()
        if len(epochs) > last_count:
            summary = format_summary(epochs)
            SUMMARY_PATH.write_text(summary)
            print(f"\n[Monitor] {len(epochs)} epochs recorded. Summary updated.")
            # 打印最新 epoch
            e = epochs[-1]
            print(f"  Epoch {e['epoch']}: VICReg={e['vicreg']:.4f} KoLeo={e['koleo']:.4f} PreUnif={e['pre_unif']:.4f} Recon={e['recon']:.4f}")
            last_count = len(epochs)
        time.sleep(30)

if __name__ == "__main__":
    main()
