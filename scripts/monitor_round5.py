#!/usr/bin/env python3
"""Round 5 训练监控脚本 — 追踪6个实验的关键指标."""
import json
import re
import time
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    "round5_kappa_temporal_mild",
    "round5_baseline_40ep",
    "round5_consist_mild",
    "round5_no_consist",
    "round5_kappa_baseline",
    "round5_temporal_plus_recon",
]

OUTPUT_BASE = Path("/workspace/outputs/round5")
REPORT_PATH = OUTPUT_BASE / "monitor_report.md"
HISTORY_PATH = OUTPUT_BASE / "monitor_history.json"

# 正则提取关键指标
RECON_RE = re.compile(r"recon=([\d.]+)")
CONSIST_RE = re.compile(r"consist=([\d.]+)")
TEMP_RE = re.compile(r"temp=([\d.]+)")
UNIF_RE = re.compile(r"l2unif=([\d.]+)")
VAR_RE = re.compile(r"var=([\d.]+)")
COV_RE = re.compile(r"cov=([\d.]+)")
DECORR_RE = re.compile(r"decorr=([\d.]+)")
ORTH_RE = re.compile(r"orth=([\d.]+)")
STEP_RE = re.compile(r"\[Step (\d+)\]")


def parse_log(log_path: Path):
    """解析训练日志，返回最新 step 的指标."""
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    # 找最后一个包含 [Step N] 的行
    for line in reversed(lines):
        if "[Step" in line:
            m = STEP_RE.search(line)
            step = int(m.group(1)) if m else -1
            recon = float(RECON_RE.search(line).group(1)) if RECON_RE.search(line) else None
            consist = float(CONSIST_RE.search(line).group(1)) if CONSIST_RE.search(line) else None
            temp = float(TEMP_RE.search(line).group(1)) if TEMP_RE.search(line) else None
            unif = float(UNIF_RE.search(line).group(1)) if UNIF_RE.search(line) else None
            var = float(VAR_RE.search(line).group(1)) if VAR_RE.search(line) else None
            cov = float(COV_RE.search(line).group(1)) if COV_RE.search(line) else None
            decorr = float(DECORR_RE.search(line).group(1)) if DECORR_RE.search(line) else None
            orth = float(ORTH_RE.search(line).group(1)) if ORTH_RE.search(line) else None
            return {
                "step": step, "recon": recon, "consist": consist, "temp": temp,
                "unif": unif, "var": var, "cov": cov, "decorr": decorr, "orth": orth,
                "line": line.strip()[:200],
            }
    return None


def check_alerts(metrics: dict, exp_name: str) -> list:
    """检查异常指标."""
    alerts = []
    if metrics is None:
        return alerts
    if metrics.get("recon") is not None and metrics["recon"] > 0.5:
        alerts.append(f"recon={metrics['recon']:.3f} > 0.5")
    if metrics.get("unif") is not None and metrics["unif"] < -0.5:
        # l2unif越小越好（负值），但这里原始值可能是正数
        # V12日志中l2unif约0.4-0.7是正常的
        pass
    if metrics.get("var") is not None and metrics["var"] > 0.5:
        alerts.append(f"var={metrics['var']:.3f} > 0.5")
    if metrics.get("decorr") is not None and metrics["decorr"] > 20:
        alerts.append(f"decorr={metrics['decorr']:.1f} > 20")
    return alerts


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    history = {}
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
        except Exception:
            history = {}

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [f"# Round 5 训练监控报告", f"", f"更新时间: {now}", f""]
        report_lines.append("| 实验 | Step | Recon | Consist | Temp | Unif | Var | 异常 |")
        report_lines.append("|------|------|-------|---------|------|------|-----|------|")

        for exp in EXPERIMENTS:
            log_path = OUTPUT_BASE / exp / "train.log"
            metrics = parse_log(log_path)
            alerts = check_alerts(metrics, exp)

            if metrics:
                r = f"{metrics['recon']:.3f}" if metrics['recon'] is not None else '-'
                c = f"{metrics['consist']:.3f}" if metrics['consist'] is not None else '-'
                t = f"{metrics['temp']:.3f}" if metrics['temp'] is not None else '-'
                u = f"{metrics['unif']:.3f}" if metrics['unif'] is not None else '-'
                v = f"{metrics['var']:.3f}" if metrics['var'] is not None else '-'
                a = ' '.join(alerts) if alerts else 'OK'
                row = f"| {exp} | {metrics['step']} | {r} | {c} | {t} | {u} | {v} | {a} |"
                report_lines.append(row)

                # 记录历史
                if exp not in history:
                    history[exp] = []
                history[exp].append({
                    "time": now,
                    "step": metrics["step"],
                    "recon": metrics["recon"],
                    "consist": metrics["consist"],
                    "temp": metrics["temp"],
                    "unif": metrics["unif"],
                    "var": metrics["var"],
                })
            else:
                report_lines.append(f"| {exp} | - | - | - | - | - | - | 等待日志 |")

        report_lines.append("")
        report_lines.append("## 说明")
        report_lines.append("- Recon < 0.5 正常, > 0.5 警告")
        report_lines.append("- Var > 0.5 警告")
        report_lines.append("- 每60秒更新一次")

        REPORT_PATH.write_text("\n".join(report_lines))
        HISTORY_PATH.write_text(json.dumps(history, indent=2))

        time.sleep(60)


if __name__ == "__main__":
    main()
