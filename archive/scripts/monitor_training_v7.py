#!/usr/bin/env python3
"""V7 训练监控脚本 — 定期检查训练状态并告警."""
from __future__ import annotations

import os
import sys
import re
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")

# 监控配置
CHECK_INTERVAL = 300  # 每 5 分钟检查一次
ALERT_COOLDOWN = 1800  # 同一告警 30 分钟内不重复
TMUX_SESSION = "v7_train"
OUTPUT_DIR = Path("/workspace/outputs/xuannv_backbone_v7_phase1_v2")
LOG_FILE = OUTPUT_DIR / "monitor.log"
STATUS_FILE = OUTPUT_DIR / "monitor_status.json"

# 告警阈值
THRESHOLDS = {
    "vicreg_max": 1.0,          # vicreg > 1.0 警告
    "pre_unif_max": -0.5,       # pre_unif > -0.5 坍缩
    "recon_max": 0.5,           # recon > 0.5 重建失败（warmup 后）
    "stagnation_epochs": 5,     # vicreg 连续 N epoch 不下降
}

# 告警历史（防重复）
alerts_sent: dict[str, float] = {}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_latest_log_text(n_lines: int = 50) -> str:
    """从日志文件读取最新输出（避免 tmux 换行截断问题）."""
    try:
        log_dir = Path("/workspace/outputs/xuannv_backbone_v7_phase1_v2")
        log_files = sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            # fallback to tmux
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", TMUX_SESSION, "-p"],
                capture_output=True, text=True, timeout=10
            )
            return "\n".join(result.stdout.splitlines()[-n_lines:])
        
        with open(log_files[0], "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        # 处理 tmux 日志中可能的换行截断：先把所有内容合并，再用正则匹配
        return text
    except Exception as e:
        return f"[ERROR] Failed to read log: {e}"


def check_process_alive() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["tmux", "ls"], capture_output=True, text=True, timeout=5
        )
        if TMUX_SESSION in result.stdout:
            return True, "running"
        return False, "tmux session not found"
    except Exception as e:
        return False, str(e)


def parse_latest_epoch(log_text: str) -> dict | None:
    """从日志文本解析最新的 epoch 指标（支持 tmux 换行截断）."""
    # 先把可能的换行截断合并（tmux 窄窗口会在数字中间换行）
    # 策略：删除数字和字母之间的换行，保留 Epoch 之间的完整行
    merged = log_text.replace('\n', ' ')
    
    # 匹配格式: Epoch N/M | Loss: x.xxxx | Recon: x.xxxx | VICReg: x.xxxx ...
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+"
        r"Loss:\s+([\d.e+-]+)\s+\|\s+"
        r"Recon:\s+([\d.e+-]+)\s+\|\s+"
        r"VICReg:\s+([\d.e+-]+)\s+\(var=([\d.e+-]+)\s+cov=([\d.e+-]+)\)\s+\|\s+"
        r"KoLeo:\s+([\d.e+-]+)\s+\|\s+"
        r"PreUnif:\s+([\d.e+-]+)\s+\|\s+"
        r"EncUnif:\s+([\d.e+-]+)\s+\|\s+"
        r"Consist:\s+([\d.e+-]+)\s+\|\s+"
        r"Cls:\s+([\d.e+-]+)\s+\|\s+"
        r"Temporal:\s+([\d.e+-]+)\s+\|\s+"
        r"LR:\s+([\d.e+-]+)\s+\|"
    )
    
    matches = list(pattern.finditer(merged))
    if not matches:
        return None
    
    m = matches[-1]  # 取最新的
    return {
        "epoch": int(m.group(1)),
        "total_loss": float(m.group(2)),
        "recon": float(m.group(3)),
        "vicreg": float(m.group(4)),
        "vicreg_var": float(m.group(5)),
        "vicreg_cov": float(m.group(6)),
        "koleo": float(m.group(7)),
        "pre_unif": float(m.group(8)),
        "enc_unif": float(m.group(9)),
        "consist": float(m.group(10)),
        "cls": float(m.group(11)),
        "temporal": float(m.group(12)),
        "lr": float(m.group(13)),
    }


def check_npu_usage() -> list[dict]:
    """检查 NPU 使用率."""
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines()
        # 简单解析，看是否有运行中的进程
        gpus = []
        for i, line in enumerate(lines):
            if f"NPU {i}" in line and i < 8:
                # 简化处理
                pass
        return []
    except Exception:
        return []


def send_alert(alert_type: str, message: str) -> None:
    now = time.time()
    last = alerts_sent.get(alert_type, 0)
    if now - last < ALERT_COOLDOWN:
        return
    alerts_sent[alert_type] = now
    
    # 打印醒目的告警
    banner = "=" * 60
    log(banner)
    log(f"🚨 ALERT [{alert_type}]")
    log(message)
    log(banner)


def evaluate_epoch(metrics: dict, history: list[dict]) -> list[str]:
    """评估当前 epoch 指标，返回告警列表."""
    alerts = []
    epoch = metrics["epoch"]
    
    # 1. vicreg 过高
    if metrics["vicreg"] > THRESHOLDS["vicreg_max"]:
        alerts.append(f"Epoch {epoch}: VICReg={metrics['vicreg']:.4f} > {THRESHOLDS['vicreg_max']} — 嵌入坍缩风险")
    
    # 2. pre_unif 过高（嵌入坍缩）
    if metrics["pre_unif"] > THRESHOLDS["pre_unif_max"]:
        alerts.append(f"Epoch {epoch}: PreUnif={metrics['pre_unif']:.4f} > {THRESHOLDS['pre_unif_max']} — 嵌入坍缩！")
    
    # 3. recon 过高（warmup 后，epoch > 20）
    if epoch > 20 and metrics["recon"] > THRESHOLDS["recon_max"]:
        alerts.append(f"Epoch {epoch}: Recon={metrics['recon']:.4f} > {THRESHOLDS['recon_max']} — 重建质量差")
    
    # 4. vicreg 停滞检测（连续 N epoch 不下降）
    if len(history) >= THRESHOLDS["stagnation_epochs"]:
        recent = [h["vicreg"] for h in history[-THRESHOLDS["stagnation_epochs"]:]]
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            alerts.append(f"Epoch {epoch}: VICReg 连续 {THRESHOLDS['stagnation_epochs']} epoch 未下降 — 训练停滞")
    
    # 5. NaN/Inf 检测
    for k, v in metrics.items():
        if isinstance(v, float) and (v != v or abs(v) > 1e6):  # NaN or huge
            alerts.append(f"Epoch {epoch}: {k}={v} — 数值异常（NaN/Inf/Overflow）")
    
    return alerts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("V7 训练监控启动")
    log(f"检查间隔: {CHECK_INTERVAL}s, 告警冷却: {ALERT_COOLDOWN}s")
    log("=" * 60)
    
    history: list[dict] = []
    last_epoch = 0
    
    while True:
        time.sleep(CHECK_INTERVAL)
        
        # 1. 检查进程存活
        alive, status = check_process_alive()
        if not alive:
            send_alert("process_dead", f"训练进程异常终止: {status}")
            continue
        
        # 2. 获取最新输出
        output = get_latest_log_text(n_lines=50)
        
        # 3. 解析指标
        metrics = parse_latest_epoch(output)
        if metrics is None:
            # 可能还在加载 cache
            if "waiting for cache" in output:
                log("[INFO] Rank 0 仍在生成 dataset cache...")
            elif "Training from scratch" in output:
                log("[INFO] 训练尚未开始第一个 epoch...")
            else:
                log("[WARN] 无法解析训练指标，请检查 tmux 输出")
            continue
        
        # 4. 记录历史
        epoch = metrics["epoch"]
        if epoch > last_epoch:
            history.append(metrics)
            last_epoch = epoch
            log(
                f"Epoch {epoch}/{metrics['total_epochs']} | "
                f"Loss={metrics['total_loss']:.4f} | "
                f"Recon={metrics['recon']:.4f} | "
                f"VICReg={metrics['vicreg']:.4f} | "
                f"PreUnif={metrics['pre_unif']:.4f} | "
                f"LR={metrics['lr']:.2e}"
            )
            
            # 5. 评估并告警
            alerts = evaluate_epoch(metrics, history)
            for alert in alerts:
                send_alert("metric_alert", alert)
        
        # 6. 保存状态
        status = {
            "last_check": datetime.now().isoformat(),
            "last_epoch": last_epoch,
            "latest_metrics": metrics,
            "history_length": len(history),
        }
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2, default=str)


if __name__ == "__main__":
    main()
