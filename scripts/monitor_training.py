#!/usr/bin/env python3
"""通用训练监控脚本 — 支持多种训练器输出格式，可监控任意实验.

用法示例:
    # 基本监控（只读日志，不干预）
    python scripts/monitor_training.py \
        --output-dir /workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523 \
        --interval 300

    # 带 tmux 进程存活检查
    python scripts/monitor_training.py \
        --output-dir /workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523 \
        --tmux-session expE_train \
        --interval 300

    # 自动恢复模式（检测到崩溃时自动 resume）
    python scripts/monitor_training.py \
        --output-dir /workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523 \
        --tmux-session expE_train \
        --auto-restart \
        --config configs/xuannv_v2_expE_pure_recon.yaml \
        --train-script scripts/train/train_xuannv_v2.py \
        --npus 0,1,2,3,4,5,6 \
        --interval 300
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")

# ---------------------------------------------------------------------------
# 指标解析模式（支持多种训练器输出格式）
# ---------------------------------------------------------------------------

# V12/V13/ExpD 格式: [Step 1234] recon=0.1234 consist=0.5678 ...
STEP_PATTERN_V1 = re.compile(
    r"\[Step\s+(\d+)\]\s+"
    r"(?:recon=([\d.]+)\s+)?"
    r"(?:consist=([\d.]+)\s+)?"
    r"(?:cls=([\d.]+)\s+)?"
    r"(?:var=([\d.]+)\s+)?"
    r"(?:cov=([\d.]+)\s+)?"
    r"(?:decorr=([\d.]+)\s+)?"
    r"(?:orth=([\d.]+)\s+)?"
    r"(?:l2unif=([-\d.]+)\s+)?"
    r"(?:raw_unif=([-\d.]+)\s+)?"
    r"(?:pre_unif=([-\d.]+)\s+)?"
    r"(?:inter_var=([\d.]+)\s+)?"
    r"(?:bank=(\d+/\d+)\s+)?"
    r"(?:spatial=\[(\d+)/(\d+):([\d.]+)\]\s+)?"
    r"(?:inter=\[(\d+)/(\d+):([\d.]+)\]\s+)?"
    r"(?:active=(\d+)/(\d+)\s+)?"
    r"(?:std_mean=([\d.]+)\s+)?"
    r"(?:lr=([\d.e+-]+))?"
)

# Epoch 汇总行: Epoch 001/100 | total=0.3886 recon=1.1580 ...
EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+).*?"
    r"(?:total=([\d.]+)\s+)?"
    r"(?:recon=([\d.]+)\s+)?"
    r"(?:consist=([\d.]+)\s+)?"
    r"(?:uniform=([-\d.]+)\s+)?"
    r"(?:l2unif=([-\d.]+)\s+)?"
    r"(?:raw_unif=([-\d.]+)\s+)?"
    r"(?:pre_unif=([-\d.]+)\s+)?"
    r"(?:active=(\d+)/(\d+)\s+)?"
    r"(?:std_mean=([\d.]+)\s+)?"
    r"(?:lr=([\d.e+-]+))?"
)


def parse_metrics(lines: list[str]) -> list[dict]:
    """从日志行解析指标，支持多种格式."""
    matches = []
    for line in lines:
        # 先尝试 Step 模式
        m = STEP_PATTERN_V1.search(line)
        if m:
            d = {"type": "step", "step": int(m.group(1))}
            fields = [
                ("recon", 2, float), ("consist", 3, float), ("cls", 4, float),
                ("var", 5, float), ("cov", 6, float), ("decorr", 7, float),
                ("orth", 8, float), ("l2unif", 9, float), ("raw_unif", 10, float),
                ("pre_unif", 11, float), ("inter_var", 12, float), ("bank", 13, str),
                ("spatial_active", 14, int), ("spatial_total", 15, int), ("spatial_mean", 16, float),
                ("inter_active", 17, int), ("inter_total", 18, int), ("inter_mean", 19, float),
                ("active", 20, int), ("active_denom", 21, int), ("std_mean", 22, float),
                ("lr", 23, float),
            ]
            for key, idx, cast in fields:
                val = m.group(idx)
                if val is not None:
                    try:
                        d[key] = cast(val)
                    except (ValueError, TypeError):
                        pass
            matches.append(d)
            continue

        # 再尝试 Epoch 模式
        m = EPOCH_PATTERN.search(line)
        if m:
            d = {"type": "epoch", "epoch": int(m.group(1))}
            fields = [
                ("total", 2, float), ("recon", 3, float), ("consist", 4, float),
                ("uniform", 5, float), ("l2unif", 6, float), ("raw_unif", 7, float),
                ("pre_unif", 8, float), ("active", 9, int), ("active_denom", 10, int),
                ("std_mean", 11, float), ("lr", 12, float),
            ]
            for key, idx, cast in fields:
                val = m.group(idx)
                if val is not None:
                    try:
                        d[key] = cast(val)
                    except (ValueError, TypeError):
                        pass
            matches.append(d)

    return matches


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, timeout: int = 30) -> str:
    """运行 shell 命令并返回输出."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def log(msg: str, report_file: Path | None = None):
    """打印并记录日志."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if report_file:
        with open(report_file, "a") as f:
            f.write(line + "\n")


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {"restarts": 0, "last_restart_time": 0, "history": [], "prev_step": 0, "stuck_count": 0}


def save_state(state: dict, state_file: Path):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def get_latest_log_lines(tmux_session: str | None, log_file: Path | None, n: int = 150) -> list[str]:
    """获取最新日志行，优先 tmux，其次日志文件."""
    if tmux_session:
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", tmux_session, "-p", "-J"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            return lines[-n:]
        except Exception:
            pass

    if log_file and log_file.exists():
        lines = log_file.read_text().splitlines()
        return lines[-n:]

    return []


def check_training_alive(tmux_session: str | None) -> bool:
    if not tmux_session:
        return True  # 无 tmux 模式，假设存活
    try:
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True, text=True, timeout=5
        )
        return tmux_session in result.stdout
    except Exception:
        return False


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    ckpts = list(output_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None
    regular = [p for p in ckpts if "best" not in p.name]
    best = [p for p in ckpts if "best" in p.name]

    def epoch_num(p: Path) -> int:
        m = re.search(r"epoch_(\d+)", p.name)
        return int(m.group(1)) if m else 0

    if regular:
        return max(regular, key=epoch_num)
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    return None


def stop_training(tmux_session: str):
    log(f"Stopping tmux session: {tmux_session}")
    try:
        subprocess.run(["tmux", "send-keys", "-t", tmux_session, "C-c"], timeout=10)
        time.sleep(5)
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True, text=True, timeout=5
        )
        if tmux_session in result.stdout:
            subprocess.run(["tmux", "kill-session", "-t", tmux_session], timeout=10)
            time.sleep(2)
    except Exception as e:
        log(f"Error stopping training: {e}")


def restart_training(
    tmux_session: str,
    train_script: str,
    config: str,
    npus: str,
    nproc: int,
    resume_ckpt: Path | None,
):
    env_vars = f"export ASCEND_RT_VISIBLE_DEVICES={npus} && export HCCL_CONNECT_TIMEOUT=600 && export HCCL_EXEC_TIMEOUT=600"
    conda = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv"
    cd = "cd /workspace/xuannv"

    if resume_ckpt:
        cmd = f"torchrun --nproc_per_node={nproc} {train_script} --config {config} --resume {resume_ckpt} --save-every 20"
        log(f"Restarting with resume: {resume_ckpt.name}")
    else:
        cmd = f"torchrun --nproc_per_node={nproc} {train_script} --config {config} --save-every 20"
        log("Restarting from scratch")

    full_cmd = f"{env_vars} && {conda} && {cd} && {cmd}"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", tmux_session, "-c", "/workspace/xuannv"],
        timeout=10,
    )
    subprocess.run(
        ["tmux", "send-keys", "-t", tmux_session, full_cmd, "Enter"],
        timeout=10,
    )
    log("Training restarted in tmux session")


def diagnose(
    latest: dict,
    thresholds: dict,
    warmup_steps: int,
) -> tuple[bool, str, str]:
    """诊断问题，返回 (has_issue, reason, severity)."""
    issues = []
    severity = "warning"

    step = latest.get("step", 0)

    # 1. active_dims 坍缩
    active = latest.get("active") or latest.get("spatial_active")
    active_thresh = thresholds.get("collapse_active_dims", 15)
    if active is not None and active < active_thresh:
        issues.append(f"active={active} < {active_thresh}")
        severity = "critical"

    # 2. std_mean 过低
    std_mean = latest.get("std_mean") or latest.get("spatial_mean")
    std_thresh = thresholds.get("collapse_std_mean", 0.10)
    if std_mean is not None and std_mean < std_thresh:
        issues.append(f"std_mean={std_mean:.4f} < {std_thresh}")
        severity = "critical"

    # 3. uniformity 异常
    for unif_key in ["l2unif", "raw_unif", "pre_unif", "uniform"]:
        val = latest.get(unif_key)
        if val is not None:
            unif_thresh = thresholds.get("collapse_l2unif", -0.3)
            if val > unif_thresh:
                issues.append(f"{unif_key}={val:.4f} > {unif_thresh}")
                severity = "critical"
            break  # 只检查第一个存在的 uniformity 指标

    # 4. recon 过高（warmup 后）
    recon = latest.get("recon")
    recon_thresh = thresholds.get("bad_recon_after_warmup", 1.0)
    if step > warmup_steps and recon is not None and recon > recon_thresh:
        issues.append(f"recon={recon:.4f} > {recon_thresh} after warmup")

    # 5. NaN 检测
    for k, v in latest.items():
        if isinstance(v, float) and v != v:  # NaN check
            issues.append(f"NaN in {k}")
            severity = "critical"

    if issues:
        return True, "; ".join(issues), severity
    return False, "ok", "ok"


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="通用训练监控脚本")
    parser.add_argument("--output-dir", type=Path, required=True, help="实验输出目录")
    parser.add_argument("--tmux-session", type=str, default=None, help="tmux session 名称")
    parser.add_argument("--interval", type=int, default=300, help="检查间隔（秒）")
    parser.add_argument("--log-file", type=Path, default=None, help="训练日志文件路径（默认自动搜索 output-dir/train*.log）")
    parser.add_argument("--auto-restart", action="store_true", help="检测到崩溃时自动恢复训练")
    parser.add_argument("--config", type=str, default=None, help="训练配置路径（auto-restart 需要）")
    parser.add_argument("--train-script", type=str, default=None, help="训练脚本路径（auto-restart 需要）")
    parser.add_argument("--npus", type=str, default="0,1,2,3,4,5,6,7", help="NPU 设备列表")
    parser.add_argument("--nproc", type=int, default=8, help="DDP 进程数")
    parser.add_argument("--max-restarts", type=int, default=10, help="最大自动重启次数")
    parser.add_argument("--warmup-steps", type=int, default=100, help="warmup 步数阈值")
    parser.add_argument("--thresholds", type=str, default=None,
                        help='告警阈值 JSON，如 \'{"collapse_active_dims":15,"collapse_std_mean":0.1}\'')
    parser.add_argument("--max-no-log-seconds", type=int, default=600, help="超过此时间无日志输出则告警")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file
    if log_file is None:
        log_files = sorted(output_dir.glob("train*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if log_files:
            log_file = log_files[0]

    report_file = output_dir / "monitor_report.txt"
    state_file = output_dir / "monitor_state.json"

    thresholds = {
        "collapse_active_dims": 15,
        "collapse_std_mean": 0.10,
        "collapse_l2unif": -0.3,
        "bad_recon_after_warmup": 1.0,
    }
    if args.thresholds:
        try:
            thresholds.update(json.loads(args.thresholds))
        except json.JSONDecodeError as e:
            log(f"Invalid thresholds JSON: {e}", report_file)
            sys.exit(1)

    state = load_state(state_file)
    restarts = state.get("restarts", 0)
    prev_step = state.get("prev_step", 0)
    stuck_count = state.get("stuck_count", 0)

    log("=" * 60, report_file)
    log(f"Monitor started: {output_dir.name}", report_file)
    log(f"Log file: {log_file}", report_file)
    log(f"Tmux session: {args.tmux_session}", report_file)
    log(f"Interval: {args.interval}s", report_file)
    log(f"Auto-restart: {args.auto_restart}", report_file)
    log(f"Thresholds: {thresholds}", report_file)
    log("=" * 60, report_file)

    while True:
        try:
            now = time.time()

            # 1. 检查训练进程存活
            alive = check_training_alive(args.tmux_session)
            if not alive and args.tmux_session:
                log("CRITICAL: Training session not found!", report_file)
                if args.auto_restart and restarts < args.max_restarts:
                    ckpt = find_latest_checkpoint(output_dir)
                    restart_training(
                        args.tmux_session, args.train_script or "",
                        args.config or "", args.npus, args.nproc, ckpt,
                    )
                    restarts += 1
                    state["restarts"] = restarts
                    state["last_restart_time"] = now
                    save_state(state, state_file)
                else:
                    log("Max restarts reached or auto-restart disabled, stopping monitor", report_file)
                    break
                time.sleep(args.interval)
                continue

            # 2. 获取并解析指标
            lines = get_latest_log_lines(args.tmux_session, log_file, 150)
            metrics = parse_metrics(lines)

            if metrics:
                latest = metrics[-1]
                current_step = latest.get("step", latest.get("epoch", 0))

                # 检测卡住
                if current_step == prev_step and current_step > 0:
                    stuck_count += 1
                    if stuck_count >= 2:
                        log(f"WARNING: Step stuck at {current_step} for {stuck_count} checks", report_file)
                else:
                    stuck_count = 0
                prev_step = current_step

                state["history"].append({"time": datetime.now().isoformat(), **latest})
                state["history"] = state["history"][-500:]
                state["prev_step"] = prev_step
                state["stuck_count"] = stuck_count
                save_state(state, state_file)

                # 打印摘要
                parts = [f"step={current_step}"]
                for k in ["recon", "l2unif", "raw_unif", "uniform", "active", "std_mean"]:
                    if k in latest:
                        parts.append(f"{k}={latest[k]:.4f}" if isinstance(latest[k], float) else f"{k}={latest[k]}")
                log(" | ".join(parts), report_file)

                # 3. 诊断
                has_issue, reason, severity = diagnose(latest, thresholds, args.warmup_steps)
                if has_issue:
                    log(f"ALERT [{severity.upper()}]: {reason}", report_file)
                    if severity == "critical":
                        alert_file = output_dir / "ALERT_COLLAPSE.txt"
                        with open(alert_file, "w") as f:
                            f.write(f"Time: {datetime.now().isoformat()}\n")
                            f.write(f"Reason: {reason}\n")
                            f.write(f"Metrics: {json.dumps(latest, indent=2, default=str)}\n")
            else:
                log("No recent metrics parsed", report_file)
                # 检查日志是否卡住
                if log_file and log_file.exists():
                    elapsed = now - log_file.stat().st_mtime
                    if elapsed > args.max_no_log_seconds:
                        log(f"WARNING: No log output for {int(elapsed)}s", report_file)

        except KeyboardInterrupt:
            log("Monitor stopped by user", report_file)
            break
        except Exception as e:
            log(f"Monitor error: {e}", report_file)
            import traceback
            log(traceback.format_exc(), report_file)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
