#!/usr/bin/env python3
"""
ExpD 训练自动监控与恢复脚本
每 5 分钟检查一次训练状态，自动检测异常并恢复
"""
import os
import sys
import time
import re
import json
import subprocess
import signal
from pathlib import Path
from datetime import datetime

# 配置
LOG_DIR = Path("/workspace/outputs/v2_skipL2_7target_lowrecon_7card_0520")
TRAIN_NAME = "expD_train"
CHECK_INTERVAL = 300  # 5 分钟检查一次
RESTART_COOLDOWN = 60  # 重启后冷却时间
MAX_RESTARTS = 10
CONFIG_PATH = "/workspace/xuannv/configs/xuannv_v2_expD_7target_lowrecon.yaml"
TRAIN_SCRIPT = "/workspace/xuannv/scripts/train/train_ddp_xuannv_v2.py"
NPUS = "0,1,2,3,4,5,6"
NPROC = 7

# 监控指标阈值
COLLAPSE_ACTIVE_DIMS = 15      # active_dims < 15 判定坍缩
COLLAPSE_STD_MEAN = 0.10       # std_mean < 0.10 判定坍缩
COLLAPSE_L2UNIF = -0.3         # l2unif > -0.3 判定坍缩（注意：bank满后值域会上移）
BAD_RECON_AFTER_WARMUP = 1.0   # warmup 后 recon > 1.0
WARMUP_STEPS = 100             # 约等于 warmup epoch 数
MAX_NO_LOG_SECONDS = 600       # 超过 10 分钟无日志输出 = 卡住

# 状态文件
STATE_FILE = LOG_DIR / "monitor_state.json"
REPORT_FILE = LOG_DIR / "monitor_report.txt"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(REPORT_FILE, "a") as f:
        f.write(line + "\n")

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"restarts": 0, "last_restart_time": 0, "history": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_latest_log_lines(n=100):
    """获取 tmux session 最近 n 行输出"""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", TRAIN_NAME, "-p", "-J"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        return lines[-n:]
    except Exception as e:
        return [f"ERROR getting tmux output: {e}"]

def parse_metrics(lines):
    """从日志行解析指标 - 支持行可能被截断的情况"""
    pattern = re.compile(
        r"\[Step\s+(\d+)\]\s+"
        r"recon=([\d.]+)\s+consist=([\d.]+)\s+cls=([\d.]+)\s+"
        r"var=([\d.]+)\s+cov=([\d.]+)\s+decorr=([\d.]+)\s+orth=([\d.]+)\s+"
        r"l2unif=([\-\d.]+)\s+inter_var=([\d.]+)\s+bank=(\d+/\d+)\s+"
        r"spatial=\[(\d+)/(\d+):([\d.]+)\]\s+inter=\[(\d+)/(\d+):([\d.]+)\]"
    )
    matches = []
    for line in lines:
        m = pattern.search(line)
        if m:
            matches.append({
                "step": int(m.group(1)),
                "recon": float(m.group(2)),
                "consist": float(m.group(3)),
                "cls": float(m.group(4)),
                "var": float(m.group(5)),
                "cov": float(m.group(6)),
                "decorr": float(m.group(7)),
                "orth": float(m.group(8)),
                "l2unif": float(m.group(9)),
                "inter_var": float(m.group(10)),
                "bank": m.group(11),
                "spatial_active": int(m.group(12)),
                "spatial_total": int(m.group(13)),
                "spatial_mean": float(m.group(14)),
                "inter_active": int(m.group(15)),
                "inter_total": int(m.group(16)),
                "inter_mean": float(m.group(17)),
            })
    return matches

def check_training_alive():
    """检查训练进程是否存活"""
    try:
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True, text=True, timeout=5
        )
        return TRAIN_NAME in result.stdout
    except:
        return False

def get_last_log_timestamp():
    """获取日志文件最后修改时间"""
    log_files = list(LOG_DIR.glob("train_*.log"))
    if not log_files:
        return time.time()  # 如果没有日志文件，返回当前时间（避免误判）
    latest = max(log_files, key=lambda p: p.stat().st_mtime)
    return latest.stat().st_mtime

def get_last_step_from_metrics(metrics_history):
    """从已解析的 metrics 获取最后 step"""
    if metrics_history:
        return metrics_history[-1]["step"]
    return 0

def stop_training():
    """停止训练进程"""
    log("Stopping training...")
    try:
        subprocess.run(["tmux", "send-keys", "-t", TRAIN_NAME, "C-c"], timeout=10)
        time.sleep(5)
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True, text=True, timeout=5
        )
        if TRAIN_NAME in result.stdout:
            subprocess.run(["tmux", "kill-session", "-t", TRAIN_NAME], timeout=10)
            time.sleep(2)
    except Exception as e:
        log(f"Error stopping training: {e}")

def find_latest_checkpoint():
    """找到最新的 checkpoint"""
    ckpts = list(LOG_DIR.glob("epoch_*.pt"))
    if not ckpts:
        return None
    best_ckpts = [p for p in ckpts if "best" in p.name]
    regular_ckpts = [p for p in ckpts if "best" not in p.name]
    
    latest = None
    if regular_ckpts:
        def epoch_num(p):
            m = re.search(r'epoch_(\d+)', p.name)
            return int(m.group(1)) if m else 0
        latest = max(regular_ckpts, key=epoch_num)
    elif best_ckpts:
        latest = max(best_ckpts, key=lambda p: p.stat().st_mtime)
    return latest

def restart_training(resume_ckpt=None):
    """重启训练"""
    env_vars = f"export ASCEND_RT_VISIBLE_DEVICES={NPUS} && export HCCL_CONNECT_TIMEOUT=600 && export HCCL_EXEC_TIMEOUT=600"
    conda = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv"
    cd = "cd /workspace/xuannv"
    
    if resume_ckpt:
        cmd = (f"torchrun --nproc_per_node={NPROC} {TRAIN_SCRIPT} "
               f"--config {CONFIG_PATH} --resume {resume_ckpt} --save-every 20")
        log(f"Restarting with resume: {resume_ckpt.name}")
    else:
        cmd = (f"torchrun --nproc_per_node={NPROC} {TRAIN_SCRIPT} "
               f"--config {CONFIG_PATH} --save-every 20")
        log("Restarting from scratch")
    
    full_cmd = f"{env_vars} && {conda} && {cd} && {cmd}"
    
    subprocess.run(["tmux", "new-session", "-d", "-s", TRAIN_NAME, "-c", "/workspace/xuannv"], timeout=10)
    subprocess.run(["tmux", "send-keys", "-t", TRAIN_NAME, full_cmd, "Enter"], timeout=10)
    log("Training restarted in tmux session")

def clean_cache():
    """清理 dataset cache"""
    cache_files = list(LOG_DIR.glob("dataset_cache_*.pt"))
    for f in cache_files:
        f.unlink()
        log(f"Removed cache: {f.name}")

def diagnose_and_fix(metrics_history):
    """诊断问题并返回 (has_issue, reason, severity)"""
    if not metrics_history:
        return False, "no_data", "info"
    
    latest = metrics_history[-1]
    issues = []
    severity = "warning"
    
    # 1. 检查 active_dims 坍缩
    if latest["spatial_active"] < COLLAPSE_ACTIVE_DIMS:
        issues.append(f"spatial_active={latest['spatial_active']} < {COLLAPSE_ACTIVE_DIMS}")
        severity = "critical"
    
    # 2. 检查 std_mean
    if latest["spatial_mean"] < COLLAPSE_STD_MEAN:
        issues.append(f"spatial_mean={latest['spatial_mean']:.4f} < {COLLAPSE_STD_MEAN}")
        severity = "critical"
    
    # 3. 检查 l2unif（bank 满后值域会上移，阈值要更宽松）
    if latest["l2unif"] > COLLAPSE_L2UNIF:
        issues.append(f"l2unif={latest['l2unif']:.4f} > {COLLAPSE_L2UNIF}")
        severity = "critical"
    
    # 4. 检查 recon（warmup 后）
    if latest["step"] > WARMUP_STEPS and latest["recon"] > BAD_RECON_AFTER_WARMUP:
        issues.append(f"recon={latest['recon']:.4f} > {BAD_RECON_AFTER_WARMUP} after warmup")
        severity = "warning"
    
    # 5. 检查 NaN
    for k in ["recon", "l2unif", "var", "spatial_mean"]:
        if k in latest and (latest[k] != latest[k]):  # NaN check
            issues.append(f"NaN detected in {k}")
            severity = "critical"
    
    if issues:
        return True, "; ".join(issues), severity
    return False, "ok", "ok"

def main():
    log("="*60)
    log("ExpD Auto Monitor Started (v2)")
    log(f"Log dir: {LOG_DIR}")
    log(f"Check interval: {CHECK_INTERVAL}s")
    log("="*60)
    
    state = load_state()
    last_restart_time = state.get("last_restart_time", 0)
    restarts = state.get("restarts", 0)
    prev_step = 0
    stuck_count = 0
    
    while True:
        try:
            now = time.time()
            
            # 1. 检查训练是否存活
            alive = check_training_alive()
            if not alive:
                log("CRITICAL: Training session not found!")
                if restarts < MAX_RESTARTS:
                    ckpt = find_latest_checkpoint()
                    clean_cache()
                    restart_training(ckpt)
                    restarts += 1
                    last_restart_time = now
                    state["restarts"] = restarts
                    state["last_restart_time"] = last_restart_time
                    save_state(state)
                else:
                    log(f"MAX RESTARTS ({MAX_RESTARTS}) reached, giving up")
                    break
                time.sleep(CHECK_INTERVAL)
                continue
            
            # 2. 获取并解析指标
            lines = get_latest_log_lines(150)
            metrics = parse_metrics(lines)
            
            if metrics:
                latest = metrics[-1]
                current_step = latest["step"]
                
                # 检测是否卡住（step 没有前进）
                if current_step == prev_step and current_step > 0:
                    stuck_count += 1
                    if stuck_count >= 2:  # 连续两次检查 step 不变
                        log(f"WARNING: Step stuck at {current_step} for {stuck_count} checks")
                else:
                    stuck_count = 0
                prev_step = current_step
                
                state["history"].append({
                    "time": datetime.now().isoformat(),
                    **latest
                })
                state["history"] = state["history"][-500:]
                save_state(state)
                
                # 打印状态摘要
                log(f"Step {latest['step']:4d} | "
                    f"spatial=[{latest['spatial_active']}/{latest['spatial_total']}:{latest['spatial_mean']:.3f}] | "
                    f"inter=[{latest['inter_active']}/{latest['inter_total']}:{latest['inter_mean']:.3f}] | "
                    f"recon={latest['recon']:.4f} l2unif={latest['l2unif']:.4f} "
                    f"var={latest['var']:.4f} bank={latest['bank']}")
                
                # 3. 诊断
                has_issue, reason, severity = diagnose_and_fix(metrics)
                if has_issue:
                    log(f"ALERT [{severity.upper()}]: {reason}")
                    if severity == "critical":
                        log("EMBEDDING COLLAPSE DETECTED - manual intervention required")
                        # 对于坍缩，记录到告警文件但不自动重启（需要代码修复）
                        alert_file = LOG_DIR / "ALERT_COLLAPSE.txt"
                        with open(alert_file, "w") as f:
                            f.write(f"Time: {datetime.now().isoformat()}\n")
                            f.write(f"Reason: {reason}\n")
                            f.write(f"Latest metrics: {json.dumps(latest, indent=2)}\n")
            else:
                log("No recent metrics parsed")
                # 检查是否卡住
                log_mtime = get_last_log_timestamp()
                elapsed = now - log_mtime
                if elapsed > MAX_NO_LOG_SECONDS:
                    log(f"WARNING: No log output for {int(elapsed)}s, may be stuck")
            
        except Exception as e:
            log(f"Monitor error: {e}")
            import traceback
            log(traceback.format_exc())
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
