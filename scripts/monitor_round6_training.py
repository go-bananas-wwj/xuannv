#!/usr/bin/env python3
"""Round 6 训练监控守护进程 — 每10分钟检查一次，异常时自动修复。

检测项:
1. 训练进程是否存在
2. 最新epoch指标是否正常
3. NaN/Inf
4. 坍缩信号 (raw_unif > -0.5)
5. 重建质量差 (recon > 0.5 after warmup)

修复措施:
- 进程崩溃 → 自动 resume 最新 checkpoint
- NaN/Inf → 降低 risky loss weights 后 resume
- 坍缩 → 增加 anti-collapse weights 后 resume
"""

import os
import re
import sys
import time
import glob
import subprocess
import shutil
from datetime import datetime

sys.path.insert(0, "/workspace/xuannv")

EXPERIMENTS = [
    "r6_consist_kappa5k",
    "r6_consist_mild_100ep",
    "r6_temporal_consist_k5k",
    "r6_high_consist_k5k",
    "r6_no_consist_k5k",
    "r6_temporal_recon_consist",
]

OUTPUT_ROOT = "/workspace/outputs/round6"
MONITOR_LOG = f"{OUTPUT_ROOT}/monitor.log"
REPORT_FILE = f"{OUTPUT_ROOT}/monitor_report.md"

# 异常阈值
THRESHOLDS = {
    "raw_unif_warn": -1.0,   # raw_unif > -1.0 警告
    "raw_unif_crit": -0.5,   # raw_unif > -0.5 严重（坍缩）
    "recon_warn": 0.3,       # recon > 0.3 警告
    "recon_crit": 0.5,       # recon > 0.5 严重
    "var_warn": 0.5,         # var > 0.5 警告
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def get_latest_epoch_metrics(exp_name):
    """从train.log提取最新epoch的指标."""
    log_path = f"{OUTPUT_ROOT}/{exp_name}/train.log"
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        lines = f.readlines()

    # 找最后一个 Epoch 行
    for line in reversed(lines):
        if "Epoch" in line and "recon=" in line:
            return line.strip()
    return None


def parse_metrics(line):
    """解析epoch指标行."""
    metrics = {}
    patterns = {
        "epoch": r"Epoch\s+(\d+)",
        "recon": r"recon=([0-9.e+-]+)",
        "var": r"var=([0-9.e+-]+)",
        "raw_unif": r"l2unif=([0-9.e+-]+)",
        "consist": r"consist=([0-9.e+-]+)",
        "temporal": r"temporal=([0-9.e+-]+)",
        "bank": r"bank=(\d+/\d+)",
        "active": r"active=(\d+/\d+)",
        "lr": r"lr=([0-9.e+-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, line)
        if m:
            val = m.group(1)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    # 找epoch
    m = re.search(r"Epoch\s+(\d+)", line)
    if m:
        metrics["epoch"] = int(m.group(1))
    return metrics


def check_nan_inf(exp_name):
    """检查train.log是否有NaN/Inf."""
    log_path = f"{OUTPUT_ROOT}/{exp_name}/train.log"
    if not os.path.exists(log_path):
        return False, ""
    with open(log_path, "r") as f:
        content = f.read()
    if "nan" in content.lower() or "inf" in content.lower():
        # 找最后一处
        lines = content.split("\n")
        for line in reversed(lines):
            if "nan" in line.lower() or "inf" in line.lower():
                return True, line.strip()
    return False, ""


def find_process(exp_name):
    """查找训练进程."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"train_ddp_v13.py.*{exp_name}"],
            capture_output=True, text=True
        )
        pids = [p for p in result.stdout.strip().split("\n") if p]
        return len(pids) > 0
    except Exception:
        return False


def find_latest_checkpoint(exp_name):
    """找最新的checkpoint文件."""
    ckpt_dir = f"{OUTPUT_ROOT}/{exp_name}"
    if not os.path.exists(ckpt_dir):
        return None
    # 找 epoch_best_*.pt 或 epoch_*.pt
    ckpts = glob.glob(f"{ckpt_dir}/epoch_*.pt")
    if not ckpts:
        return None
    # 按修改时间排序
    ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return ckpts[0]


def kill_tmux_session(exp_name):
    """杀掉tmux session."""
    session = f"r6_{exp_name}"
    subprocess.run(["tmux", "kill-session", "-t", session],
                   capture_output=True)
    log(f"  [{exp_name}] Killed tmux session: {session}")


def launch_resume(exp_name, checkpoint_path, extra_args=""):
    """从checkpoint恢复训练."""
    session = f"r6_{exp_name}"
    config = f"configs/round6_8gpu/{exp_name}.yaml"
    # 分配NPU（按实验索引）
    gpu_map = {exp: i+2 for i, exp in enumerate(EXPERIMENTS)}
    gpu = gpu_map.get(exp_name, 2)
    port = 29700 + gpu

    kill_tmux_session(exp_name)
    time.sleep(1)

    cmd = (
        f"export ASCEND_RT_VISIBLE_DEVICES={gpu} && "
        f"source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv && "
        f"torchrun --nproc_per_node=1 --master_port={port} "
        f"scripts/train/train_ddp_v13.py --config {config} "
        f"--resume {checkpoint_path} --save-every 20 {extra_args}"
    )

    subprocess.run([
        "tmux", "new-session", "-d", "-s", session, "-c", "/workspace/xuannv",
        "bash", "-c", cmd
    ], capture_output=True)

    log(f"  [{exp_name}] Resumed from {os.path.basename(checkpoint_path)} on NPU {gpu}")


def modify_config_and_resume(exp_name, modifications, reason):
    """修改config后soft-restart."""
    config_path = f"/workspace/xuannv/configs/round6_8gpu/{exp_name}.yaml"
    backup_path = f"/workspace/xuannv/configs/round6_8gpu/{exp_name}.yaml.bak"

    # 备份
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)

    # 读取并修改
    with open(config_path, "r") as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    for line in lines:
        new_line = line
        for key, new_val in modifications.items():
            # 匹配 training: 下的参数
            if re.match(rf"^\s+{key}\s*:\s*", line):
                old_val = line.split(":", 1)[1].strip()
                new_lines.append(f"  {key}: {new_val}\n")
                log(f"  [{exp_name}] Config changed: {key} {old_val} -> {new_val} ({reason})")
                modified = True
                break
        else:
            new_lines.append(line)

    if modified:
        with open(config_path, "w") as f:
            f.writelines(new_lines)

    # 找最新ckpt并resume
    ckpt = find_latest_checkpoint(exp_name)
    if ckpt:
        launch_resume(exp_name, ckpt)
    else:
        log(f"  [{exp_name}] WARNING: No checkpoint found for resume!")


def check_experiment(exp_name):
    """检查单个实验的状态，返回 (status, action_taken)."""
    status = "OK"
    action = ""

    # 1. 检查进程是否存在
    has_process = find_process(exp_name)
    if not has_process:
        log(f"[{exp_name}] CRASH DETECTED: No training process!")
        ckpt = find_latest_checkpoint(exp_name)
        if ckpt:
            launch_resume(exp_name, ckpt)
            return "CRASHED_RESUMED", f"Resumed from {os.path.basename(ckpt)}"
        else:
            log(f"  [{exp_name}] No checkpoint available!")
            return "CRASHED_NO_CKPT", "No checkpoint"

    # 2. 检查NaN/Inf
    has_nan, nan_line = check_nan_inf(exp_name)
    if has_nan:
        log(f"[{exp_name}] NaN/Inf DETECTED: {nan_line}")
        # 降低所有loss weights到50%
        modifications = {}
        # 读取config获取当前值
        config_path = f"/workspace/xuannv/configs/round6_8gpu/{exp_name}.yaml"
        with open(config_path) as f:
            content = f.read()
        for key in ["consistency_weight", "temporal_contrastive_weight", "variance_weight", "batch_uniformity_weight"]:
            m = re.search(rf"^\s+{key}\s*:\s*([0-9.e+-]+)", content, re.MULTILINE)
            if m:
                old_val = float(m.group(1))
                if old_val > 0:
                    new_val = round(old_val * 0.5, 4)
                    modifications[key] = new_val
        modify_config_and_resume(exp_name, modifications, "NaN/Inf auto-fix")
        return "NAN_FIXED", "Reduced loss weights by 50%"

    # 3. 检查epoch指标
    line = get_latest_epoch_metrics(exp_name)
    if not line:
        return "NO_METRICS", "Waiting for first epoch"

    metrics = parse_metrics(line)
    epoch = metrics.get("epoch", 0)
    raw_unif = metrics.get("raw_unif", None)
    recon = metrics.get("recon", None)
    var = metrics.get("var", None)

    issues = []

    # 检查raw_unif（注意l2unif值越小越好，这里看是否太大）
    if raw_unif is not None:
        # l2unif值越小表示uniformity越好，但这里我们用原始值判断
        # 如果l2unif很大（接近1），说明uniformity差
        if raw_unif > 0.8:
            issues.append(f"l2unif={raw_unif:.3f} (too high, possible collapse)")

    # 检查recon
    if recon is not None and epoch > 10:
        if recon > THRESHOLDS["recon_crit"]:
            issues.append(f"recon={recon:.3f} > {THRESHOLDS['recon_crit']}")
        elif recon > THRESHOLDS["recon_warn"]:
            issues.append(f"recon={recon:.3f} > {THRESHOLDS['recon_warn']} (warn)")

    # 检查var
    if var is not None and var > THRESHOLDS["var_warn"]:
        issues.append(f"var={var:.3f} > {THRESHOLDS['var_warn']}")

    if issues:
        log(f"[{exp_name}] Epoch {epoch}: {', '.join(issues)}")
        # 对于严重问题，尝试修复
        if any("recon=" in i and THRESHOLDS["recon_crit"] in i for i in issues) or \
           any("collapse" in i for i in issues):
            log(f"  [{exp_name}] Attempting auto-fix...")
            # 增加anti-collapse力度
            modifications = {
                "batch_uniformity_weight": 0.02,
                "variance_weight": 0.8,
            }
            modify_config_and_resume(exp_name, modifications, "Anti-collapse fix")
            return "ISSUE_FIXED", "Increased anti-collapse weights"
        status = "WARNING"
    else:
        status = "OK"

    return status, action


def generate_report():
    """生成监控报告."""
    lines = ["# Round 6 训练监控报告", "", f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("| 实验 | Epoch | Recon | Var | L2Unif | Bank | 状态 |")
    lines.append("|------|-------|-------|-----|--------|------|------|")

    for exp in EXPERIMENTS:
        line = get_latest_epoch_metrics(exp)
        if line:
            m = parse_metrics(line)
            epoch = m.get("epoch", "-")
            recon = f"{m.get('recon', '-'):.3f}" if isinstance(m.get('recon'), float) else "-"
            var = f"{m.get('var', '-'):.3f}" if isinstance(m.get('var'), float) else "-"
            unif = f"{m.get('raw_unif', '-'):.3f}" if isinstance(m.get('raw_unif'), float) else "-"
            bank = m.get("bank", "-")
            status = check_experiment(exp)[0]
            lines.append(f"| {exp} | {epoch} | {recon} | {var} | {unif} | {bank} | {status} |")
        else:
            lines.append(f"| {exp} | - | - | - | - | - | 等待数据 |")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))


def main():
    log("=" * 60)
    log("Round 6 Training Monitor Started")
    log(f"Experiments: {', '.join(EXPERIMENTS)}")
    log(f"Check interval: 600 seconds (10 minutes)")
    log("=" * 60)

    iteration = 0
    while True:
        iteration += 1
        log(f"--- Check #{iteration} ---")

        for exp in EXPERIMENTS:
            try:
                status, action = check_experiment(exp)
                if status != "OK" and action:
                    log(f"  [{exp}] Action: {action}")
            except Exception as e:
                log(f"  [{exp}] ERROR during check: {e}")

        # 生成报告
        try:
            generate_report()
        except Exception as e:
            log(f"  Report generation error: {e}")

        log(f"--- Next check in 600s ---")
        time.sleep(600)


if __name__ == "__main__":
    main()
