#!/usr/bin/env python3
"""AEF_qwen Training Monitor + Auto Fine-tune.

Watches training log, evaluates when done, auto-starts Phase 2 if needed.
"""
import os, sys, time, subprocess, re, glob
from datetime import datetime

LOG_DIR = "/workspace/logs"
TRAIN_LOG = f"{LOG_DIR}/qwen_v1_train_v3.log"
REPORT = f"{LOG_DIR}/qwen_v1_training_report.txt"
OUTPUT_DIR = "/workspace/outputs/aef_qwen_v1"
AUF_DIR = "/workspace/xuannv"

THRESHOLDS = {
    "min_raw_unif": -3.0,
    "min_pre_unif": -3.0,
    "max_recon": 2.5,
    "max_decor": 25.0,
    "min_epoch": 300,
}

PHASE2 = {
    "decor_weight": 0.05,
    "lr": 0.00001,
    "epochs": 100,
}

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(REPORT, "a") as f:
        f.write(line + "\n")

def get_metrics():
    try:
        with open(TRAIN_LOG) as f:
            lines = f.readlines()
    except:
        return None
    for line in reversed(lines):
        if "Epoch" in line and "Recon" in line and "Traceback" not in line:
            m = {}
            for key in ["Epoch", "Recon", "RawUnif", "PreUnif", "Decor", "Var", "Orth"]:
                pat = rf'{key}:\s*([-0-9.]+)'
                match = re.search(pat, line)
                if match:
                    val = match.group(1)
                    m[key.lower()] = float(val) if key != "Epoch" else int(val.split("/")[0])
            if "epoch" in m and "recon" in m:
                return m
    return None

def is_training():
    r = subprocess.run(["pgrep", "-f", "train_ddp.*qwen_v1"], capture_output=True)
    return r.returncode == 0

def evaluate():
    log("=" * 50)
    log("训练已结束，开始评估最终效果...")
    log("=" * 50)
    m = get_metrics()
    if not m:
        log("ERROR: 无法读取最终指标！")
        return False
    log(f"\n--- 最终训练指标 (Epoch {m['epoch']}) ---")
    log(f"  Recon:   {m['recon']:.4f}  (目标 < {THRESHOLDS['max_recon']})")
    log(f"  RawUnif: {m['rawunif']:.4f}  (目标 < {THRESHOLDS['min_raw_unif']})")
    log(f"  PreUnif: {m['preunif']:.4f}  (目标 < {THRESHOLDS['min_pre_unif']})")
    log(f"  Decor:   {m['decor']:.4f}  (目标 < {THRESHOLDS['max_decor']})")
    log(f"  Var:     {m['var']:.4f}")
    log(f"  Orth:    {m['orth']:.6f}\n")

    reasons = []
    if m["epoch"] < THRESHOLDS["min_epoch"]:
        reasons.append(f"训练未完成 ({m['epoch']}/{THRESHOLDS['min_epoch']} epoch)")
    if m["rawunif"] > THRESHOLDS["min_raw_unif"]:
        reasons.append(f"RawUnif={m['rawunif']:.2f} 不够分散 (目标 < {THRESHOLDS['min_raw_unif']})")
    if m["preunif"] > THRESHOLDS["min_pre_unif"]:
        reasons.append(f"PreUnif={m['preunif']:.2f} 不够分散 (目标 < {THRESHOLDS['min_pre_unif']})")
    if m["recon"] > THRESHOLDS["max_recon"]:
        reasons.append(f"Recon={m['recon']:.2f} 重建质量不足 (目标 < {THRESHOLDS['max_recon']})")

    if reasons:
        log("❌ 训练效果不达标，原因:")
        for r in reasons:
            log(f"  - {r}")
        return False
    else:
        log("✅ 训练效果达标！无需微调。")
        return True

def start_phase2():
    log("=" * 50)
    log("启动 Phase 2 自动微调...")
    log("=" * 50)

    ckpts = sorted(glob.glob(f"{OUTPUT_DIR}/epoch_*.pt"), key=os.path.getmtime, reverse=True)
    if not ckpts:
        log("ERROR: 找不到 checkpoint！")
        return
    best_ckpt = ckpts[0]
    # Prefer best.pt if exists
    best_pt = f"{OUTPUT_DIR}/best.pt"
    if os.path.exists(best_pt):
        best_ckpt = best_pt
    log(f"使用 checkpoint: {best_ckpt}")

    p2c = f"{AUF_DIR}/configs/qwen_v1_phase2_auto.yaml"
    with open(p2c, "w") as f:
        f.write(f"""# Phase 2 自动微调 (监控脚本生成)
_base_: qwen_v1_scenes.yaml
training:
  decorrelation_weight: {PHASE2['decor_weight']}
  lr: {PHASE2['lr']}
  epochs: {PHASE2['epochs']}
  warmup_epochs: 5
  recon_warmup_epochs: 5
  save_every: 20
  checkpoint_interval: 20
  early_stop_patience: 50
  best_balanced_uniform_min: -4.5
  best_balanced_uniform_max: -2.0
""")

    p2log = f"{LOG_DIR}/qwen_v1_phase2_auto.log"
    est_h = PHASE2["epochs"] * 70 // 3600
    est_m = (PHASE2["epochs"] * 70 % 3600) // 60
    log(f"Phase 2 配置: decor={PHASE2['decor_weight']}, lr={PHASE2['lr']}, epochs={PHASE2['epochs']}")
    log(f"预计耗时: ~{est_h}h{est_m}m")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    cmd = ["torchrun", "--nproc_per_node=3", f"{AUF_DIR}/scripts/train_ddp.py",
           "--config", p2c, "--resume", best_ckpt, "--save-every", "20", "--warmup-epochs", "5"]
    with open(p2log, "w") as lf:
        subprocess.Popen(cmd, cwd=AUF_DIR, env=env, stdout=lf, stderr=lf,
                        start_new_session=True)
    log(f"Phase 2 微调已启动 (日志: {p2log})")
    log("=" * 50)

def main():
    log("=" * 50)
    log("AEF_qwen 训练监控脚本启动")
    log(f"训练日志: {TRAIN_LOG}")
    log("=" * 50)
    log("")

    if not is_training():
        log("训练未运行，直接评估...")
        if not evaluate():
            start_phase2()
        return

    last_epoch = 0
    fails = 0
    while True:
        if not is_training():
            log("训练进程已结束")
            break

        m = get_metrics()
        if not m:
            fails += 1
            if fails > 5:
                log("WARNING: 连续5次无法读取指标，训练可能卡死")
                break
            time.sleep(60)
            continue

        fails = 0
        epoch = m["epoch"]
        if epoch != last_epoch:
            if epoch % 10 == 0:
                log(f"E{epoch}/400 Recon={m['recon']:.2f} RawUnif={m['rawunif']:.2f} "
                    f"PreUnif={m['preunif']:.2f} Decor={m['decor']:.1f} Var={m['var']:.3f}")
            if m["rawunif"] > -1.0:
                log(f"⚠️ WARNING: RawUnif={m['rawunif']:.2f} 接近零，可能坍缩！")
            last_epoch = epoch

        time.sleep(60)

    if not evaluate():
        start_phase2()
    else:
        log("训练效果良好，无需微调。")
        log(f"最佳模型: {OUTPUT_DIR}/best.pt")
    log("监控完成。")

if __name__ == "__main__":
    main()
