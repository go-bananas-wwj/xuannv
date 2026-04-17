#!/usr/bin/env python3
"""批量运行 CV 实验，直接写日志到文件."""
from __future__ import annotations

import os
import subprocess
import sys
import time

EXPERIMENTS = [
    {
        "name": "v3_aug",
        "cmd": [
            sys.executable, "-u", "scripts/crossval_monthly_cd_head.py",
            "--head_type", "v3",
            "--hidden_dim", "64",
            "--dropout", "0.4",
            "--noise_std", "0.02",
            "--channel_dropout", "0.1",
            "--seed", "42",
            "--gpu", "6",
        ],
        "log": "/tmp/cv_v3_aug.log",
    },
    {
        "name": "v3_aug_ohem",
        "cmd": [
            sys.executable, "-u", "scripts/crossval_monthly_cd_head.py",
            "--head_type", "v3",
            "--hidden_dim", "64",
            "--dropout", "0.4",
            "--noise_std", "0.02",
            "--channel_dropout", "0.1",
            "--ohem",
            "--seed", "42",
            "--gpu", "7",
        ],
        "log": "/tmp/cv_v3_ohem.log",
    },
    {
        "name": "mc_aug",
        "cmd": [
            sys.executable, "-u", "scripts/crossval_monthly_cd_head.py",
            "--head_type", "mc",
            "--hidden_dim", "64",
            "--dropout", "0.4",
            "--noise_std", "0.02",
            "--channel_dropout", "0.1",
            "--seed", "42",
            "--gpu", "6",
        ],
        "log": "/tmp/cv_mc_aug.log",
    },
]


def run_experiment(exp: dict) -> subprocess.Popen:
    log_path = exp["log"]
    with open(log_path, "w") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {exp['name']}\n")
    
    f = open(log_path, "a")
    proc = subprocess.Popen(
        exp["cmd"],
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd="/workspace/xuannv",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(f"Started {exp['name']} (PID={proc.pid}) -> {log_path}")
    return proc


def main():
    # Run first two in parallel
    procs = []
    for exp in EXPERIMENTS[:2]:
        procs.append(run_experiment(exp))
    
    # Wait for both to finish
    for proc in procs:
        proc.wait()
    
    print("First two experiments done. Starting third...")
    
    # Run third on GPU 6
    proc3 = run_experiment(EXPERIMENTS[2])
    proc3.wait()
    
    print("All experiments complete.")
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}: {exp['log']}")


if __name__ == "__main__":
    main()
