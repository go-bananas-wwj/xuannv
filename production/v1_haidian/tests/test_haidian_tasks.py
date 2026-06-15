from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# 限制 CPU 线程数：默认 192 线程在该 CPU 上并发开销过大，
# 导致 5 patch 双时相提取超过 300s 前台超时；64 线程可在 ~260s 内完成。
torch.set_num_threads(64)

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import haidian_tasks


def test_gongdi_bitemporal_linear():
    label_dir = Path("/workspace/xuannv/haidian_label/labeljson")
    pids = haidian_tasks.discover_labeled_patches(label_dir)[:5]

    metrics = haidian_tasks.run_task(
        task_name="gongdi",
        model_dir=str(PROD_DIR / "model"),
        label_dir=str(label_dir),
        output_dir=str(PROD_DIR / "outputs" / "test_haidian_tasks"),
        device="cpu",
        mode="bitemporal",
        classifier="linear",
        patch_ids=pids,
    )

    assert "auc" in metrics, f"metrics missing auc: {metrics}"
    out_path = PROD_DIR / "outputs" / "test_haidian_tasks" / "gongdi" / "metrics.json"
    assert out_path.exists(), f"metrics file not found: {out_path}"
    saved = json.loads(out_path.read_text())
    assert saved["task"] == "gongdi"
