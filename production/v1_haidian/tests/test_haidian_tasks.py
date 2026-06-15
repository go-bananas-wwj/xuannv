from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import haidian_tasks


@pytest.fixture
def torch_threads():
    """将 PyTorch CPU 线程数临时设为 64，测试结束后恢复。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(64)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(autouse=True)
def clean_test_outputs():
    """每次测试前清理测试输出目录。"""
    out_dir = PROD_DIR / "outputs" / "test_haidian_tasks"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    yield


@pytest.mark.usefixtures("torch_threads")
@pytest.mark.slow
def test_gongdi_bitemporal_linear():
    label_dir = Path("/workspace/xuannv/haidian_label/labeljson")
    # 取前 3 个包含 gongdi 正例标注的 patch，确保分层划分后训练集/测试集均含正例。
    all_pids = haidian_tasks.discover_labeled_patches(label_dir)
    pids = [
        p
        for p in all_pids
        if haidian_tasks.load_label_json(
            label_dir / f"{p}_20260430_rgb_uint8.json"
        )["gongdi"].any()
    ][:3]

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

    assert metrics.get("skipped") is not True, f"task skipped: {metrics}"
    assert "auc" in metrics, f"metrics missing auc: {metrics}"
    assert metrics["auc"] > 0.5, f"AUC too low: {metrics['auc']}"

    out_path = PROD_DIR / "outputs" / "test_haidian_tasks" / "gongdi" / "metrics.json"
    assert out_path.exists(), f"metrics file not found: {out_path}"
    saved = json.loads(out_path.read_text())
    assert saved["task"] == "gongdi"
