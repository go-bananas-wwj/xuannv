from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import changedetection


@pytest.fixture(autouse=True)
def _torch_threads_and_cleanup():
    prev = torch.get_num_threads()
    torch.set_num_threads(64)
    out_dir = PROD_DIR / "outputs" / "test_changedetection"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    yield
    torch.set_num_threads(prev)


@pytest.mark.slow
def test_june_change_score_shape():
    # 选择包含 june 变化标注的 patch，确保 period_results 能计算出 AUC。
    metrics = changedetection.run_change_detection(
        model_dir=str(PROD_DIR / "model"),
        output_dir=str(PROD_DIR / "outputs" / "test_changedetection"),
        device="cpu",
        periods=["june"],
        annot_dir="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件",
        grid_path="/workspace/index/harbin/grid/harbin_grid.geojson",
        patch_ids=["patch_000033", "patch_000039", "patch_000040"],
    )

    out_path = PROD_DIR / "outputs" / "test_changedetection" / "change_score_june.npz"
    assert out_path.exists(), f"change score file not found: {out_path}"

    data = np.load(out_path)
    assert "patch_ids" in data
    assert "scores" in data
    assert data["scores"].ndim == 3

    metrics_path = PROD_DIR / "outputs" / "test_changedetection" / "metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert "june" in saved["periods"]
