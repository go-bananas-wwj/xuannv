from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import load_production_model, extract_embedding_for_month


@pytest.fixture
def torch_threads():
    """临时提高 CPU 线程数，测试结束后恢复。"""
    prev = torch.get_num_threads()
    torch.set_num_threads(64)
    yield
    torch.set_num_threads(prev)


@pytest.mark.usefixtures("torch_threads")
def test_load_and_extract():
    model, dataset, cfg = load_production_model(
        model_dir=str(PROD_DIR / "model"),
        device="cpu",
    )

    emb = extract_embedding_for_month(
        model=model,
        dataset=dataset,
        patch_id="patch_000000",
        year=2026,
        month=4,
        device="cpu",
    )

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (64, 64, 64), f"unexpected embedding shape: {emb.shape}"
