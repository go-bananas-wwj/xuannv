from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import load_production_model, extract_embedding_for_month


@pytest.mark.slow
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
    assert emb.dtype == np.float32, f"unexpected embedding dtype: {emb.dtype}"
    assert np.isfinite(emb).all(), "embedding contains non-finite values"

    # Embeddings are L2-normalized along the channel (D) dimension.
    # Zero/masked pixels may remain zero, so only assert non-zero pixels are unit norm.
    norms = np.linalg.norm(emb, axis=0)
    nonzero = norms > 0
    assert nonzero.any(), "embedding is all zeros"
    assert np.allclose(norms[nonzero], 1.0, atol=0.01), (
        f"non-zero embedding pixels are not L2-normalized: {norms[nonzero][:5]}"
    )
