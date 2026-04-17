"""模型加载与实时推理引擎."""
from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, "/workspace/xuannv")

from src.inference.engine import extract_embedding_map, load_backbone


class ModelEngine:
    """按需加载模型，支持实时推理获取指定时间窗口的 embedding."""

    def __init__(self, version: str, device: str = "cuda:0"):
        self.version = version
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model: Optional[Any] = None
        self._dataset: Optional[Any] = None
        self._cfg: Optional[Any] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from demo_v2.config import get_model_info
        info = get_model_info(self.version)
        cfg_path = str(info["config"])
        ckpt_path = str(info["checkpoint"])

        if not info["has_checkpoint"]:
            raise RuntimeError(f"Checkpoint not found for {self.version}: {ckpt_path}")

        self._model, self._dataset, self._cfg = load_backbone(
            cfg_path, ckpt_path, device=self.device, eval_mode=True
        )
        print(f"[ModelEngine] Loaded {self.version} on {self.device}")

    def extract_embedding(
        self,
        patch_id: str,
        window_start_ms: float,
        window_end_ms: float,
    ) -> Optional[np.ndarray]:
        """为指定 patch 和时间窗口提取 embedding map [D, H, W]."""
        self._load()
        if self._dataset is None or patch_id not in self._dataset.patches:
            return None

        pidx = self._dataset.patches.index(patch_id)
        return extract_embedding_map(
            self._model,
            self._dataset,
            pidx,
            window_start_ms,
            window_end_ms,
            self.device,
            normalize=False,
        )

    def normalize_embedding(self, emb: np.ndarray) -> np.ndarray:
        """L2 归一化到单位球 [D, H, W]."""
        norms = np.linalg.norm(emb, axis=0, keepdims=True)
        return emb / np.maximum(norms, 1e-8)
