from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT: Path | None = None


def _project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    return _PROJECT_ROOT


def _ensure_src_on_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_src_on_path()

from src.config import Config
from src.data.dataset import HarbinPatchDataset
from src.inference.engine import (
    load_backbone as _engine_load_backbone,
    extract_embedding_for_month as _engine_extract_embedding_for_month,
)
from src.models.model import AEFModel
from src.utils.device import get_device


def _resolve_model_dir(model_dir: str | Path) -> Path:
    md = Path(model_dir)
    if md.is_absolute():
        if not md.exists():
            raise FileNotFoundError(f"model 目录不存在: {md}")
        return md.resolve()
    if md.exists():
        return md.resolve()
    fallback = _project_root() / md
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"找不到 model 目录: {model_dir}（也尝试了 {fallback}）。"
        "请先运行 scripts/copy_model.sh。"
    )


def load_production_model(
    model_dir: str | Path = "production/v1_haidian/model",
    device: str = "npu:0",
) -> tuple[AEFModel, HarbinPatchDataset, Config]:
    md = _resolve_model_dir(model_dir)
    cfg_path = md / "config_multires_v1.yaml"
    ckpt_path = md / "epoch_80.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"缺少 checkpoint: {ckpt_path}。请运行 scripts/copy_model.sh。"
        )

    try:
        get_device(device_str=device)
    except Exception as exc:
        raise RuntimeError(
            f"设备 {device} 不可用。请设置 ASCEND_RT_VISIBLE_DEVICES 或改用 --device cpu。"
        ) from exc

    model, dataset, cfg = _engine_load_backbone(
        config_path=cfg_path,
        checkpoint_path=ckpt_path,
        device=device,
        eval_mode=True,
    )
    return model, dataset, cfg


def extract_embedding_for_month(
    model: AEFModel,
    dataset: HarbinPatchDataset,
    patch_id: str,
    year: int,
    month: int,
    device: str,
) -> np.ndarray:
    return _engine_extract_embedding_for_month(
        model=model,
        dataset=dataset,
        patch_id=patch_id,
        year=year,
        month=month,
        device=device,
        normalize=True,
        use_pre_norm=False,
    )


def extract_embeddings_for_patches(
    model: AEFModel,
    dataset: HarbinPatchDataset,
    patch_ids: list[str],
    year: int,
    month: int,
    device: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for pid in patch_ids:
        try:
            emb = _engine_extract_embedding_for_month(
                model=model,
                dataset=dataset,
                patch_id=pid,
                year=year,
                month=month,
                device=device,
                normalize=True,
                use_pre_norm=False,
            )
            out[pid] = emb
        except Exception as exc:
            warnings.warn(f"[backbone] {pid} {year}-{month:02d} 提取失败，跳过: {exc}")
    return out
