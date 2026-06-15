"""统一推理引擎.

封装 Backbone 加载、Embedding 提取、CD Head 加载等公共逻辑，
供 scripts/ 复用，避免重复实现.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config, load_config
from src.data.dataset import HarbinPatchDataset
from src.models.heads import ChangeDetectionHeadV3
from src.models.model import AEFModel
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device


def load_backbone(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
) -> tuple[AEFModel, HarbinPatchDataset, Config]:
    """加载 AEFModel backbone 及对应数据集配置.

    Args:
        config_path: YAML 配置文件路径 (支持相对路径).
        checkpoint_path: PyTorch checkpoint 路径.
        device: 目标设备，默认自动选择 CUDA.
        eval_mode: 是否调用 model.eval() 并关闭数据增强.

    Returns:
        (model, dataset, cfg)
    """
    cfg = load_config(config_path)
    device = get_device(device_str=str(device) if device else None)
    model = AEFModel(cfg).to(device)

    state = load_checkpoint(checkpoint_path, device=device, keys=("model_state_dict",))
    model.load_state_dict(state, strict=False)

    if eval_mode:
        model.eval()

    dataset = HarbinPatchDataset(cfg)
    if eval_mode:
        dataset.training = False
        dataset._spatial_augmentation = False

    return model, dataset, cfg


def load_cd_head(
    head_path: str | Path,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
) -> ChangeDetectionHeadV3:
    """加载 ChangeDetectionHeadV3.

    Args:
        head_path: Head checkpoint 路径 (由 train_monthly_cd_head.py 保存的格式).
        device: 目标设备.
        eval_mode: 是否调用 eval().

    Returns:
        加载后的 ChangeDetectionHeadV3 实例.
    """
    device = get_device(device_str=str(device) if device else None)
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    head = ChangeDetectionHeadV3(
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg.get("dropout", 0.3),
    )
    head.load_state_dict(ckpt["cd_head"])
    head.to(device)
    if eval_mode:
        head.eval()
    return head


def _find_monthly_idx(dataset: HarbinPatchDataset, patch_id: str, year: int, month: int) -> int:
    """在 dataset.monthly_samples 中查找 (patch_id, year, month) 对应的索引."""
    for idx, (pid, y, m) in enumerate(dataset.monthly_samples):
        if pid == patch_id and y == year and m == month:
            return idx
    raise ValueError(f"未找到 {patch_id} 在 {year}-{month:02d} 的月度样本")


def extract_embedding_for_month(
    model: AEFModel,
    dataset: HarbinPatchDataset,
    patch_id: str,
    year: int,
    month: int,
    device: str | torch.device,
    normalize: bool = True,
    use_pre_norm: bool = False,
) -> np.ndarray:
    """为指定 patch 和具体年月提取 embedding map [D, H, W].

    Args:
        model: 已加载的 AEFModel.
        dataset: 对应的 HarbinPatchDataset.
        patch_id: Patch ID (如 "patch_000000").
        year: 年份 (如 2025).
        month: 月份 (1-12).
        device: 运行设备.
        normalize: 是否对 embedding 做 L2 归一化.
        use_pre_norm: 是否使用 pre_norm_map.

    Returns:
        numpy array of shape [D, H, W].
    """
    idx = _find_monthly_idx(dataset, patch_id, year, month)
    device = get_device(device_str=str(device) if device else None)
    batch = dataset[idx]

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            # multires: source_frames/target_images 是 list[Tensor]，需要给每个 tensor 加 batch 维
            batch_dev[k] = [t.unsqueeze(0).to(device) for t in v]
        else:
            batch_dev[k] = v

    with torch.no_grad():
        output = model(
            source_frames=batch_dev["source_frames"],
            source_timestamps_ms=batch_dev["source_timestamps_ms"],
            source_frame_mask=batch_dev["source_frame_mask"],
            source_input_mask=batch_dev["source_input_mask"],
            source_type_ids=batch_dev["source_type_ids"],
            valid_start_ms=batch_dev["valid_start_ms"],
            valid_end_ms=batch_dev["valid_end_ms"],
            target_relative_time=batch_dev["target_relative_time"],
            target_metadata=batch_dev["target_metadata"],
        )

    if use_pre_norm and output.pre_norm_map is not None:
        emb = output.pre_norm_map  # [1, D, H, W] — 原始幅度，无 VMF 噪声
    else:
        emb = output.embedding_map  # [1, D, H, W]
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()


def extract_embedding_map(
    model: AEFModel,
    dataset: HarbinPatchDataset,
    patch_idx: int,
    window_start_ms: float,
    window_end_ms: float,
    device: str | torch.device,
    normalize: bool = True,
    use_pre_norm: bool = False,
) -> np.ndarray:
    """为指定月度样本索引和时间窗口提取 embedding map [D, H, W].

    ⚠️ 注意: patch_idx 是 monthly_samples 中的索引 (0..N_months-1)，
    不是 dataset.patches 中的索引 (0..N_patches-1)。
    如需通过 patch_id + year + month 查找，请使用 extract_embedding_for_month().

    Args:
        model: 已加载的 AEFModel.
        dataset: 对应的 HarbinPatchDataset.
        patch_idx: 月度样本在 dataset.monthly_samples 中的索引.
        window_start_ms: 窗口起始时间戳 (ms).
        window_end_ms: 窗口结束时间戳 (ms).
        device: 运行设备.
        normalize: 是否对 embedding 做 L2 归一化.
        use_pre_norm: 是否使用 pre_norm_map.

    Returns:
        numpy array of shape [D, H, W].
    """
    device = get_device(device_str=str(device) if device else None)
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(window_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(window_end_ms, dtype=torch.float64)

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            # multires: source_frames/target_images 是 list[Tensor]，需要给每个 tensor 加 batch 维
            batch_dev[k] = [t.unsqueeze(0).to(device) for t in v]
        else:
            batch_dev[k] = v

    with torch.no_grad():
        output = model(
            source_frames=batch_dev["source_frames"],
            source_timestamps_ms=batch_dev["source_timestamps_ms"],
            source_frame_mask=batch_dev["source_frame_mask"],
            source_input_mask=batch_dev["source_input_mask"],
            source_type_ids=batch_dev["source_type_ids"],
            valid_start_ms=batch_dev["valid_start_ms"],
            valid_end_ms=batch_dev["valid_end_ms"],
            target_relative_time=batch_dev["target_relative_time"],
            target_metadata=batch_dev["target_metadata"],
        )

    if use_pre_norm and output.pre_norm_map is not None:
        emb = output.pre_norm_map
    else:
        emb = output.embedding_map
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()


def run_change_detection(
    head: ChangeDetectionHeadV3,
    emb_before: np.ndarray,
    emb_after: np.ndarray,
    device: str | torch.device,
) -> np.ndarray:
    """使用 CD Head 计算变化概率图.

    Args:
        head: 已加载的 ChangeDetectionHeadV3.
        emb_before: [D, H, W] 前期 embedding.
        emb_after: [D, H, W] 后期 embedding.
        device: 运行设备.

    Returns:
        [H, W] 概率图.
    """
    device = get_device(device_str=str(device) if device else None)
    with torch.no_grad():
        eb = torch.from_numpy(emb_before).unsqueeze(0).float().to(device)
        ea = torch.from_numpy(emb_after).unsqueeze(0).float().to(device)
        logits = head(eb, ea).squeeze(1)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs
