"""AEF_qwen Embedding 提取引擎."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

_model_cache = {}


def load_model(model_name: str, device: str = "cuda:0") -> tuple[AEFModel, HarbinPatchDataset, dict]:
    """加载模型和数据集."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from demo.data_loader import MODEL_REGISTRY
    reg = MODEL_REGISTRY[model_name]

    cfg = load_config(reg["config"])
    device = torch.device(device)

    model = AEFModel(cfg).to(device)
    ckpt = torch.load(reg["checkpoint"], map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    result = (model, dataset, cfg)
    _model_cache[model_name] = result
    return result


def extract_patch_embedding(model, dataset, patch_idx: int,
                             valid_start_ms: float, valid_end_ms: float,
                             device: torch.device) -> np.ndarray | None:
    """提取单个 patch 的 embedding map.

    Returns:
        [D, H, W] embedding map
    """
    batch = dataset[patch_idx]

    # 修改 valid period
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)

    # 移动 batch 到 device
    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)
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

    # 推理时做 L2 normalize
    emb_map = output.embedding_map  # [1, D, H, W]
    emb_map = F.normalize(emb_map, p=2, dim=1)
    return emb_map[0].cpu().numpy()  # [D, H, W]


def compute_before_after(model, dataset, patch_idx: int,
                          before_window: tuple[float, float],
                          after_window: tuple[float, float],
                          device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """计算 before/after 两个时间窗口的 embedding.

    Returns:
        emb_before: [D, H, W], emb_after: [D, H, W]
    """
    emb_before = extract_patch_embedding(model, dataset, patch_idx,
                                          before_window[0], before_window[1], device)
    emb_after = extract_patch_embedding(model, dataset, patch_idx,
                                         after_window[0], after_window[1], device)
    return emb_before, emb_after


def cosine_distance_map(emb_before: np.ndarray, emb_after: np.ndarray) -> np.ndarray:
    """逐像素 cosine distance. [D, H, W] → [H, W]."""
    D, H, W = emb_before.shape
    flat_before = emb_before.reshape(D, -1)  # [D, H*W]
    flat_after = emb_after.reshape(D, -1)

    # 归一化
    norm_before = np.linalg.norm(flat_before, axis=0, keepdims=True)
    norm_after = np.linalg.norm(flat_after, axis=0, keepdims=True)
    flat_before = flat_before / np.maximum(norm_before, 1e-8)
    flat_after = flat_after / np.maximum(norm_after, 1e-8)

    # cosine similarity
    cos_sim = np.sum(flat_before * flat_after, axis=0)  # [H*W]
    cos_dist = (1.0 - cos_sim) / 2.0  # [0, 1]
    return cos_dist.reshape(H, W)


def l2_distance_map(emb_before: np.ndarray, emb_after: np.ndarray) -> np.ndarray:
    """逐像素 L2 distance. [D, H, W] → [H, W]."""
    diff = emb_before - emb_after  # [D, H, W]
    return np.linalg.norm(diff, axis=0)  # [H, W]


def embedding_map_to_pca_rgb(emb_map: np.ndarray,
                              global_pca=None,
                              norm_percentile: tuple[float, float] = (2, 98)) -> np.ndarray:
    """将 embedding map 转为 PCA RGB 可视化.

    Returns:
        [H, W, 3] uint8 RGB image
    """
    from sklearn.decomposition import PCA

    D, H, W = emb_map.shape
    flat = emb_map.reshape(D, -1).T  # [H*W, D]

    if global_pca is None:
        pca = PCA(n_components=3)
        pc = pca.fit_transform(flat)
    else:
        pc = global_pca.transform(flat)

    # Percentile 归一化到 0-255
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(3):
        ch = pc[:, c]
        lo, hi = np.percentile(ch, norm_percentile)
        if hi > lo:
            ch_norm = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
        else:
            ch_norm = 128
        rgb[:, :, c] = ch_norm.reshape(H, W)

    return rgb
