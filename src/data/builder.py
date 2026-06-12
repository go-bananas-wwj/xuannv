"""DataLoader 构建工厂."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler, default_collate

from src.config import Config
from src.data.dataset import HarbinPatchDataset
from src.data.multi_region_dataset import MultiRegionPatchDataset


def _collate_tensor_list(key: str, batch: list[dict]) -> list[torch.Tensor]:
    """对 List[Tensor] 类型的字段按索引堆叠."""
    n_items = len(batch[0][key])
    return [torch.stack([b[key][i] for b in batch]) for i in range(n_items)]


def multires_collate_fn(batch: list[dict]) -> dict:
    """支持 source_frames / target_images 为 List[Tensor] 的 collate."""
    if not batch:
        return {}
    first = batch[0]
    collated: dict = {}
    for key in first.keys():
        val = first[key]
        if key in ("source_frames", "target_images") and isinstance(val, list):
            collated[key] = _collate_tensor_list(key, batch)
        else:
            collated[key] = default_collate([b[key] for b in batch])
    return collated


def build_dataloader(
    cfg: Config,
    training: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """构建 HarbinPatchDataset / MultiRegionPatchDataset 的 DataLoader.

    Args:
        cfg: 项目配置.
        training: 是否为训练模式 (影响数据增强和 shuffle).
        distributed: 是否使用 DistributedSampler.
        world_size: 分布式总进程数.
        rank: 当前进程 rank.

    Returns:
        PyTorch DataLoader.
    """
    if getattr(cfg.data, 'multi_region_manifest', None):
        dataset = MultiRegionPatchDataset(cfg)
    else:
        dataset = HarbinPatchDataset(cfg)
    dataset.training = training
    # 透传 non_overlap 参数 (如果 config 中显式设置)
    if training:
        if hasattr(cfg.data, 'non_overlap_min_frames'):
            dataset._min_window_frames = cfg.data.non_overlap_min_frames
        if hasattr(cfg.data, 'non_overlap_max_frames'):
            dataset._max_window_frames = cfg.data.non_overlap_max_frames
        if hasattr(cfg.data, 'non_overlap_min_gap_ms'):
            dataset._min_window_gap_ms = cfg.data.non_overlap_min_gap_ms
        # window_mode 已在 dataset.__init__ 中设置，training 模式保持一致

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training) if distributed else None

    # Data is pre-loaded in memory (dataset._cache), no need for background workers
    num_workers = 0
    use_multires = getattr(cfg.data, "use_multires", False)
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and training),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=multires_collate_fn if use_multires else None,
    )
