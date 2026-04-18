"""DataLoader 构建工厂."""
from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler

from src.config import Config
from src.data.dataset import HarbinPatchDataset


def build_dataloader(
    cfg: Config,
    training: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """构建 HarbinPatchDataset 的 DataLoader.

    Args:
        cfg: 项目配置.
        training: 是否为训练模式 (影响数据增强和 shuffle).
        distributed: 是否使用 DistributedSampler.
        world_size: 分布式总进程数.
        rank: 当前进程 rank.

    Returns:
        PyTorch DataLoader.
    """
    dataset = HarbinPatchDataset(cfg)
    dataset.training = training
    if training:
        dataset._non_overlapping_windows = getattr(cfg.data, 'non_overlapping_windows', False)
        dataset._min_window_frames = getattr(cfg.data, 'min_window_frames', 4)
        dataset._max_window_frames = getattr(cfg.data, 'max_window_frames', 12)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training) if distributed else None

    # Data is pre-loaded in memory (dataset._cache), no need for background workers
    num_workers = 0
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and training),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
    )
