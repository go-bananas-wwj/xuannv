"""配置系统 — 简单dataclass配置."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    image_size: int = 128
    patch_size: int = 8
    embed_dim: int = 512
    num_encoder_layers: int = 12
    num_decoder_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    output_dim: int = 64
    dropout: float = 0.1
    use_gradient_checkpointing: bool = True


@dataclass
class MaskingConfig:
    modality_probs: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.25, 0.4])
    # 对应: NOT_SELECTED, ENCODE_ONLY, DECODE_ONLY, ENCODE_AND_DECODE
    temporal_keep_ratio: float = 0.5
    spatial_visible_ratio: float = 0.25
    channel_keep_ratio: float = 0.5
    patch_size: int = 8


@dataclass
class DataConfig:
    num_patches: int = 320
    data_root: str = "data_raw/haidian/scenes"
    planet_root: str = "data_raw/beijing/planetscene"
    stats_dir: str = "statistics/haidian"
    cache_dir: str = "haidian_recon/.cache"
    anchor_source: str = "tianyi_sar"
    temporal_window_days: float = 5.5
    image_size: int = 128
    batch_size: int = 8
    num_workers: int = 0  # NPU上先设为0避免死锁
    sources: List[Dict] = field(default_factory=lambda: [
        {"name": "tianyi_sar", "channels": 1, "stats_path": "statistics/haidian/tianyi_sar_stats.json"},
        {"name": "s2", "channels": 6, "stats_path": "statistics/haidian/s2_stats.json"},
        {"name": "landsat", "channels": 6, "stats_path": "statistics/haidian/landsat_stats.json"},
        {"name": "planet", "channels": 4, "stats_path": "statistics/haidian/planet_stats.json"},
    ])


@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 1.0e-4
    lr_min: float = 1.0e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    w_recon: float = 1.0
    w_distill: float = 0.02
    w_uniform: float = 0.01
    w_spatial_uniform: float = 0.005
    aef_checkpoint: str | None = None
    aef_config: str = "configs/config_haidian_v41.yaml"
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    save_every: int = 10
    eval_every: int = 10
    log_every: int = 50


@dataclass
class Config:
    experiment_name: str = "hre_haidian_recon"
    output_dir: str = "outputs/hre_haidian_recon"
    seed: int = 42
    device: str = "npu"
    backend: str = "hccl"
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
