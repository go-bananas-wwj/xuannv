"""配置系统 — YAML 驱动的数据类."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    name: str = "aef_qwen_v1"
    seed: int = 42
    output_dir: str = "/workspace/outputs/aef_qwen_v1"


@dataclass
class DataConfig:
    dataset_type: str = "harbin_patches"
    manifest_path: str = ""
    stats_dir: str = ""
    batch_size: int = 2
    num_workers: int = 4
    image_size: int = 128
    max_frames: int = 32
    num_samples: int = 424
    num_classes: int = 11
    input_dim: int = 6
    metadata_dim: int = 4
    num_input_sources: int = 7
    num_target_sources: int = 7
    preload: bool = True
    random_target_frame: bool = True
    spatial_augmentation: bool = True
    # 时序窗口增强
    temporal_window_augmentation: bool = True
    temporal_window_prob: float = 0.5
    temporal_window_min_frames: int = 4
    temporal_window_max_frames: int = 24
    # 云过滤
    cloud_filter_threshold: float = 0.0
    # 方差加权采样
    variance_weighted: bool = False
    filter_2025_monthly: bool = False
    # ★ OlmoEarth 离线 teacher tokens 根目录 (蒸馏用)
    olmoearth_tokens_root: str = None
    source_channels: dict = field(default_factory=dict)
    merge_hr_into_lr: bool = False
    input_sources: list | None = None
    target_sources: list | None = None
    # 双窗口采样模式
    window_mode: str = "random_split"
    # non_overlap 参数
    non_overlap_min_frames: int = 4
    non_overlap_max_frames: int = 12
    non_overlap_min_gap_ms: int = 15552000000
    # mixed_scale 参数
    mixed_scale_long_prob: float = 0.5
    mixed_scale_short_prob: float = 0.5
    mixed_scale_short_max_gap_ms: int = 7776000000
    mixed_scale_long_min_gap_ms: int = 15552000000
    # 跨时相掩码重建
    ct_mask_ratio: float = 0.0
    ct_mask_patch_size: int = 8
    # Round 2: 跨时相重建配置
    cross_temporal: bool = False
    cross_temporal_prob: float = 0.0
    cross_temporal_min_gap_months: int = 2
    # 快速验证: 随机采样部分 patch
    max_patches: int | None = None
    # 精确指定 patch 列表（优先级高于 max_patches）
    patch_list: list[str] | None = None
    # V14: 多区域混合训练 manifest
    multi_region_manifest: str | None = None


@dataclass
class ModelConfig:
    input_dim: int = 6
    stem_dim: int = 128
    precision_dim: int = 256
    time_dim: int = 256
    space_dim: int = 256
    embedding_dim: int = 128
    time_code_dim: int = 64
    window_code_dim: int = 64
    relative_time_code_dim: int = 16
    num_blocks: int = 8
    num_blocks_disable_space: int = 0  # 前 N 层禁用 Space path
    num_heads: int = 8
    vmf_kappa: float = 2000.0
    bottleneck_noise_scale: float = 0.02
    reconstruction_channels: int = 6
    metadata_dim: int = 4
    num_sensor_types: int = 16
    gradient_checkpointing: bool = True
    per_source_decoders: bool = False
    decoder_hidden_mult: int = 1
    source_channels: dict = field(default_factory=dict)
    stem_channels: int = 6
    # 反坍缩
    skip_l2_norm_training: bool = True
    spatial_dropout_rate: float = 0.0
    # P1: 2D Sincos 位置编码
    use_2d_pos_enc: bool = False
    pos_enc_2d_height: int = 8
    pos_enc_2d_width: int = 8
    # ★ OlmoEarth 蒸馏投影头
    use_distill_head: bool = False
    distill_hidden_dim: int = 512
    distill_teacher_dim: int = 768
    distill_teacher_spatial_size: int = 32


@dataclass
class TrainingConfig:
    epochs: int = 400
    gradient_accumulation_steps: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    teacher_dropout_rate: float = 0.1
    student_frame_drop_rate: float = 0.4
    student_source_drop_rate: float = 0.3
    student_front_drop_prob: float = 0.2
    student_back_drop_prob: float = 0.2
    # 损失权重
    reconstruction_weight: float = 1.0
    source_recon_weights: list = None  # 源特定重建权重
    uniformity_weight: float = 0.3
    batch_uniformity_weight: float = 0.0
    uniformity_adaptive: bool = True
    consistency_weight: float = 0.15
    classification_weight: float = 0.05
    coding_rate_weight: float = 0.0
    text_contrastive_weight: float = 0.0
    # 反坍缩四件套
    orthogonality_weight: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.1
    decorrelation_weight: float = 0.01
    # V35: AEF Batch Rotation Uniformity
    aef_batch_uniformity_weight: float = 0.0
    spatial_dropout_rate: float = 0.5
    bottleneck_cls_weight: float = 0.2
    aux_classification_weight: float = 0.1
    # 时序对比损失
    temporal_contrastive_weight: float = 0.0
    temporal_contrastive_temperature: float = 0.1
    temporal_loss_type: str = "hinge"
    temporal_margin: float = 1.0
    l2_temporal_weight: float = 0.0
    pixel_temporal_weight: float = 0.0
    pixel_temporal_samples: int = 16
    # Temporal Magnitude Loss (V5)
    temporal_magnitude_weight: float = 0.0
    temporal_max_gap_ms: int = 15552000000
    # V6: Pixel-level temporal losses
    temporal_cosine_pixel_weight: float = 0.0
    temporal_cosine_pixel_temperature: float = 0.05
    temporal_use_spatial_weight: bool = False  # Round 9: 空间感知 temporal loss
    pixel_temporal_info_nce_weight: float = 0.0
    pixel_temporal_info_nce_temperature: float = 0.1
    pixel_temporal_info_nce_samples: int = 16
    # V6: Spatial uniformity
    spatial_uniformity_weight: float = 0.0
    spatial_uniformity_samples: int = 256
    # V6.5: Gap-aware temporal
    gap_aware_temporal_weight: float = 0.0
    gap_aware_temporal_temperature: float = 0.05
    # V9: Gap-aware temporal cosine (轻量)
    temporal_gap_aware_weight: float = 0.0
    temporal_gap_aware_weight_end: float = 0.02
    temporal_gap_aware_weight_ramp_epochs: int = 50
    temporal_gap_aware_warmup_epochs: int = 30
    temporal_gap_max_months: int = 12
    # V9.5: Pixel change supervision
    pixel_change_supervision_weight: float = 0.0
    pixel_change_supervision_warmup_epochs: int = 40
    pixel_change_threshold: float = 0.1
    # Active Dimensions 阈值（pre-norm 空间默认 0.15，L2 空间可用 0.05）
    active_dims_threshold: float = 0.15
    # V10: Change consistency supervision
    change_consistency_weight: float = 0.0
    change_consistency_warmup_epochs: int = 40
    change_consistency_threshold: float = 0.1
    # 预归一化 uniformity
    pre_norm_uniform_weight: float = 3.0
    encoder_uniform_weight: float = 2.0
    # V13 实验变体开关
    use_spatial_uniformity: bool = False
    use_pre_norm_uniform: bool = False
    use_spatial_vicreg: bool = False
    use_spatial_raw_uniformity: bool = False
    use_l2_space_vicreg: bool = False
    vicreg_min_std: float = 1.0
    # V12 样本间方差
    inter_variance_weight: float = 0.0
    inter_variance_min_std: float = 0.3
    # V19 Inter-Patch InfoNCE (NT-Xent)，防止方向坍缩
    inter_patch_infonce_weight: float = 0.0
    inter_patch_infonce_temperature: float = 0.1
    # V22 Inter-Patch Decorrelation (Barlow Twins on gathered_pre)，防止维度坍缩
    inter_decorr_weight: float = 0.0
    # V23 直接 erank 最大化（SVD 奇异值熵），N<D 时唯一可靠的维度坍缩对抗
    erank_loss_weight: float = 0.0
    # Warmup
    recon_warmup_epochs: int = 20
    warmup_epochs: int = 10
    lr_schedule: str = "cosine_no_restart"
    lr_min: float = 1e-6
    # Kappa 渐进
    kappa_start: float = 50.0
    kappa_end: float = 500.0
    kappa_warmup_epochs: int = 100
    # 其他
    vicreg_weight: float = 0.0
    koleo_weight: float = 0.0
    ct_reconstruction_weight: float = 0.0
    dino_weight: float = 0.0
    teacher_momentum: float = 0.996
    save_every: int = 20
    max_steps_per_epoch: int = 0  # 0 = 不限制，使用全部数据
    expander_dim: int = 0
    # 检查点
    save_best_balanced: bool = True
    best_balanced_uniform_min: float = -0.6
    best_balanced_uniform_max: float = 0.3
    early_stop_patience: int = 150
    eval_every: int = 10  # 每隔多少 epoch 跑一次 kNN 下游探针评估
    save_every: int = 50
    checkpoint_interval: int = 20
    # EMA Teacher
    teacher_momentum: float = 0.996
    # DINO
    dino_weight: float = 0.0
    # VICReg + KoLeo
    vicreg_weight: float = 0.0
    koleo_weight: float = 0.0
    vicreg_lambda_var: float = 1.0
    vicreg_lambda_cov: float = 0.04
    vicreg_temporal_dropout: float = 0.15
    expander_dim: int = 0
    # 跨时相掩码重建
    ct_reconstruction_weight: float = 0.0
    # LMIM：潜在空间掩码预测损失（OlmoEarth/AnySat JEPA 风格）
    lmim_weight: float = 0.0
    # 球面 Uniformity Loss — L2 归一化后计算，直接防方向坍缩
    hyperspherical_uniform_weight: float = 0.0
    # Decoder Conditioning Dropout — 随机清零时间条件码，防decoder依赖时间码走捷径
    decoder_cond_dropout: float = 0.0
    # 球面方差正则 — 在 L2 归一化 embedding 上强制各维度方差 ≥ min_std（补充 pairwise_cosine 从维度角度防坍缩）
    spherical_variance_weight: float = 0.0
    spherical_variance_min_std: float = 0.1
    # ★ 实例判别 (Instance Discrimination) — 预测 patch 身份 (0 ~ N-1)
    # 坍缩时 CE = log(N)，对每个 patch 方向不同 → 真正打破坍缩的核心机制
    patch_id_loss_weight: float = 0.0
    patch_id_num_patches: int = 0  # 0 = 禁用
    # Memory Bank — 扩大 uniformity loss 的有效 batch
    # K 越大梯度越稳定，但 uniformity O(N²) 会变慢；建议 512-1024
    memory_bank_size: int = 512
    # ★ OlmoEarth 离线蒸馏
    olmoearth_spatial_distill_weight: float = 0.0
    olmoearth_global_distill_weight: float = 0.0
    distill_projector_warmup_epochs: int = 0
    backbone_lr_scale: float = 1.0  # backbone参数LR倍率，<1可减缓backbone解冻后的坍塌


@dataclass
class EvaluationConfig:
    knn_k: int = 5
    bootstrap_samples: int = 100


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pretrained: str | None = None


def _merge_dict(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge_dict(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | Path) -> Config:
    base_dir = Path(path).parent

    def _load_recursive(p: Path) -> dict:
        with open(p, "r") as f:
            raw = yaml.safe_load(f)
        base_path = raw.pop("_base_", None)
        if base_path:
            with open(p.parent / base_path, "r") as f:
                base_raw = yaml.safe_load(f)
            raw = _merge_dict(_load_recursive(p.parent / base_path), raw)
        return raw

    raw = _load_recursive(Path(path))

    cfg = Config()
    for section, values in raw.items():
        if not isinstance(values, dict):
            continue
        section_cls = {
            "experiment": ExperimentConfig,
            "data": DataConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "evaluation": EvaluationConfig,
        }.get(section)
        if section_cls:
            known = {f.name for f in section_cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in values.items() if k in known}
            setattr(cfg, section, section_cls(**filtered))
    return cfg
