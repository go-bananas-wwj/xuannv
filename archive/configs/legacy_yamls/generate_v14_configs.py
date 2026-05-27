#!/usr/bin/env python3
"""生成V14实验配置文件."""
import os

BASE_CONFIG = """data:
  batch_size: 4
  cross_temporal: false
  cross_temporal_min_gap_months: 2
  cross_temporal_prob: 0.0
  dataset_type: harbin_patches
  image_size: 128
  input_dim: 6
  manifest_path: /workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered
  max_frames: 32
  metadata_dim: 4
  num_classes: 11
  num_input_sources: 3
  num_samples: 1133
  num_target_sources: 4
  num_workers: 0
  preload: true
  random_target_frame: true
  source_channels:
    landsat: 6
    s1: 2
    s2: 6
  spatial_augmentation: true
  stats_dir: /workspace/statistics/harbin_scenes
  target_sources:
  - loss_type: 0
    name: s2
    out_channels: 6
    sensor_src: s2
  - loss_type: 0
    name: s1
    out_channels: 2
    sensor_src: s1
  - loss_type: 0
    name: landsat
    out_channels: 6
    sensor_src: landsat
  - loss_type: 0
    name: dem
    out_channels: 1
    sensor_src: null
  temporal_window_augmentation: true
  temporal_window_max_frames: 24
  temporal_window_min_frames: 4
  temporal_window_prob: 0.5
  multi_region_manifest: /workspace/xuannv/configs/multi_region_manifest.json
  cloud_filter_threshold: 0.5
  variance_weighted: false
  filter_2025_monthly: false
evaluation:
  bootstrap_samples: 100
  knn_k: 5
experiment:
  name: {name}
  output_dir: /workspace/outputs/exp_v14_0518/{name}
  seed: {seed}
model:
  bottleneck_noise_scale: 0.02
  decoder_hidden_mult: 1
  embedding_dim: 64
  gradient_checkpointing: true
  input_dim: 6
  metadata_dim: 4
  num_blocks: 8
  num_heads: 8
  num_sensor_types: 16
  per_source_decoders: true
  precision_dim: 128
  relative_time_code_dim: 16
  skip_l2_norm_training: false
  space_dim: 256
  stem_dim: 128
  time_code_dim: 64
  time_dim: 256
  vmf_kappa: {kappa}
  window_code_dim: 64
training:
  batch_uniformity_weight: {batch_unif}
  classification_weight: 0.0
  coding_rate_weight: {cr_weight}
  consistency_weight: {consist}
  covariance_weight: {cov_weight}
  decorr_weight: 0.0
  early_stop_patience: 20
  epochs: {epochs}
  grad_clip_norm: 1.0
  gradient_accumulation_steps: {accum_steps}
  lr: 0.0001
  lr_min: 1.0e-06
  lr_schedule: cosine_no_restart
  max_steps_per_epoch: 200
  orthogonality_weight: 0.0
  pre_norm_uniform_weight: 0.0
  recon_warmup_epochs: 0
  reconstruction_weight: 1.0
  save_best_balanced: false
  save_every: {save_every}
  source_recon_weights:
  - 1.0
  - 1.0
  - 1.0
  - 0.05
  student_back_drop_prob: 0.15
  student_frame_drop_rate: 0.5
  student_front_drop_prob: 0.15
  student_source_drop_rate: 0.3
  teacher_dropout_rate: 0.1
  teacher_momentum: 0.996
  temporal_contrastive_weight: {temporal}
  use_spatial_vicreg: true
  variance_weight: {var_weight}
  vicreg_min_std: 1.0
  warmup_epochs: 10
  weight_decay: 0.01
"""

EXPERIMENTS = [
    # (name, kappa, consist, temporal, cr_weight, batch_unif, var_weight, cov_weight, accum_steps, epochs, save_every, seed)
    ("v14_multi_shared_ts",     5000, 0.05, 0.1, 0.1,  0.1, 0.5, 0.1, 1, 50, 10, 42),
    ("v14_multi_baseline",      5000, 0.05, 0.1, 0.0,  0.1, 0.5, 0.1, 1, 50, 10, 43),
    ("v14_harbin_only",         5000, 0.05, 0.1, 0.1,  0.1, 0.5, 0.1, 1, 50, 10, 44),
    ("v14_high_consist",        5000, 0.10, 0.1, 0.1,  0.1, 0.5, 0.1, 1, 50, 10, 45),
    ("v14_high_batch",          5000, 0.05, 0.1, 0.1,  0.1, 0.5, 0.1, 4, 50, 10, 46),
    ("v14_high_temporal",       5000, 0.05, 0.25, 0.1, 0.1, 0.5, 0.1, 1, 50, 10, 47),
]

out_dir = "/workspace/xuannv/configs/v14"
os.makedirs(out_dir, exist_ok=True)

for name, kappa, consist, temporal, cr_weight, batch_unif, var_weight, cov_weight, accum_steps, epochs, save_every, seed in EXPERIMENTS:
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as f:
        f.write(BASE_CONFIG.format(
            name=name, kappa=kappa, consist=consist, temporal=temporal,
            cr_weight=cr_weight, batch_unif=batch_unif,
            var_weight=var_weight, cov_weight=cov_weight,
            accum_steps=accum_steps, epochs=epochs, save_every=save_every, seed=seed,
        ))
    print(f"Generated: {path}")

print(f"\nAll {len(EXPERIMENTS)} V14 configs generated in {out_dir}/")
