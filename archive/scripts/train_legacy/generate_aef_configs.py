#!/usr/bin/env python3
"""生成 8 个 AEF 对齐实验配置."""
from pathlib import Path
import yaml

BASE = {
    "experiment": {
        "name": "PLACEHOLDER",
        "seed": 42,
        "output_dir": "PLACEHOLDER",
    },
    "data": {
        "dataset_type": "harbin_patches",
        "manifest_path": "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered",
        "stats_dir": "/workspace/statistics/harbin_scenes",
        "batch_size": 4,
        "num_workers": 0,
        "image_size": 128,
        "max_frames": 32,
        "num_samples": 100,
        "num_classes": 11,
        "input_dim": 6,
        "metadata_dim": 4,
        "num_input_sources": 3,
        "num_target_sources": 4,
        "random_target_frame": True,
        "spatial_augmentation": True,
        "temporal_window_augmentation": True,
        "temporal_window_prob": 0.5,
        "temporal_window_min_frames": 4,
        "temporal_window_max_frames": 24,
        "preload": True,
        "source_channels": {"s2": 6, "s1": 2, "landsat": 6},
        "target_sources": [
            {"name": "s2", "loss_type": 0, "sensor_src": "s2", "out_channels": 6},
            {"name": "s1", "loss_type": 0, "sensor_src": "s1", "out_channels": 2},
            {"name": "landsat", "loss_type": 0, "sensor_src": "landsat", "out_channels": 6},
            {"name": "dem", "loss_type": 0, "sensor_src": None, "out_channels": 1},
        ],
    },
    "model": {
        "input_dim": 6,
        "stem_dim": 128,
        "precision_dim": 128,
        "time_dim": 256,
        "space_dim": 256,
        "embedding_dim": 64,
        "time_code_dim": 64,
        "window_code_dim": 64,
        "relative_time_code_dim": 16,
        "num_blocks": 8,
        "num_heads": 8,
        "vmf_kappa": 50.0,
        "bottleneck_noise_scale": 0.02,
        "metadata_dim": 4,
        "num_sensor_types": 16,
        "gradient_checkpointing": True,
        "per_source_decoders": True,
        "decoder_hidden_mult": 1,
        "skip_l2_norm_training": False,
    },
    "training": {
        "epochs": 20,
        "gradient_accumulation_steps": 1,
        "lr": 0.0001,
        "lr_min": 1.0e-06,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "reconstruction_weight": 1.0,
        "consistency_weight": 0.02,
        "classification_weight": 0.0,
        "decorr_weight": 0.0,
        "orthogonality_weight": 0.0,
        "pre_norm_uniform_weight": 0.0,
        "variance_weight": 0.0,
        "covariance_weight": 0.0,
        "batch_uniformity_weight": 0.05,
        "vicreg_min_std": 1.0,
        "source_recon_weights": [1.0, 1.0, 1.0, 0.05],
        "teacher_momentum": 0.996,
        "teacher_dropout_rate": 0.1,
        "student_frame_drop_rate": 0.5,
        "student_source_drop_rate": 0.3,
        "student_front_drop_prob": 0.15,
        "student_back_drop_prob": 0.15,
        "lr_schedule": "cosine_no_restart",
        "warmup_epochs": 5,
        "recon_warmup_epochs": 0,
        "max_steps_per_epoch": 50,
        "save_every": 10,
        "save_best_balanced": False,
        "early_stop_patience": 20,
    },
    "evaluation": {
        "knn_k": 5,
        "bootstrap_samples": 100,
    },
}

EXPERIMENTS = [
    {
        "name": "aef_baseline",
        "desc": "AEF 完全对齐",
        "overrides": {},
    },
    {
        "name": "aef_high_consist",
        "desc": "高一致性权重",
        "overrides": {"training": {"consistency_weight": 0.05}},
    },
    {
        "name": "aef_no_static",
        "desc": "无 static 目标 (DEM=0)",
        "overrides": {"training": {"source_recon_weights": [1.0, 1.0, 1.0, 0.0]}},
    },
    {
        "name": "aef_skip_l2",
        "desc": "Skip L2 训练时 (对比)",
        "overrides": {"model": {"skip_l2_norm_training": True}},
    },
    {
        "name": "aef_diff_recon",
        "desc": "差异化重建权重 (S2=1.0, S1=0.5, Landsat=0.8, DEM=0.05)",
        "overrides": {"training": {"source_recon_weights": [1.0, 0.5, 0.8, 0.05]}},
    },
    {
        "name": "aef_high_kappa",
        "desc": "Kappa=2000",
        "overrides": {"model": {"vmf_kappa": 2000.0}},
    },
    {
        "name": "aef_cyclic_unif",
        "desc": "Cyclic Shift Batch Uniformity (AEF风格)",
        "overrides": {"training": {"batch_uniformity_type": "cyclic_shift", "batch_uniformity_weight": 0.05}},
    },
    {
        "name": "aef_no_uniform",
        "desc": "无 uniformity (仅 Recon+Consist)",
        "overrides": {"training": {"batch_uniformity_weight": 0.0}},
    },
]


def deep_merge(base, override):
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def main():
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    for exp in EXPERIMENTS:
        cfg = deep_merge(BASE, exp["overrides"])
        cfg["experiment"]["name"] = exp["name"]
        cfg["experiment"]["output_dir"] = f"/workspace/outputs/{exp['name']}"

        # 使用已有的 100 patch 采样结果（从变化检测清单 + 随机选取）
        import yaml
        with open("configs/v2_baseline.yaml") as f:
            v2_cfg = yaml.safe_load(f)
        cfg["data"]["patch_list"] = v2_cfg["data"]["patch_list"]

        path = config_dir / f"{exp['name']}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Generated: {path} — {exp['desc']}")


if __name__ == "__main__":
    main()
