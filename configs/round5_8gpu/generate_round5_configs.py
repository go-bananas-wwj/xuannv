#!/usr/bin/env python3
"""生成 Round 5 的 6 个实验配置."""
from pathlib import Path
import yaml

# 基础配置: 基于 round4_full_vicreg_baseline.yaml
BASE_CONFIG = {
    "data": {
        "batch_size": 4,
        "cross_temporal": False,
        "cross_temporal_min_gap_months": 2,
        "cross_temporal_prob": 0.0,
        "dataset_type": "harbin_patches",
        "image_size": 128,
        "input_dim": 6,
        "manifest_path": "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered",
        "max_frames": 32,
        "metadata_dim": 4,
        "num_classes": 11,
        "num_input_sources": 3,
        "num_samples": 424,
        "num_target_sources": 4,
        "num_workers": 0,
        "preload": True,
        "random_target_frame": True,
        "source_channels": {"landsat": 6, "s1": 2, "s2": 6},
        "spatial_augmentation": True,
        "stats_dir": "/workspace/statistics/harbin_scenes",
        "target_sources": [
            {"loss_type": 0, "name": "s2", "out_channels": 6, "sensor_src": "s2"},
            {"loss_type": 0, "name": "s1", "out_channels": 2, "sensor_src": "s1"},
            {"loss_type": 0, "name": "landsat", "out_channels": 6, "sensor_src": "landsat"},
            {"loss_type": 0, "name": "dem", "out_channels": 1, "sensor_src": None},
        ],
        "temporal_window_augmentation": True,
        "temporal_window_max_frames": 24,
        "temporal_window_min_frames": 4,
        "temporal_window_prob": 0.5,
    },
    "evaluation": {"bootstrap_samples": 100, "knn_k": 5},
    "experiment": {"name": "PLACEHOLDER", "output_dir": "PLACEHOLDER", "seed": 42},
    "model": {
        "bottleneck_noise_scale": 0.02,
        "decoder_hidden_mult": 1,
        "embedding_dim": 64,
        "gradient_checkpointing": True,
        "input_dim": 6,
        "metadata_dim": 4,
        "num_blocks": 8,
        "num_heads": 8,
        "num_sensor_types": 16,
        "per_source_decoders": True,
        "precision_dim": 128,
        "relative_time_code_dim": 16,
        "skip_l2_norm_training": False,
        "space_dim": 256,
        "stem_dim": 128,
        "time_code_dim": 64,
        "time_dim": 256,
        "vmf_kappa": 2000.0,
        "window_code_dim": 64,
    },
    "training": {
        "batch_uniformity_weight": 0.01,
        "classification_weight": 0.0,
        "consistency_weight": 0.02,
        "covariance_weight": 0.1,
        "decorr_weight": 0.0,
        "early_stop_patience": 20,
        "epochs": 20,
        "grad_clip_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "lr": 0.0001,
        "lr_min": 1.0e-06,
        "lr_schedule": "cosine_no_restart",
        "max_steps_per_epoch": 100,
        "orthogonality_weight": 0.0,
        "pre_norm_uniform_weight": 0.0,
        "recon_warmup_epochs": 0,
        "reconstruction_weight": 1.0,
        "save_best_balanced": False,
        "save_every": 10,
        "source_recon_weights": [1.0, 1.0, 1.0, 0.05],
        "student_back_drop_prob": 0.15,
        "student_frame_drop_rate": 0.5,
        "student_front_drop_prob": 0.15,
        "student_source_drop_rate": 0.3,
        "teacher_dropout_rate": 0.1,
        "teacher_momentum": 0.996,
        "temporal_contrastive_weight": 0.1,
        "use_spatial_vicreg": True,
        "variance_weight": 0.5,
        "vicreg_min_std": 1.0,
        "warmup_epochs": 5,
        "weight_decay": 0.01,
    },
}


def deep_merge(base, override):
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


EXPERIMENTS = [
    {
        "name": "round5_kappa_temporal_mild",
        "desc": "kappa=5000 + temporal=0.25, 40 epochs — 打破trade-off",
        "overrides": {
            "model": {"vmf_kappa": 5000.0},
            "training": {
                "temporal_contrastive_weight": 0.25,
                "epochs": 40,
                "save_every": 10,
            },
        },
    },
    {
        "name": "round5_baseline_40ep",
        "desc": "baseline训练40 epochs — 验证训练充分性",
        "overrides": {
            "training": {
                "epochs": 40,
                "save_every": 10,
            },
        },
    },
    {
        "name": "round5_consist_mild",
        "desc": "consist=0.05 — 找consistency平衡点",
        "overrides": {
            "training": {
                "consistency_weight": 0.05,
            },
        },
    },
    {
        "name": "round5_no_consist",
        "desc": "consist=0.0 — 验证consistency对AUC的影响",
        "overrides": {
            "training": {
                "consistency_weight": 0.0,
            },
        },
    },
    {
        "name": "round5_kappa_baseline",
        "desc": "kappa=5000 — 确认最佳单参数",
        "overrides": {
            "model": {"vmf_kappa": 5000.0},
        },
    },
    {
        "name": "round5_temporal_plus_recon",
        "desc": "temporal=0.5 + recon=1.5 — 强重建拉住时序",
        "overrides": {
            "training": {
                "temporal_contrastive_weight": 0.5,
                "reconstruction_weight": 1.5,
            },
        },
    },
]


def main():
    config_dir = Path(__file__).parent
    config_dir.mkdir(exist_ok=True)

    for exp in EXPERIMENTS:
        cfg = deep_merge(BASE_CONFIG, exp["overrides"])
        cfg["experiment"]["name"] = exp["name"]
        cfg["experiment"]["output_dir"] = f"/workspace/outputs/round5/{exp['name']}"

        path = config_dir / f"{exp['name']}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Generated: {path} — {exp['desc']}")


if __name__ == "__main__":
    main()
