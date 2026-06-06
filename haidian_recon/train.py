"""训练入口 — 支持torchrun DDP."""
from __future__ import annotations

import argparse
import os
import sys

import torch

torch.set_num_threads(4)

import yaml

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch.distributed as dist
import torch_npu  # NPU必须

from haidian_recon.config import Config, ModelConfig, DataConfig, MaskingConfig, TrainingConfig
from haidian_recon.training.trainer import HRETrainer


def load_config(path: str) -> Config:
    """从YAML加载配置."""
    with open(path) as f:
        d = yaml.safe_load(f)

    cfg = Config()
    if "experiment_name" in d:
        cfg.experiment_name = d["experiment_name"]
    if "output_dir" in d:
        cfg.output_dir = d["output_dir"]
    if "seed" in d:
        cfg.seed = d["seed"]
    if "device" in d:
        cfg.device = d["device"]
    if "backend" in d:
        cfg.backend = d["backend"]

    if "model" in d:
        for k, v in d["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
    if "masking" in d:
        for k, v in d["masking"].items():
            if hasattr(cfg.masking, k):
                setattr(cfg.masking, k, v)
    if "data" in d:
        for k, v in d["data"].items():
            if hasattr(cfg.data, k):
                setattr(cfg.data, k, v)
    if "training" in d:
        for k, v in d["training"].items():
            if hasattr(cfg.training, k):
                setattr(cfg.training, k, v)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="haidian_recon/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # DDP初始化
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
        torch.npu.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if rank == 0:
        print(f"=== HaidianReconEncoder Training ===")
        print(f"Rank: {rank}/{world_size}, LocalRank: {local_rank}, Device: npu:{local_rank}")

    # 配置
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        cfg = Config()

    if rank == 0:
        print(f"Config: epochs={cfg.training.epochs}, batch_size={cfg.data.batch_size}, lr={cfg.training.lr}")

    # 训练器
    trainer = HRETrainer(cfg, rank=rank, world_size=world_size, local_rank=local_rank)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 训练
    try:
        trainer.train()
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
