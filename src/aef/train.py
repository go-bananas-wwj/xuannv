"""AEF DDP 训练入口."""
from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import random

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn
from src.aef.training import Trainer


def setup_distributed() -> tuple[int, int, int]:
    """初始化 DDP."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
        )
        torch.npu.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def main() -> None:
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_aef_haidian.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="outputs/aef_haidian")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill-warmup-steps", type=int, default=1000)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed)

    if rank == 0:
        print(f"Rank: {rank}/{world_size}, LocalRank: {local_rank}, Device: npu:{local_rank}")
        print(f"Config: batch_size={args.batch_size}, lr={args.lr}, max_steps={args.max_steps}")

    # Build datasets
    source_names = ["tianyi_sar", "s1", "s2", "landsat", "planet"]
    train_dataset = HaidianAEFDataset(
        data_root="data_raw/haidian/scenes",
        planet_root="data_raw/beijing/planetscene",
        stats_dir="statistics/haidian",
        split="train",
        image_size=128,
        source_names=source_names,
        max_frames=16,
    )
    val_dataset = HaidianAEFDataset(
        data_root="data_raw/haidian/scenes",
        planet_root="data_raw/beijing/planetscene",
        stats_dir="statistics/haidian",
        split="val",
        image_size=128,
        source_names=source_names,
        max_frames=16,
    )

    # Samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Build model
    source_channels = {
        "tianyi_sar": 1,
        "s1": 2,
        "s2": 6,
        "landsat": 6,
        "planet": 4,
    }
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources=source_channels,
        decode_sources=source_channels,
        per_source_latent=32,
        enable_text_align=False,
    )

    device = f"npu:{local_rank}"
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Resume
    start_step = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        # 处理 module. 前缀：checkpoint 可能是原始模型或 DDP 模型
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if world_size > 1 and not has_module_prefix:
            # checkpoint 是原始模型，DDP wrapper 需要加前缀
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif world_size == 1 and has_module_prefix:
            # checkpoint 是 DDP 模型，单卡恢复需要去掉前缀
            state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        start_step = ckpt.get("step", 0)
        resume_optimizer_state = ckpt.get("optimizer_state_dict")
        resume_scheduler_state = ckpt.get("scheduler_state_dict")
        if rank == 0:
            print(f"[Resume] Loaded checkpoint from {args.resume} at step {start_step}")

    # Trainer 接收 DDP wrapper，确保梯度 all_reduce 生效
    trainer = Trainer(
        model=model,  # DDP wrapper
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
        rank=rank,
        world_size=world_size,
        distill_warmup_steps=args.distill_warmup_steps,
        resume_step=start_step,
        resume_optimizer_state=resume_optimizer_state,
        resume_scheduler_state=resume_scheduler_state,
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
