"""AEF 蒸馏版 8 卡 NPU 训练入口 — 输出隔离在 aef_reference/outputs/."""
from __future__ import annotations

import argparse
import os
import sys

# 引用 xuannv 主项目的 src/aef/ 代码
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    # CANN 8.5.1 多卡 NPU 存在 SDMA 硬件竞态条件，必须强制同步执行
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill-warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # 设置 CANN 编译缓存路径（避免 8 卡并发冲突）
    cache_path = f"/tmp/ascend_cache_{local_rank}"
    os.makedirs(cache_path, exist_ok=True)
    os.environ["ASCEND_CACHE_PATH"] = cache_path

    set_seed(args.seed)

    if rank == 0:
        print(f"Rank: {rank}/{world_size}, LocalRank: {local_rank}, Device: npu:{local_rank}")
        print(f"batch_size={args.batch_size}, lr={args.lr}, max_steps={args.max_steps}")
        print(f"output_dir={args.output_dir}")
        print(f"distill_warmup_steps={args.distill_warmup_steps}")

    # Build datasets (海淀数据，5源输入 + 辅助重建目标)
    source_names = ["s1", "s2", "tianyi_sar", "landsat", "planet", "dem", "worldcover", "dynamic_world", "jrc_water"]
    train_dataset = HaidianAEFDataset(
        data_root="data_raw/haidian/scenes",
        planet_root="data_raw/beijing/planetscene",
        stats_dir="statistics/haidian",
        split="train",
        image_size=128,
        source_names=source_names,
        max_frames=16,
        aef_embedding_root="data_raw/haidian/aef_embeddings/haidian_2025_patches",
        start_date="20251201",
        end_date="20260430",
    )
    val_dataset = HaidianAEFDataset(
        data_root="data_raw/haidian/scenes",
        planet_root="data_raw/beijing/planetscene",
        stats_dir="statistics/haidian",
        split="val",
        image_size=128,
        source_names=source_names,
        max_frames=16,
        aef_embedding_root="data_raw/haidian/aef_embeddings/haidian_2025_patches",
        start_date="20251201",
        end_date="20260430",
    )

    # Samplers
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )

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

    # Build model (5源输入)
    input_sources = {
        "s1": 2,
        "s2": 6,
        "tianyi_sar": 1,
        "landsat": 6,
        "planet": 4,
    }
    decode_sources = {
        "s1": 2,
        "s2": 6,
        "tianyi_sar": 1,
        "landsat": 6,
        "planet": 4,
        "dem": 1,
        "worldcover": 11,
        "dynamic_world": 9,
        "jrc_water": 1,
    }
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources=input_sources,
        decode_sources=decode_sources,
        per_source_latent=32,
        enable_text_align=False,
    )

    device = f"npu:{local_rank}"
    model = model.to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Resume
    start_step = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if world_size > 1 and not has_module_prefix:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif world_size == 1 and has_module_prefix:
            state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        start_step = ckpt.get("step", 0)
        resume_optimizer_state = ckpt.get("optimizer_state_dict")
        resume_scheduler_state = ckpt.get("scheduler_state_dict")
        if rank == 0:
            print(f"[Resume] Loaded checkpoint from {args.resume} at step {start_step}")

    output_dir = args.output_dir or f"/workspace/xuannv/aef_reference/outputs/aef_distill_seed{args.seed}"

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        distill_warmup_steps=args.distill_warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        resume_step=start_step,
        resume_optimizer_state=resume_optimizer_state,
        resume_scheduler_state=resume_scheduler_state,
        seed=args.seed,
        viz_patch_ids=["patch_000036", "patch_000069", "patch_000091", "patch_000120", "patch_000150"],
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
