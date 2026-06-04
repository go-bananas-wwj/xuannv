#!/usr/bin/env python3
"""DDP 训练入口.

用法:
    cd /workspace/xuannv
    torchrun --nproc_per_node=8 \
        scripts/train/train.py --config configs/config.yaml \
        --save-every 20

V12 核心:
  - 只重建 S2/S1/Landsat 3个动态源
  - 极简3-loss: Recon + BatchUniformity + Consistency
  - 128维 Embedding
  - Teacher-Student 一致性为核心机制
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import time

torch.set_num_threads(4)

from src.config import load_config
from src.data.builder import build_dataloader
from src.training.trainer import DDPv13Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--soft-restart", type=str, default=None, help="软重启: 从旧 checkpoint 加载 encoder，重置其余")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的训练轮数")
    parser.add_argument("--save-every", type=int, default=None, help="每隔多少 epoch 保存检查点")
    parser.add_argument("--local-rank", type=int, default=0)
    return parser.parse_args()


class FileLogger:
    """同时写入 stdout 和文件的 logger（仅 rank 0）."""
    def __init__(self, filepath: str, rank: int):
        self.rank = rank
        self.file = None
        if rank == 0:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.file = open(filepath, "a", buffering=1, encoding="utf-8")

    def write(self, msg: str):
        if self.rank == 0:
            sys.stdout.write(msg)
            sys.stdout.flush()
            if self.file:
                self.file.write(msg)
                self.file.flush()

    def Print(self, msg: str):
        self.write(msg + "\n")


def main():
    args = parse_args()

    # DDP 初始化
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.npu.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    cfg = load_config(args.config)
    # 时间戳由 rank 0 生成后广播，确保所有 rank 使用同一文件名
    from datetime import datetime
    ts_tensor = torch.zeros(15, dtype=torch.uint8, device=f"npu:{local_rank}")
    if global_rank == 0:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, c in enumerate(ts_str):
            ts_tensor[i] = ord(c)
    dist.broadcast(ts_tensor, src=0)
    ts = "".join(chr(int(ts_tensor[i])) for i in range(15))
    log_path = Path(cfg.experiment.output_dir) / f"train_{ts}.log"
    logger = FileLogger(str(log_path), global_rank)

    # 覆盖参数
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.save_every is not None:
        cfg.training.save_every = args.save_every

    # 固定随机种子
    seed = getattr(cfg.experiment, "seed", 42) + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

    if global_rank == 0:
        logger.Print("=" * 70)
        logger.Print(f"  DDP V12 训练  [8×NPU]  —  {cfg.experiment.name}")
        logger.Print(f"  World size: {world_size}  |  Rank: {global_rank}")
        logger.Print("=" * 70)
        logger.Print(f"  Config: {args.config}")
        logger.Print(f"  Epochs: {cfg.training.epochs}")
        logger.Print(f"  Batch per GPU: {cfg.data.batch_size}")
        logger.Print(f"  Grad accum: {getattr(cfg.training, 'gradient_accumulation_steps', 2)}")
        logger.Print(f"  Effective batch: {cfg.data.batch_size * world_size * getattr(cfg.training, 'gradient_accumulation_steps', 2)}")
        logger.Print(f"  Recon weight: {getattr(cfg.training, 'reconstruction_weight', 1.0)}")
        logger.Print(f"  Consistency weight: {getattr(cfg.training, 'consistency_weight', 0.0)}")
        logger.Print(f"  Temporal contrastive weight: {getattr(cfg.training, 'temporal_contrastive_weight', 0.0)}")
        logger.Print(f"  Use L2 space VICReg: {getattr(cfg.training, 'use_l2_space_vicreg', False)}")
        logger.Print(f"  BatchUniformity weight: {getattr(cfg.training, 'batch_uniformity_weight', 0.0)}")
        src_weights = getattr(cfg.training, 'source_recon_weights', [1.0]*3)
        logger.Print(f"  Source recon weights: {src_weights}")
        logger.Print(f"  Embedding dim: {getattr(cfg.model, 'embedding_dim', 128)}")
        logger.Print(f"  VMF kappa: {getattr(cfg.model, 'vmf_kappa', 2000.0)}")
        logger.Print("=" * 70)

    # DataLoader
    dataloader = build_dataloader(
        cfg,
        training=True,
        distributed=True,
        world_size=world_size,
        rank=global_rank,
    )

    if global_rank == 0:
        ds = dataloader.dataset
        logger.Print("\n[Dataset Info]")
        logger.Print(f"  Dataset type: {type(ds).__name__}")
        logger.Print(f"  Total patches: {getattr(ds, 'num_samples', len(getattr(ds, 'patches', [])))}")
        logger.Print(f"  Monthly samples: {len(getattr(ds, 'monthly_samples', []))}")
        logger.Print(f"  Batch size (per GPU): {cfg.data.batch_size}")
        logger.Print(f"  World size: {world_size}")
        logger.Print(f"  Effective batch: {cfg.data.batch_size * world_size}")
        logger.Print(f"  Num workers: {getattr(cfg.data, 'num_workers', 0)}")
        logger.Print(f"  Preload: {getattr(cfg.data, 'preload', False)}")
        logger.Print(f"  Max frames: {getattr(cfg.data, 'max_frames', 'N/A')}")
        logger.Print(f"  Max steps/epoch: {getattr(cfg.training, 'max_steps_per_epoch', 'N/A (full pass)')}")
        logger.Print(f"  Multi-region manifest: {getattr(cfg.data, 'multi_region_manifest', 'N/A')}")
        logger.Print(f"  Input sources: {getattr(cfg.data, 'input_sources', 'N/A')}")
        logger.Print(f"  Target sources: {[t.get('name', t) for t in getattr(cfg.data, 'target_sources', [])]}")
        logger.Print("=" * 70)

    # Trainer（传入同一 ts，确保 step 日志与启动日志写入同一文件）
    trainer = DDPv13Trainer(cfg, local_rank=local_rank, log_ts=ts)

    start_epoch = 0
    if args.soft_restart:
        trainer.soft_restart(args.soft_restart)
        if global_rank == 0:
            logger.Print(f"[train] Soft restart from {args.soft_restart}")
        dist.barrier()
    elif args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if global_rank == 0:
            logger.Print(f"[train] Resumed from {args.resume}, starting at epoch {start_epoch + 1}")
        dist.barrier()

    total_epochs = cfg.training.epochs
    best_recon = float("inf")
    best_miou = -1.0
    save_every = args.save_every if args.save_every is not None else getattr(cfg.training, "save_every", 20)
    eval_every = getattr(cfg.training, "eval_every", 10)
    epoch_start_time = time.time()
    bank_size = 0

    for epoch in range(start_epoch, total_epochs):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        epoch_t0 = time.time()
        losses = trainer.train_epoch(epoch, dataloader)
        epoch_dt = time.time() - epoch_t0
        elapsed = time.time() - epoch_start_time
        remain_epochs = total_epochs - (epoch + 1)
        eta_sec = epoch_dt * remain_epochs if remain_epochs > 0 else 0
        eta_str = f"{int(eta_sec // 3600)}h{int((eta_sec % 3600) // 60)}m"

        # 获取 Memory Bank 大小
        if hasattr(trainer, 'memory_bank'):
            bank_size = trainer.memory_bank.size

        # ========== 精简 Epoch 日志（只显示关键指标） ==========
        if global_rank == 0:
            aef_sp = losses.get('aef_sp', 0.0)
            aef_gl = losses.get('aef_gl', 0.0)
            olmo_sp = losses.get('olmo_sp', 0.0)
            olmo_gl = losses.get('olmo_gl', 0.0)
            logger.Print(
                f"[{time.strftime('%H:%M:%S')}] Epoch {epoch + 1:03d}/{cfg.training.epochs} | "
                f"total={losses['total']:.3f} recon={losses['recon']:.3f} cls={losses.get('cls', 0.0):.3f} "
                f"var={losses['var']:.3f} cov={losses['cov']:.3f} l2unif={losses['l2unif']:.3f} "
                f"erank={losses.get('erank', 0.0):.1f} "
                f"aef=[sp={aef_sp:.3f},gl={aef_gl:.3f}] olmo=[sp={olmo_sp:.3f},gl={olmo_gl:.3f}] "
                f"lr={losses['lr']:.6f} | "
                f"time={epoch_dt:.1f}s elapsed={int(elapsed//60)}m ETA={eta_str}"
            )

        # 定期保存断点（覆盖式 epoch_last.pt，含 optimizer，用于 OOM 后续训）
        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch + 1, losses, tag="last")

        # ★ 下游探针评估（kNN: global embedding -> WorldCover 众数类别）
        #   每次评估后存 best 权重，按 mIoU 仅保留最优 3 个
        do_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == total_epochs)
        if do_eval:
            metrics = trainer.evaluate_knn(dataloader, max_batches=64)  # 限制评估样本量(~256 patch)
            if dist.is_initialized():
                dist.barrier()
            if global_rank == 0:
                logger.Print(
                    f"  [Eval] epoch {epoch + 1}: kNN acc={metrics['acc']:.4f} "
                    f"mIoU={metrics['miou']:.4f}"
                )
                trainer.save_checkpoint(epoch + 1, losses, miou=metrics["miou"])
                if metrics["miou"] > best_miou:
                    best_miou = metrics["miou"]
                    logger.Print(
                        f"  [Best] New best mIoU={best_miou:.4f} at epoch {epoch + 1}"
                    )

        # ★ 完整下游任务评估（每 eval_every epoch 运行）
        #   包括: kNN 语义分割 mIoU + 变化检测 AUC
        do_full_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == total_epochs)
        if do_full_eval:
            ckpt_path = Path(cfg.experiment.output_dir) / f"epoch_{epoch + 1}.pt"
            if global_rank == 0:
                # 先保存普通 checkpoint（供评估脚本加载）
                trainer.save_checkpoint(epoch + 1, losses)
                eval_json_path = Path(cfg.experiment.output_dir) / f"eval_epoch_{epoch + 1}.json"
                eval_script = Path(__file__).parent.parent / "eval" / "run_periodic_eval.py"
                cmd = (
                    f"cd /workspace/xuannv && source activate xuannv && "
                    f"python {eval_script} "
                    f"--config {args.config} "
                    f"--checkpoint {ckpt_path} "
                    f"--output {eval_json_path} "
                    f"--device npu:{local_rank} "
                    f"--skip-cd"
                )
                logger.Print(f"  [FullEval] Running downstream evaluation for epoch {epoch + 1}...")
                import subprocess
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
                except subprocess.TimeoutExpired:
                    logger.Print(f"  [FullEval] Timeout after 600s, skipping")
                    result = None
                if result is not None and result.returncode == 0:
                    try:
                        with open(eval_json_path) as f:
                            eval_res = json.load(f)
                        knn = eval_res.get("knn", {})
                        logger.Print(
                            f"  [FullEval] epoch {epoch + 1}: "
                            f"pixel-kNN mIoU={knn.get('mIoU', 0):.4f} OA={knn.get('OA', 0):.4f} "
                            f"({eval_res.get('elapsed_seconds', 0):.1f}s)"
                        )
                    except Exception as e:
                        logger.Print(f"  [FullEval] Error reading results: {e}")
                else:
                    logger.Print(f"  [FullEval] Failed: {result.stderr[:200]}")
            if dist.is_initialized():
                dist.barrier()

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if global_rank == 0:
        logger.Print("[train] Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
