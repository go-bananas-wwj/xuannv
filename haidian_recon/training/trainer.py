"""训练器 — DDP训练循环."""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from haidian_recon.data.dataset import HaidianReconDataset, collate_fn
from haidian_recon.data.masking import FourLayerMask
from haidian_recon.models.hre_model import HREModel
from haidian_recon.losses.reconstruction import reconstruction_loss
from haidian_recon.losses.distillation import AEFDistiller, aef_distillation_loss
from haidian_recon.losses.uniformity import uniformity_loss
from haidian_recon.training.optimizer import build_optimizer, CosineScheduler


class HRETrainer:
    """海淀区重建底座训练器."""

    def __init__(self, cfg, rank: int = 0, world_size: int = 1, local_rank: int = 0) -> None:
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"npu:{local_rank}")

        # 模型
        source_channels = {s["name"]: s["channels"] for s in cfg.data.sources}
        self.model = HREModel(
            source_channels=source_channels,
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            num_encoder_layers=cfg.model.num_encoder_layers,
            num_decoder_layers=cfg.model.num_decoder_layers,
            num_heads=cfg.model.num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            output_dim=cfg.model.output_dim,
            dropout=cfg.model.dropout,
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        ).to(self.device)

        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False)

        # Masking
        self.masking = FourLayerMask(
            source_names=list(source_channels.keys()),
            image_size=cfg.model.image_size,
            patch_size=cfg.model.patch_size,
            modality_probs=cfg.masking.modality_probs,
            temporal_keep_ratio=cfg.masking.temporal_keep_ratio,
            spatial_visible_ratio=cfg.masking.spatial_visible_ratio,
            channel_keep_ratio=cfg.masking.channel_keep_ratio,
        ).to(self.device)

        # AEF蒸馏 — 所有rank都加载（已冻结，保证DDP同步）
        self.aef_distiller = None
        if cfg.training.aef_checkpoint:
            try:
                self.aef_distiller = AEFDistiller(
                    checkpoint_path=cfg.training.aef_checkpoint,
                    config_path=cfg.training.aef_config,
                    device=str(self.device),
                )
                self.aef_distiller.to(self.device)
                self.aef_distiller.eval()
            except Exception as e:
                if rank == 0:
                    print(f"[WARN] Failed to load AEF distiller: {e}")

        # 优化器
        self.optimizer = build_optimizer(self.model, cfg.training.lr, cfg.training.weight_decay)

        # 数据集
        self.train_dataset = HaidianReconDataset(
            data_root=cfg.data.data_root,
            planet_root=cfg.data.planet_root,
            stats_dir=cfg.data.stats_dir,
            split="train",
            image_size=cfg.data.image_size,
            source_names=list(source_channels.keys()),
            cache_dir=cfg.data.cache_dir,
        )
        self.val_dataset = HaidianReconDataset(
            data_root=cfg.data.data_root,
            planet_root=cfg.data.planet_root,
            stats_dir=cfg.data.stats_dir,
            split="val",
            image_size=cfg.data.image_size,
            source_names=list(source_channels.keys()),
            cache_dir=cfg.data.cache_dir,
        )

        if world_size > 1:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.data.batch_size,
            sampler=self.train_sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,  # NPU不支持pin_memory
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg.data.batch_size,
            sampler=self.val_sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        # 调度器
        steps_per_epoch = len(self.train_loader)
        total_steps = cfg.training.epochs * steps_per_epoch
        warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
        self.scheduler = CosineScheduler(self.optimizer, total_steps, warmup_steps, cfg.training.lr_min)

        # 输出目录
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.start_epoch = 0

    def train(self) -> None:
        cfg = self.cfg.training

        for epoch in range(self.start_epoch, cfg.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_distill = 0.0
            epoch_uniform = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                # 数据移到device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}

                # 应用mask
                masked_batch, mask_info = self.masking(batch)

                # 前向
                output = self.model(masked_batch, mask_info)

                # 重建损失
                loss_recon = reconstruction_loss(output["reconstructions"], batch, mask_info)

                # AEF蒸馏
                loss_distill = torch.tensor(0.0, device=self.device)
                if self.aef_distiller is not None:
                    aef_emb = self.aef_distiller(batch)
                    loss_distill = aef_distillation_loss(output["embedding"], aef_emb)

                # 推散损失
                loss_uniform = uniformity_loss(output["embedding"])

                # 总损失
                loss = (
                    cfg.w_recon * loss_recon
                    + cfg.w_distill * loss_distill
                    + cfg.w_uniform * loss_uniform
                )

                # NaN/Inf检查
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.rank == 0:
                        print(f"[WARN] NaN/Inf loss at step {self.global_step}, skipping...")
                    self.optimizer.zero_grad()
                    # dummy backward保持DDP同步
                    if self.world_size > 1:
                        for p in self.model.parameters():
                            if p.requires_grad:
                                p.grad = torch.zeros_like(p)
                                break
                    continue

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()

                # 记录
                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_distill += loss_distill.item()
                epoch_uniform += loss_uniform.item()
                self.global_step += 1

                if self.rank == 0 and self.global_step % cfg.log_every == 0:
                    print(f"[Step {self.global_step}] loss={loss.item():.4f} "
                          f"recon={loss_recon.item():.4f} distill={loss_distill.item():.4f} "
                          f"uniform={loss_uniform.item():.4f} lr={self.scheduler.optimizer.param_groups[0]['lr']:.6f}")

            # Epoch日志
            n_batches = len(self.train_loader)
            if self.rank == 0:
                print(f"[Epoch {epoch}] loss={epoch_loss/n_batches:.4f} "
                      f"recon={epoch_recon/n_batches:.4f} "
                      f"distill={epoch_distill/n_batches:.4f} "
                      f"uniform={epoch_uniform/n_batches:.4f}")

            # 保存
            if self.rank == 0 and (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(epoch)

            # 验证
            if (epoch + 1) % cfg.eval_every == 0:
                self.evaluate()

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_recon = 0.0
        total_samples = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            masked_batch, mask_info = self.masking(batch)
            output = self.model(masked_batch, mask_info)
            loss_recon = reconstruction_loss(output["reconstructions"], batch, mask_info)
            total_recon += loss_recon.item()
            total_samples += 1

        avg_recon = total_recon / max(total_samples, 1)

        # 跨rank聚合
        if self.world_size > 1:
            recon_tensor = torch.tensor(avg_recon, device=self.device)
            dist.all_reduce(recon_tensor, op=dist.ReduceOp.SUM)
            avg_recon = recon_tensor.item() / self.world_size

        if self.rank == 0:
            print(f"[Val] recon_loss={avg_recon:.4f}")
        return {"recon_loss": avg_recon}

    def save_checkpoint(self, epoch: int) -> None:
        path = self.output_dir / f"epoch_{epoch}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step_count": self.scheduler.step_count,
            "global_step": self.global_step,
        }
        torch.save(state, path)
        print(f"[Save] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model_state_dict"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.scheduler.step_count = ckpt.get("scheduler_step_count", 0)
        print(f"[Load] Checkpoint loaded from {path}")
