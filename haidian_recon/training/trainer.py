"""训练器 — DDP训练循环."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from haidian_recon.data.dataset import HaidianReconDataset, collate_fn
from haidian_recon.data.masking import FourLayerMask
from haidian_recon.models.hre_model import HREModel
from haidian_recon.losses.reconstruction import reconstruction_loss
from haidian_recon.losses.distillation import aef_distillation_loss
from haidian_recon.losses.uniformity import uniformity_loss, spatial_uniformity_loss
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
        self.source_channels = source_channels
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
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)

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

        # AEF蒸馏 — 使用预计算的embedding（从dataset加载）
        self.use_aef_distill = bool(cfg.training.aef_checkpoint) and cfg.training.aef_checkpoint != "null"

        # 优化器
        self.optimizer = build_optimizer(self.model, cfg.training.lr, cfg.training.weight_decay)

        # 数据集
        aef_emb_root = getattr(cfg.data, "aef_embedding_root", "data_raw/haidian/aef_embeddings/haidian_2025_patches")
        self.train_dataset = HaidianReconDataset(
            data_root=cfg.data.data_root,
            planet_root=cfg.data.planet_root,
            stats_dir=cfg.data.stats_dir,
            split="train",
            image_size=cfg.data.image_size,
            source_names=list(source_channels.keys()),
            cache_dir=cfg.data.cache_dir,
            aef_embedding_root=aef_emb_root,
        )
        self.val_dataset = HaidianReconDataset(
            data_root=cfg.data.data_root,
            planet_root=cfg.data.planet_root,
            stats_dir=cfg.data.stats_dir,
            split="val",
            image_size=cfg.data.image_size,
            source_names=list(source_channels.keys()),
            cache_dir=cfg.data.cache_dir,
            aef_embedding_root=aef_emb_root,
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

        # 预取固定验证 batch 用于可视化（绕过 DistributedSampler，避免每次 spawn worker）
        if self.rank == 0:
            val_iter = iter(self.val_loader)
            self.fixed_val_batch = next(val_iter)
            self.fixed_val_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.fixed_val_batch.items()
            }
        else:
            self.fixed_val_batch = None

        # 调度器：使用 rank 0 的步数统一所有 rank，避免 DDP 下样本分配不均导致偏差
        steps_per_epoch = len(self.train_loader)
        total_steps = cfg.training.epochs * steps_per_epoch
        if self.world_size > 1:
            steps_tensor = torch.tensor(total_steps, device=self.device)
            dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX)
            total_steps = int(steps_tensor.item())
        warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
        if self.world_size > 1:
            we_tensor = torch.tensor(warmup_steps, device=self.device)
            dist.all_reduce(we_tensor, op=dist.ReduceOp.MAX)
            warmup_steps = int(we_tensor.item())
        self.scheduler = CosineScheduler(self.optimizer, total_steps, warmup_steps, cfg.training.lr_min)

        # 输出目录
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.start_epoch = 0
        self.loss_history = []  # [(epoch, loss, recon, distill, uniform, spatial_u), ...]

    def train(self) -> None:
        cfg = self.cfg.training

        for epoch in range(self.start_epoch, cfg.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            epoch_loss = torch.tensor(0.0, device=self.device)
            epoch_recon = torch.tensor(0.0, device=self.device)
            epoch_distill = torch.tensor(0.0, device=self.device)
            epoch_uniform = torch.tensor(0.0, device=self.device)
            epoch_spatial_uniform = torch.tensor(0.0, device=self.device)
            actual_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # 数据移到device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}

                # 应用mask
                masked_batch, mask_info = self.masking(batch)

                # 前向
                output = self.model(masked_batch, mask_info)

                # 重建损失
                loss_recon = reconstruction_loss(output["reconstructions"], batch, mask_info)

                # AEF蒸馏 — 使用预计算的embedding，过滤掉全0的无效样本
                loss_distill = torch.tensor(0.0, device=self.device)
                if self.use_aef_distill and batch.get("aef_embedding") is not None:
                    aef_emb = batch["aef_embedding"]
                    aef_valid = aef_emb.abs().sum(dim=-1) > 1e-6
                    if aef_valid.any():
                        loss_distill = aef_distillation_loss(
                            output["embedding"][aef_valid],
                            aef_emb[aef_valid],
                        )

                # 推散损失（全局）
                loss_uniform = uniformity_loss(output["embedding"])

                # 推散损失（空间）
                loss_spatial_uniform = torch.tensor(0.0, device=self.device)
                if output["embedding_map"] is not None:
                    loss_spatial_uniform = spatial_uniformity_loss(output["embedding_map"])

                # 总损失
                w_spatial = getattr(cfg, "w_spatial_uniform", 0.02)
                loss = (
                    cfg.w_recon * loss_recon
                    + cfg.w_distill * loss_distill
                    + cfg.w_uniform * loss_uniform
                    + w_spatial * loss_spatial_uniform
                )

                # NaN/Inf检查
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.rank == 0:
                        print(f"[WARN] NaN/Inf loss at step {self.global_step}, skipping...")
                    self.optimizer.zero_grad()
                    # dummy backward: 触发所有参数的DDP all-reduce hook，防止死锁
                    # 使用单个参数的 zero-sum 替代逐参数循环，避免 NPU→CPU 频繁同步
                    if self.world_size > 1:
                        dummy = next(p for p in self.model.parameters() if p.requires_grad).sum() * 0.0
                        dummy.backward()
                    self.scheduler.step()
                    self.global_step += 1
                    actual_batches += 1
                    continue

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()

                # 记录（device上累加，避免NPU同步瓶颈）
                epoch_loss += loss.detach()
                epoch_recon += loss_recon.detach()
                epoch_distill += loss_distill.detach()
                epoch_uniform += loss_uniform.detach()
                epoch_spatial_uniform += loss_spatial_uniform.detach()
                self.global_step += 1
                actual_batches += 1

                if self.rank == 0 and self.global_step % cfg.log_every == 0:
                    print(f"[Step {self.global_step}] loss={loss.item():.4f} "
                          f"recon={loss_recon.item():.4f} distill={loss_distill.item():.4f} "
                          f"uniform={loss_uniform.item():.4f} spatial_u={loss_spatial_uniform.item():.4f} "
                          f"lr={self.scheduler.optimizer.param_groups[0]['lr']:.6f}")

            # Epoch日志
            n_batches = actual_batches if actual_batches > 0 else 1
            if self.rank == 0:
                avg_loss = (epoch_loss / n_batches).item()
                avg_recon = (epoch_recon / n_batches).item()
                avg_distill = (epoch_distill / n_batches).item()
                avg_uniform = (epoch_uniform / n_batches).item()
                avg_spatial = (epoch_spatial_uniform / n_batches).item()
                print(f"[Epoch {epoch}] loss={avg_loss:.4f} "
                      f"recon={avg_recon:.4f} "
                      f"distill={avg_distill:.4f} "
                      f"uniform={avg_uniform:.4f} "
                      f"spatial_u={avg_spatial:.4f}")
                self.loss_history.append({
                    "epoch": epoch,
                    "loss": avg_loss,
                    "recon": avg_recon,
                    "distill": avg_distill,
                    "uniform": avg_uniform,
                    "spatial_u": avg_spatial,
                })

            # 保存
            if self.rank == 0 and (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(epoch + 1)

            # 验证
            if (epoch + 1) % cfg.eval_every == 0:
                self.evaluate()

            # 每20epoch可视化：损失曲线 + 重建图像对比
            if self.rank == 0 and ((epoch + 1) % 20 == 0 or (epoch + 1) % 20 == 1):
                if len(self.loss_history) > 0:
                    self.plot_losses(epoch)
                self.visualize_reconstructions(epoch)
            # DDP barrier: 确保 rank 0 可视化完成前其他 rank 不进入下一 epoch
            if self.world_size > 1:
                dist.barrier()

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_recon = 0.0
        total_samples = 0

        # 固定随机状态，保证验证结果可复现
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        random_state = random.getstate()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            masked_batch, mask_info = self.masking(batch)
            output = self.model(masked_batch, mask_info)
            loss_recon = reconstruction_loss(output["reconstructions"], batch, mask_info)
            total_recon += loss_recon.item()
            total_samples += 1

        # 恢复训练随机状态
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(random_state)

        avg_recon = total_recon / max(total_samples, 1)

        # 跨rank聚合
        if self.world_size > 1:
            recon_tensor = torch.tensor(avg_recon, device=self.device)
            dist.all_reduce(recon_tensor, op=dist.ReduceOp.SUM)
            avg_recon = recon_tensor.item() / self.world_size

        if self.rank == 0:
            print(f"[Val] recon_loss={avg_recon:.4f}")
        self.model.train()  # 恢复训练模式（所有 rank 统一）
        return {"recon_loss": avg_recon}

    def save_checkpoint(self, epoch: int) -> None:
        path = self.output_dir / f"epoch_{epoch}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step_count": self.scheduler.step_count,
            "scheduler_base_lr": self.scheduler.base_lr,
            "global_step": self.global_step,
            "loss_history": self.loss_history,
        }
        torch.save(state, path)
        print(f"[Save] Checkpoint saved to {path}")

        # 只保留最近 3 个 checkpoint，删除旧的
        if self.rank == 0:
            ckpt_files = sorted(
                self.output_dir.glob("epoch_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            if len(ckpt_files) > 3:
                for old_ckpt in ckpt_files[:-3]:
                    old_ckpt.unlink()
                    print(f"[Cleanup] Removed old checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model_state_dict"]
        # 过滤掉形状不匹配的参数（如spatial_head维度变化）
        model_state = (self.model.module if hasattr(self.model, "module") else self.model).state_dict()
        filtered_state = {}
        shape_mismatch = []
        for k, v in state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    shape_mismatch.append(f"{k}: ckpt{v.shape} != model{model_state[k].shape}")
        if shape_mismatch:
            print(f"[Load] Shape mismatch (skipped {len(shape_mismatch)} params):")
            for m in shape_mismatch[:5]:
                print(f"  {m}")
        # strict=False: 跳过不匹配的参数
        missing, unexpected = (self.model.module if hasattr(self.model, "module") else self.model).load_state_dict(filtered_state, strict=False)
        if missing:
            print(f"[Load] Missing keys (will be initialized): {missing}")
        if unexpected:
            print(f"[Load] Unexpected keys (skipped): {unexpected}")
        if missing or shape_mismatch:
            # 模型架构变化（如新增cond_proj），optimizer state不匹配，重新初始化
            print("[Load] Model architecture changed, reinitializing optimizer")
            self.optimizer = build_optimizer(
                self.model.module if hasattr(self.model, "module") else self.model,
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )
            # 重建optimizer后必须更新scheduler引用，否则LR调度失效
            self.scheduler.optimizer = self.optimizer
        else:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        self.scheduler.step_count = ckpt.get("scheduler_step_count", 0)
        self.scheduler.base_lr = ckpt.get("scheduler_base_lr", self.cfg.training.lr)
        self.loss_history = ckpt.get("loss_history", [])
        print(f"[Load] Checkpoint loaded from {path}")

    def plot_losses(self, epoch: int) -> None:
        """绘制并保存损失曲线."""
        if len(self.loss_history) < 2:
            return
        epochs = [d["epoch"] for d in self.loss_history]
        vis_dir = self.output_dir / f"Epoch-{epoch}-VIS"
        vis_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Training Loss Curves (up to Epoch {epoch})", fontsize=14)

        metrics = [
            ("loss", "Total Loss", axes[0, 0]),
            ("recon", "Reconstruction Loss", axes[0, 1]),
            ("distill", "Distillation Loss", axes[0, 2]),
            ("uniform", "Uniformity Loss", axes[1, 0]),
            ("spatial_u", "Spatial Uniformity Loss", axes[1, 1]),
        ]
        for key, title, ax in metrics:
            values = [d[key] for d in self.loss_history]
            ax.plot(epochs, values, "b-o", markersize=3, linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        axes[1, 2].axis("off")

        plt.tight_layout()
        save_path = vis_dir / "loss_curve.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Viz] Loss curve saved to {save_path}")

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """将 [C, H, W] tensor 转为 [H, W, 3] numpy 图像，用于可视化."""
        arr = tensor.detach().cpu().numpy()
        # 取前3通道作为RGB，单通道则重复
        if arr.shape[0] >= 3:
            rgb = np.transpose(arr[:3], (1, 2, 0))
        else:
            gray = arr[0]
            rgb = np.stack([gray, gray, gray], axis=-1)
        # percentile stretch
        p_low, p_high = np.percentile(rgb, [2, 98])
        if p_high > p_low:
            rgb = np.clip((rgb - p_low) / (p_high - p_low), 0, 1)
        return rgb

    @torch.no_grad()
    def visualize_reconstructions(self, epoch: int) -> None:
        """可视化重建图像与原图对比，每20epoch调用."""
        if self.rank != 0:
            return
        self.model.eval()
        vis_dir = self.output_dir / f"Epoch-{epoch}-VIS"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 使用预取的固定验证 batch（避免每次 spawn DataLoader worker）
        if self.fixed_val_batch is None:
            return
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else None for k, v in self.fixed_val_batch.items()}

        # 固定随机状态（masking.py 已全部使用 torch 随机流）
        torch_state = torch.get_rng_state()
        if hasattr(torch, "npu") and torch.npu.is_available():
            npu_state = torch.npu.get_rng_state()
        torch.manual_seed(self.cfg.seed)
        masked_batch, mask_info = self.masking(batch)
        torch.set_rng_state(torch_state)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_rng_state(npu_state)

        output = self.model(masked_batch, mask_info)

        # 显示所有 source（不只是 decode_sources），对于没被 decode 的标注出来
        decode_sources_set = set(mask_info.get("decode_sources", []))
        all_sources = list(self.source_channels.keys())

        for source_name in all_sources:
            original = mask_info["original_batch"].get(source_name)
            if original is None:
                continue  # 数据本身缺失

            n_show = min(2, original.shape[0])
            fig = None
            try:
                fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
                if n_show == 1:
                    axes = axes.reshape(1, -1)

                for i in range(n_show):
                    orig_img = self._tensor_to_image(original[i, 0])
                    axes[i, 0].imshow(orig_img)
                    axes[i, 0].set_title(f"{source_name} Original")
                    axes[i, 0].axis("off")

                    if source_name in decode_sources_set and source_name in output.get("reconstructions", {}):
                        recon = output["reconstructions"][source_name]
                        recon_img = self._tensor_to_image(recon[i, 0])
                        diff_img = np.abs(orig_img - recon_img)
                        axes[i, 1].imshow(recon_img)
                        axes[i, 1].set_title(f"{source_name} Reconstruction")
                        axes[i, 2].imshow(diff_img)
                        axes[i, 2].set_title(f"{source_name} Diff")
                    else:
                        # 显示灰色占位图，标注该 source 本次未被 decode
                        gray = np.ones_like(orig_img) * 0.5
                        axes[i, 1].imshow(gray)
                        axes[i, 1].set_title(f"{source_name} (Not Decoded)")
                        axes[i, 2].imshow(gray)
                        axes[i, 2].set_title(f"{source_name} (No Diff)")

                    axes[i, 1].axis("off")
                    axes[i, 2].axis("off")

                plt.tight_layout()
                save_path = vis_dir / f"recon_{source_name}.png"
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                print(f"[Viz] Reconstruction visualization saved to {save_path}")
            finally:
                if fig is not None:
                    plt.close(fig)

        self.model.train()
