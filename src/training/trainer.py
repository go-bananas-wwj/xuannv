"""DDP 训练器 — 改进版损失组合.

核心改进:
- raw_uniformity_loss 在 pre-norm 空间计算 (绕过 L2 梯度屏障)
- 反坍缩四件套: raw_uniformity + decorrelation + variance + orthogonality
- 渐进式 recon warmup
- 最佳模型保存 (基于 uniform + recon 平衡)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


from src.config import Config, load_config
from src.data.builder import build_dataloader
from src.models.model import AEFModel
from src.training.losses import (
    raw_uniformity_loss, decorrelation_loss, variance_regularizer,
    bottleneck_orthogonality_loss, reconstruction_loss, consistency_loss,
    classification_loss, pre_norm_uniformity_loss, directional_uniformity_loss,
    temporal_contrastive_loss, temporal_info_nce_loss, pixel_temporal_info_nce_loss,
)
from src.training.loops import compute_recon_loss
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


class DDPTrainer:
    """DDP 分布式训练器."""

    def __init__(self, cfg: Config, local_rank: int = 0) -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0

        # 模型
        self.model = AEFModel(cfg).to(self.device)
        self.model = DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)

        # 加载预训练权重（仅模型，不加载优化器状态）
        pretrained_path = getattr(cfg, "pretrained", None)
        if pretrained_path and self.global_rank == 0:
            ckpt = torch.load(pretrained_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.module.load_state_dict(state_dict, strict=False)
            print(f"[ddp] Loaded pretrained weights from {pretrained_path}")
        if dist.is_initialized():
            dist.barrier()

        # 优化器与调度器
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # 输出目录
        self.output_dir = Path(cfg.experiment.output_dir)
        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        t = self.cfg.training

        loss_accum: dict[str, float] = {}
        n_steps = 0

        for step, batch in enumerate(dataloader):
            # 移动 batch 到 device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 学习率
            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # 前向传播
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                teacher_out = self.model(
                    source_frames=batch["source_frames"],
                    source_timestamps_ms=batch["source_timestamps_ms"],
                    source_frame_mask=batch["source_frame_mask"],
                    source_input_mask=batch["source_input_mask"],
                    source_type_ids=batch["source_type_ids"],
                    valid_start_ms=batch["valid_start_ms"],
                    valid_end_ms=batch["valid_end_ms"],
                    target_relative_time=batch["target_relative_time"],
                    target_metadata=batch["target_metadata"],
                    target_loss_type=batch.get("target_loss_type"),
                    target_source_idx=batch.get("target_source_idx"),
                )

                # 重建损失
                recon = compute_recon_loss(
                    teacher_out.reconstructions,
                    batch["target_images"],
                    batch["target_mask"],
                    batch.get("target_loss_type"),
                    self.cfg.data.num_classes,
                )

                # 跨 GPU 聚合 embedding
                gathered = teacher_out.embedding
                gathered_pre_norm = teacher_out.pre_norm_embedding

                if dist.is_initialized() and self.world_size > 1:
                    all_emb = [torch.zeros_like(gathered) for _ in range(self.world_size)]
                    dist.all_gather(all_emb, gathered)
                    all_emb[self.local_rank] = gathered
                    gathered = torch.cat(all_emb, dim=0)

                    all_pn = [torch.zeros_like(gathered_pre_norm) for _ in range(self.world_size)]
                    dist.all_gather(all_pn, gathered_pre_norm)
                    all_pn[self.local_rank] = gathered_pre_norm
                    gathered_pre_norm = torch.cat(all_pn, dim=0)

                # === 核心: 反坍缩四件套 ===
                raw_unif = raw_uniformity_loss(gathered_pre_norm)
                decorr = decorrelation_loss(gathered_pre_norm)
                var_reg = variance_regularizer(gathered_pre_norm)
                bnw = self.model.module.bottleneck.to_embedding.weight
                orth = bottleneck_orthogonality_loss(bnw)
                pre_unif = pre_norm_uniformity_loss(gathered_pre_norm)
                enc_unif = directional_uniformity_loss(gathered_pre_norm)
                consist = consistency_loss(gathered.detach(), gathered)
                cls = classification_loss(teacher_out.logits, batch["label"])
                aux_cls = classification_loss(teacher_out.aux_logits, batch["label"]) if teacher_out.aux_logits is not None else torch.tensor(0.0, device=self.device)
                bn_cls = classification_loss(teacher_out.bottleneck_logits, batch["label"]) if teacher_out.bottleneck_logits is not None else torch.tensor(0.0, device=self.device)

                # 时序对比损失
                temporal_w = getattr(t, 'temporal_contrastive_weight', 0.0)
                temporal_temp = getattr(t, 'temporal_contrastive_temperature', 0.1)
                temporal_loss_type = getattr(t, 'temporal_loss_type', 'hinge')
                pixel_temporal_w = getattr(t, 'pixel_temporal_weight', 0.0)
                pixel_temporal_samples = getattr(t, 'pixel_temporal_samples', 16)
                temporal = torch.tensor(0.0, device=self.device)
                pixel_temporal = torch.tensor(0.0, device=self.device)
                if temporal_w > 0 and "valid_start_w1" in batch:
                    try:
                        emb_w1, emb_w2, pre_w1, pre_w2 = self.model.module.encode_dual_window(
                            source_frames=batch["source_frames"],
                            source_timestamps_ms=batch["source_timestamps_ms"],
                            source_frame_mask=batch["source_frame_mask"],
                            source_input_mask=batch["source_input_mask"],
                            source_type_ids=batch["source_type_ids"],
                            valid_start_w1=batch["valid_start_w1"],
                            valid_end_w1=batch["valid_end_w1"],
                            valid_start_w2=batch["valid_start_w2"],
                            valid_end_w2=batch["valid_end_w2"],
                        )
                        if temporal_loss_type == 'info_nce_antidiagonal':
                            temporal = temporal_info_nce_loss(pre_w1, pre_w2, temperature=temporal_temp)
                            if pixel_temporal_w > 0:
                                pixel_temporal = pixel_temporal_info_nce_loss(
                                    pre_w1, pre_w2, temperature=temporal_temp, num_samples=pixel_temporal_samples
                                )
                        else:
                            temporal = temporal_contrastive_loss(emb_w1, emb_w2, temperature=temporal_temp)
                    except Exception as e:
                        if dist.get_rank() == 0:
                            print(f"  [Temporal] Error: {e}")
                        temporal = torch.tensor(0.0, device=self.device)
                        pixel_temporal = torch.tensor(0.0, device=self.device)

                # Recon warmup
                recon_warmup = min(1.0, (epoch + 1) / max(t.recon_warmup_epochs, 1))
                recon_weight = t.reconstruction_weight * recon_warmup

                # 总损失
                total = (
                    recon_weight * recon
                    + t.uniformity_weight * raw_unif
                    + t.variance_weight * var_reg
                    + t.decorrelation_weight * decorr
                    + t.orthogonality_weight * orth
                    + t.pre_norm_uniform_weight * pre_unif
                    + t.encoder_uniform_weight * enc_unif
                    + t.consistency_weight * consist
                    + t.classification_weight * cls
                    + t.aux_classification_weight * aux_cls
                    + t.bottleneck_cls_weight * bn_cls
                    + temporal_w * temporal
                    + pixel_temporal_w * pixel_temporal
                )

            # 梯度累积
            total = total / t.gradient_accumulation_steps
            total.backward()

            if (step + 1) % t.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 累积损失
            n_steps += 1
            loss_accum["total"] = loss_accum.get("total", 0.0) + total.item() * t.gradient_accumulation_steps
            loss_accum["recon"] = loss_accum.get("recon", 0.0) + recon.item()
            loss_accum["raw_unif"] = loss_accum.get("raw_unif", 0.0) + raw_unif.item()
            loss_accum["decorr"] = loss_accum.get("decorr", 0.0) + decorr.item()
            loss_accum["var_reg"] = loss_accum.get("var_reg", 0.0) + var_reg.item()
            loss_accum["orth"] = loss_accum.get("orth", 0.0) + orth.item()
            loss_accum["pre_unif"] = loss_accum.get("pre_unif", 0.0) + pre_unif.item()
            loss_accum["enc_unif"] = loss_accum.get("enc_unif", 0.0) + enc_unif.item()
            loss_accum["consist"] = loss_accum.get("consist", 0.0) + consist.item()
            loss_accum["cls"] = loss_accum.get("cls", 0.0) + cls.item()
            loss_accum["aux_cls"] = loss_accum.get("aux_cls", 0.0) + aux_cls.item()
            loss_accum["bn_cls"] = loss_accum.get("bn_cls", 0.0) + bn_cls.item()
            loss_accum["temporal"] = loss_accum.get("temporal", 0.0) + temporal.item()
            loss_accum["pixel_temporal"] = loss_accum.get("pixel_temporal", 0.0) + pixel_temporal.item()

        # 平均
        for k in loss_accum:
            loss_accum[k] /= n_steps

        # 同步
        if dist.is_initialized():
            for k in loss_accum:
                v = torch.tensor(loss_accum[k], device=self.device)
                dist.all_reduce(v, op=dist.ReduceOp.AVG)
                loss_accum[k] = v.item()

        loss_accum["lr"] = lr
        return loss_accum

    def save_checkpoint(self, epoch: int, losses: dict) -> None:
        if self.global_rank != 0:
            return
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": losses,
        }
        path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        best_path = self.output_dir / "best.pt"
        current_best = self._is_best(losses)
        if current_best:
            torch.save(checkpoint, best_path)

    def _is_best(self, losses: dict) -> bool:
        t = self.cfg.training
        if not t.save_best_balanced:
            return True
        u = losses.get("raw_unif", 0.0)
        r = losses.get("recon", 1.0)
        return t.best_balanced_uniform_min < u < t.best_balanced_uniform_max and r < 0.5

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0)


def train_worker(local_rank: int, cfg: Config, resume_from: str | None = None) -> None:
    """Single worker training function."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    trainer = DDPTrainer(cfg, local_rank)

    start_epoch = 0
    if resume_from and local_rank == 0:
        start_epoch = trainer.load_checkpoint(resume_from)
        print(f"[ddp] Resumed from {resume_from}, starting at epoch {start_epoch+1}")
    if dist.is_initialized():
        t = torch.tensor(start_epoch, device=trainer.device)
        dist.broadcast(t, src=0)
        start_epoch = t.item()

    dataloader = build_dataloader(
        cfg,
        training=True,
        distributed=dist.is_initialized(),
        world_size=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
    )

    t = cfg.training
    best_uniform = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, t.epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.time()

        losses = trainer.train_epoch(epoch, dataloader)
        epoch_time = time.time() - epoch_start

        if dist.get_rank() == 0:
            msg = (
                f"Epoch {epoch+1}/{t.epochs} | "
                f"Loss: {losses['total']:.4f} | "
                f"Recon: {losses['recon']:.4f} | "
                f"RawUnif: {losses['raw_unif']:.4f} | "
                f"PreUnif: {losses['pre_unif']:.4f} | "
                f"Decor: {losses['decorr']:.4f} | "
                f"Var: {losses['var_reg']:.4f} | "
                f"Orth: {losses['orth']:.4f} | "
                f"Cls: {losses['cls']:.4f} | "
                f"BNCls: {losses['bn_cls']:.4f} | "
                f"Temporal: {losses.get('temporal', 0):.4f} | "
                f"PixTemp: {losses.get('pixel_temporal', 0):.4f} | "
                f"LR: {losses['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            print(msg, flush=True)

        if (epoch + 1) % t.save_every == 0 or (epoch + 1) % t.checkpoint_interval == 0:
            trainer.save_checkpoint(epoch, losses)

        current_unif = losses.get("raw_unif", float("inf"))
        if current_unif < best_uniform:
            best_uniform = current_unif
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= t.early_stop_patience:
            if dist.get_rank() == 0:
                print(f"Early stopping at epoch {epoch+1}")
            break

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_worker(args.local_rank, cfg)


if __name__ == "__main__":
    main()
