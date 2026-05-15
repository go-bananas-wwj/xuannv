"""V7 Phase 1 DDP 训练器 — 10 blocks + EMA Teacher + VICReg + KoLeo.

核心设计:
- 10 blocks (前2禁用 Space)
- EMA Teacher (momentum=0.996)
- VICReg + KoLeo 替代四件套
- checkpoint 策略: best 2 + latest 2
- ETA 显示
"""
from __future__ import annotations

import os
import time
import glob
import copy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    wandb = None

from src.config import Config, load_config
from src.data.builder import build_dataloader
from src.models.model import AEFModel
from src.training.losses import (
    reconstruction_loss, consistency_loss,
    classification_loss, pre_norm_uniformity_loss, directional_uniformity_loss,
    raw_uniformity_loss,
    temporal_contrastive_loss, temporal_cosine_pixel_loss, pixel_temporal_info_nce_loss,
)
from src.training.vicreg_loss import koleo_loss
from src.training.loops import compute_recon_loss
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


class DDPv7Trainer:
    """V7 Phase 1 DDP 训练器."""

    def __init__(self, cfg: Config, local_rank: int = 0, wandb_config: dict | None = None) -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"npu:{local_rank}")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0

        # 模型
        self.model = AEFModel(cfg).to(self.device)
        self.model = DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)

        # EMA Teacher (复制原始 module，避免 DDP wrapper 问题)
        self.teacher = copy.deepcopy(self.model.module).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 优化器与调度器
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # 输出目录
        self.output_dir = Path(cfg.experiment.output_dir)
        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # best checkpoint 追踪
        self.best_checkpoints: list[tuple[float, Path]] = []

        # 计时
        self.epoch_times: list[float] = []

        # wandb 初始化 (仅 rank 0)
        self.wandb_run = None
        if self.global_rank == 0 and _HAS_WANDB and wandb_config:
            wc = wandb_config
            mode = wc.get("mode", "online")
            try:
                self.wandb_run = wandb.init(
                    project=wc.get("project", "xuannv-backbone"),
                    entity=wc.get("entity"),
                    name=wc.get("name") or cfg.experiment.name,
                    config=cfg.to_dict() if hasattr(cfg, "to_dict") else {"config_path": wc.get("config_path", "")},
                    dir=str(self.output_dir),
                    mode=mode,
                    resume=wc.get("resume", False),
                )
                print(f"[wandb] Initialized: {self.wandb_run.url if hasattr(self.wandb_run, 'url') else 'offline'}")
            except Exception as e:
                print(f"[wandb] Init failed: {e}")
                self.wandb_run = None

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        """EMA 更新 teacher."""
        for pt, ps in zip(self.teacher.parameters(), self.model.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)

    def train_epoch(self, epoch: int, dataloader) -> dict[str, float]:
        self.model.train()
        t = self.cfg.training

        loss_accum: dict[str, float] = {}
        n_steps = 0
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            # 移动 batch 到 device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 学习率
            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # 前向传播
            with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
                out = self.model(
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
                    out.reconstructions,
                    batch["target_images"],
                    batch["target_mask"],
                    batch.get("target_loss_type"),
                    self.cfg.data.num_classes,
                )

                # 跨 GPU 聚合 embedding
                gathered = out.embedding
                gathered_pre_norm = out.pre_norm_embedding

                if dist.is_initialized() and self.world_size > 1:
                    all_emb = [torch.zeros_like(gathered) for _ in range(self.world_size)]
                    dist.all_gather(all_emb, gathered)
                    all_emb[self.local_rank] = gathered
                    gathered = torch.cat(all_emb, dim=0)

                    all_pn = [torch.zeros_like(gathered_pre_norm) for _ in range(self.world_size)]
                    dist.all_gather(all_pn, gathered_pre_norm)
                    all_pn[self.local_rank] = gathered_pre_norm
                    gathered_pre_norm = torch.cat(all_pn, dim=0)

                # === V7: Variance + Covariance + KoLeo ===
                gamma = getattr(t, 'vicreg_gamma', 1.0)
                std = torch.sqrt(gathered_pre_norm.var(dim=0) + 1e-4)
                vicreg_var = torch.mean(F.relu(gamma - std))

                z = gathered_pre_norm - gathered_pre_norm.mean(dim=0)
                cov = (z.T @ z) / (z.shape[0] - 1)
                vicreg_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / gathered_pre_norm.shape[1]

                lambda_var = getattr(t, 'vicreg_lambda_var', 1.0)
                lambda_cov = getattr(t, 'vicreg_lambda_cov', 0.04)
                vicreg = lambda_var * vicreg_var + lambda_cov * vicreg_cov
                vicreg_inv = torch.tensor(0.0, device=self.device)

                # KoLeo
                koleo = koleo_loss(gathered_pre_norm)

                # ★ Round8: uniformity 在 query norm (L2) 空间计算
                query_norm = F.normalize(gathered_pre_norm, p=2, dim=1)
                pre_unif = pre_norm_uniformity_loss(gathered_pre_norm)
                enc_unif = directional_uniformity_loss(gathered_pre_norm)
                raw_unif = raw_uniformity_loss(query_norm.float())

                # Teacher forward (无梯度)
                with torch.no_grad():
                    teacher_out = self.teacher(
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
                    teacher_emb = teacher_out.embedding
                
                # 跨 GPU 聚合 teacher embedding
                if dist.is_initialized() and self.world_size > 1:
                    all_t = [torch.zeros_like(teacher_emb) for _ in range(self.world_size)]
                    dist.all_gather(all_t, teacher_emb)
                    all_t[self.local_rank] = teacher_emb
                    teacher_emb = torch.cat(all_t, dim=0)
                
                # 其他损失
                consist = consistency_loss(teacher_emb, gathered)
                cls = classification_loss(out.logits, batch["label"])
                aux_cls = classification_loss(out.aux_logits, batch["label"]) if out.aux_logits is not None else torch.tensor(0.0, device=self.device)
                bn_cls = classification_loss(out.bottleneck_logits, batch["label"]) if out.bottleneck_logits is not None else torch.tensor(0.0, device=self.device)

                # 时序对比损失
                temporal_w = getattr(t, 'temporal_magnitude_weight', 0.0)
                temporal = torch.tensor(0.0, device=self.device)
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
                        # 使用 pixel-level cosine loss（无 hinge，持续有梯度）
                        temporal = temporal_cosine_pixel_loss(emb_w1, emb_w2, temperature=getattr(t, 'temporal_magnitude_temperature', 0.1))
                    except Exception as e:
                        if dist.get_rank() == 0:
                            print(f"  [Temporal] Error: {e}")

                # Recon warmup
                recon_warmup = min(1.0, (epoch + 1) / max(t.recon_warmup_epochs, 1))
                recon_weight = t.reconstruction_weight * recon_warmup

                # === V7 总损失 ===
                pre_norm_uniform_w = getattr(t, 'pre_norm_uniform_weight', 0.0)
                total = (
                    recon_weight * recon
                    + t.vicreg_weight * vicreg
                    + t.koleo_weight * koleo
                    + t.consistency_weight * consist
                    + t.classification_weight * cls
                    + t.aux_classification_weight * aux_cls
                    + t.bottleneck_cls_weight * bn_cls
                    + temporal_w * temporal
                    + pre_norm_uniform_w * raw_unif
                )

            # 梯度累积
            total = total / t.gradient_accumulation_steps
            total.backward()

            if (step + 1) % t.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # EMA 更新 teacher
                self.update_teacher(momentum=getattr(t, 'teacher_momentum', 0.996))

            # 累积损失
            n_steps += 1
            loss_accum["total"] = loss_accum.get("total", 0.0) + total.item() * t.gradient_accumulation_steps
            loss_accum["recon"] = loss_accum.get("recon", 0.0) + recon.item()
            loss_accum["vicreg"] = loss_accum.get("vicreg", 0.0) + vicreg.item()
            loss_accum["vicreg_var"] = loss_accum.get("vicreg_var", 0.0) + vicreg_var.item()
            loss_accum["vicreg_cov"] = loss_accum.get("vicreg_cov", 0.0) + vicreg_cov.item()
            loss_accum["koleo"] = loss_accum.get("koleo", 0.0) + koleo.item()
            loss_accum["pre_unif"] = loss_accum.get("pre_unif", 0.0) + pre_unif.item()
            loss_accum["enc_unif"] = loss_accum.get("enc_unif", 0.0) + enc_unif.item()
            loss_accum["raw_unif"] = loss_accum.get("raw_unif", 0.0) + raw_unif.item()
            loss_accum["consist"] = loss_accum.get("consist", 0.0) + consist.item()
            loss_accum["cls"] = loss_accum.get("cls", 0.0) + cls.item()
            loss_accum["aux_cls"] = loss_accum.get("aux_cls", 0.0) + aux_cls.item()
            loss_accum["bn_cls"] = loss_accum.get("bn_cls", 0.0) + bn_cls.item()
            loss_accum["temporal"] = loss_accum.get("temporal", 0.0) + temporal.item()

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
        loss_accum["epoch_time"] = time.time() - epoch_start

        # wandb log (rank 0)
        if self.global_rank == 0 and self.wandb_run is not None:
            try:
                log_dict = {f"train/{k}": v for k, v in loss_accum.items()}
                log_dict["train/epoch"] = epoch
                self.wandb_run.log(log_dict)
            except Exception as e:
                print(f"[wandb] Log failed: {e}")

        return loss_accum

    def save_checkpoint(self, epoch: int, losses: dict) -> None:
        if self.global_rank != 0:
            return

        # 保存前清零残留梯度
        self.optimizer.zero_grad()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "teacher_state_dict": self.teacher.state_dict(),
            "losses": losses,
        }

        # 1. 保存当前 epoch
        path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # 2. 清理旧 epoch
        self._cleanup_old_epochs(keep_latest=2)

        # 3. 保存 best 2
        self._save_best_checkpoints(epoch, losses, checkpoint)

        # 4. wandb artifact (仅保留最新 checkpoint 引用)
        if self.wandb_run is not None:
            try:
                artifact = wandb.Artifact(
                    name=f"checkpoint-epoch-{epoch}",
                    type="model",
                    metadata={"epoch": epoch, **losses},
                )
                artifact.add_file(str(path))
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                print(f"[wandb] Artifact log failed: {e}")

    def _cleanup_old_epochs(self, keep_latest: int = 2) -> None:
        """删除旧的 epoch 文件，只保留最新的 N 个."""
        import re
        all_files = glob.glob(str(self.output_dir / "epoch_*.pt"))
        epoch_files = [p for p in all_files if re.match(r"epoch_\d+\.pt$", Path(p).name)]
        epoch_files = sorted(
            epoch_files,
            key=lambda p: int(Path(p).stem.split("_")[1]),
            reverse=True,
        )
        for old_path in epoch_files[keep_latest:]:
            try:
                os.remove(old_path)
            except OSError:
                pass

    def _save_best_checkpoints(self, epoch: int, losses: dict, checkpoint: dict) -> None:
        """维护 best 2 checkpoints."""
        score = self._get_checkpoint_score(losses)
        if score is None:
            return

        best_path = self.output_dir / f"epoch_best_{epoch}.pt"
        torch.save(checkpoint, best_path)

        self.best_checkpoints.append((score, best_path))
        self.best_checkpoints.sort(key=lambda x: x[0])
        if len(self.best_checkpoints) > 2:
            for _, old_path in self.best_checkpoints[2:]:
                if old_path.exists():
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
            self.best_checkpoints = self.best_checkpoints[:2]

    def _get_checkpoint_score(self, losses: dict) -> float | None:
        """评估 checkpoint 质量的分数 (越小越好)."""
        v = losses.get("vicreg", 0.0)
        r = losses.get("recon", 1.0)
        if v < 1.0 and r < 0.5:
            return v + r * 2.0
        return None

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        
        # 使用 strict=False 加载，允许新增层（如从 8 blocks → 10 blocks）
        # 同时处理 size mismatch（前 N 层 fusion 输入通道变化）
        state_dict = checkpoint["model_state_dict"]
        model_dict = self.model.module.state_dict()
        
        # 过滤掉 shape 不匹配的层（前 N 层禁用 Space 导致 fusion 输入变化）
        filtered = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered[k] = v
                else:
                    if self.global_rank == 0:
                        print(f"  [Load] Skip {k}: ckpt {tuple(v.shape)} != model {tuple(model_dict[k].shape)}")
            else:
                if self.global_rank == 0:
                    print(f"  [Load] Skip missing key: {k}")
        
        # 加载匹配的权重
        model_dict.update(filtered)
        self.model.module.load_state_dict(model_dict, strict=False)
        
        if self.global_rank == 0:
            n_loaded = len(filtered)
            n_total = len(model_dict)
            print(f"  [Load] Loaded {n_loaded}/{n_total} params from checkpoint")
        
        # 尝试加载 optimizer state（正常 resume 时恢复，兼容性加载失败时重新初始化）
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.global_rank == 0:
                print("  [Load] Optimizer state restored")
        except Exception as e:
            if self.global_rank == 0:
                print(f"  [Load] Optimizer reinitialized: {e}")
        
        # 尝试加载 scheduler state
        if self.scheduler:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if self.global_rank == 0:
                    print("  [Load] Scheduler state restored")
            except Exception as e:
                if self.global_rank == 0:
                    print(f"  [Load] Scheduler reinitialized: {e}")
        
        # 加载 EMA teacher 状态（兼容旧 checkpoint 的 module. 前缀）
        if "teacher_state_dict" in checkpoint:
            t_state = checkpoint["teacher_state_dict"]
            # 去掉可能的 module. 前缀（旧版 teacher 是 DDP wrapper）
            cleaned = {}
            for k, v in t_state.items():
                new_k = k[7:] if k.startswith("module.") else k
                cleaned[new_k] = v
            try:
                self.teacher.load_state_dict(cleaned, strict=False)
                if self.global_rank == 0:
                    print("  [Load] Teacher state restored")
            except Exception as e:
                if self.global_rank == 0:
                    print(f"  [Load] Teacher reinitialized: {e}")
        
        # 恢复后清零残留梯度
        self.optimizer.zero_grad()
        return checkpoint.get("epoch", 0)


def format_eta(elapsed: float, current: int, total: int) -> str:
    """格式化 ETA 时间."""
    if current <= 0:
        return "N/A"
    avg_time = elapsed / current
    remaining = total - current
    eta_seconds = avg_time * remaining
    hours = int(eta_seconds // 3600)
    minutes = int((eta_seconds % 3600) // 60)
    return f"{hours}h{minutes:02d}m"


def train_worker(local_rank: int, cfg: Config, resume_from: str | None = None, wandb_config: dict | None = None) -> None:
    """Single worker training function."""
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    torch.npu.set_device(local_rank)

    trainer = DDPv7Trainer(cfg, local_rank, wandb_config=wandb_config)

    # 处理预训练权重或 resume
    start_epoch = 0
    if resume_from and local_rank == 0:
        if Path(resume_from).exists():
            start_epoch = trainer.load_checkpoint(resume_from)
            print(f"[ddp] Resumed from {resume_from}, starting at epoch {start_epoch+1}")
        else:
            print(f"[ddp] Warning: resume path not found: {resume_from}")
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
    sampler = dataloader.sampler

    t = cfg.training
    best_score = float("inf")
    patience_counter = 0
    total_start = time.time()

    for epoch in range(start_epoch, t.epochs):
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        epoch_start = time.time()

        losses = trainer.train_epoch(epoch, dataloader)
        epoch_time = time.time() - epoch_start
        trainer.epoch_times.append(epoch_time)
        elapsed = time.time() - total_start
        eta = format_eta(elapsed, epoch + 1 - start_epoch, t.epochs - start_epoch)

        if dist.get_rank() == 0:
            msg = (
                f"Epoch {epoch+1}/{t.epochs} | "
                f"Loss: {losses['total']:.4f} | "
                f"Recon: {losses['recon']:.4f} | "
                f"VICReg: {losses['vicreg']:.4f} (var={losses['vicreg_var']:.4f} cov={losses['vicreg_cov']:.4f}) | "
                f"KoLeo: {losses['koleo']:.4f} | "
                f"PreUnif: {losses['pre_unif']:.4f} | "
                f"EncUnif: {losses['enc_unif']:.4f} | "
                f"RawUnif: {losses['raw_unif']:.4f} | "
                f"Consist: {losses['consist']:.4f} | "
                f"Cls: {losses['cls']:.4f} | "
                f"Temporal: {losses.get('temporal', 0):.4f} | "
                f"LR: {losses['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s | "
                f"Elapsed: {elapsed/60:.1f}m | ETA: {eta}"
            )
            print(msg, flush=True)

        if (epoch + 1) % t.save_every == 0 or (epoch + 1) % getattr(t, 'checkpoint_interval', 20) == 0:
            trainer.save_checkpoint(epoch, losses)

        # 早停监控
        current_vicreg = losses.get("vicreg", float("inf"))
        if current_vicreg < best_score:
            best_score = current_vicreg
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= t.early_stop_patience:
            if dist.get_rank() == 0:
                print(f"Early stopping at epoch {epoch+1}")
            break

        if trainer.scheduler is not None:
            trainer.scheduler.step()

    if trainer.wandb_run is not None:
        try:
            trainer.wandb_run.finish()
            print("[wandb] Run finished")
        except Exception as e:
            print(f"[wandb] Finish error: {e}")

    dist.destroy_process_group()
