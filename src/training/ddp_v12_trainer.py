"""DDP V12 训练器 — 纯动态重建基线.

核心设计:
- 只重建 S2/S1/Landsat 3个动态源
- 损失: Reconstruction(1.0) + BatchUniformity(0.1) + Consistency(0.2)
- Teacher-Student 一致性为核心机制（对齐AEF原文）
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from src.config import Config
from src.models.model import AEFModel
from src.training.losses import (
    reconstruction_loss,
    batch_uniformity_loss_l2,
    consistency_loss,
)
from src.training.memory_bank import EmbeddingMemoryBank
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


# ---------------------------------------------------------------------------
# 源 → 重建权重映射
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_RECON_WEIGHTS = [1.0, 1.0, 1.0]


def _build_student_view(
    source_frames: torch.Tensor,
    source_timestamps_ms: torch.Tensor,
    source_frame_mask: torch.Tensor,
    source_input_mask: torch.Tensor,
    drop_rate: float,
    source_drop_rate: float,
    front_drop_prob: float,
    back_drop_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """构建 Student 的扰动输入视图 — 对齐 AEF 原文 S2.2.5."""
    frames = source_frames.clone()
    frame_mask = source_frame_mask.clone()
    input_mask = source_input_mask.clone()
    stats = {
        "source_drop_ratio": 0.0,
        "frame_drop_ratio": 0.0,
        "front_cut_ratio": 0.0,
        "back_cut_ratio": 0.0,
    }

    B, S, T = frame_mask.shape[:3]
    device = input_mask.device

    # Stage 1: 源级别 drop (AEF: S2 永不 drop, S1/Landsat 30%)
    if source_drop_rate > 0:
        for s_idx in range(S):
            if s_idx == 0:
                continue
            drop = torch.rand(B, device=device) < source_drop_rate
            input_mask[drop, s_idx] = False
        stats["source_drop_ratio"] = float((~input_mask).float().mean().item())
        for batch_index in range(B):
            if not input_mask[batch_index].any():
                input_mask[batch_index, 0] = True

    # Stage 2: 三种策略选一种
    strat = torch.randint(0, 3, (1,)).item()

    if strat == 0:
        for s_idx in range(S):
            frac = 0.5 if s_idx == 0 else 0.3
            for b in range(B):
                if not input_mask[b, s_idx]:
                    continue
                drops = torch.rand(T, device=device) < frac
                frame_mask[b, s_idx, drops] = False
        stats["frame_drop_ratio"] = float((~frame_mask & source_frame_mask).float().mean().item())

    elif strat in (1, 2):
        for b in range(B):
            for s_idx in range(S):
                if not input_mask[b, s_idx]:
                    continue
                valid_ts = [t for t in range(T) if source_frame_mask[b, s_idx, t]]
                if len(valid_ts) <= 1:
                    continue
                ts_vals = [source_timestamps_ms[b, s_idx, t].item() for t in valid_ts]
                sorted_pairs = sorted(zip(ts_vals, valid_ts))
                mid_idx = len(sorted_pairs) // 2
                if strat == 1:
                    keep_ts = set(t for _, t in sorted_pairs[:mid_idx])
                else:
                    keep_ts = set(t for _, t in sorted_pairs[mid_idx:])
                for t in range(T):
                    if t not in keep_ts:
                        frame_mask[b, s_idx, t] = False
        if strat == 1:
            stats["back_cut_ratio"] = 0.25
        else:
            stats["front_cut_ratio"] = 0.25

    # Stage 3: 截断前后段
    if front_drop_prob > 0 or back_drop_prob > 0:
        batch_size = frame_mask.shape[0]
        time_steps = frame_mask.shape[2]
        front_hits = 0
        back_hits = 0
        for b in range(batch_size):
            for s in range(frame_mask.shape[1]):
                valid_ts = [t for t in range(time_steps) if frame_mask[b, s, t]]
                if len(valid_ts) <= 1:
                    continue
                ts_sorted = sorted(valid_ts, key=lambda t: source_timestamps_ms[b, s, t].item())
                front_cut = 0
                if front_drop_prob > 0 and torch.rand(1).item() < front_drop_prob:
                    front_cut = max(1, len(ts_sorted) // 4)
                    front_hits += 1
                back_cut = 0
                if back_drop_prob > 0 and torch.rand(1).item() < back_drop_prob:
                    back_cut = max(1, len(ts_sorted) // 4)
                    back_hits += 1
                keep_ts = set(ts_sorted[front_cut:len(ts_sorted)-back_cut])
                for t in range(time_steps):
                    if t not in keep_ts:
                        frame_mask[b, s, t] = False
        stats["front_cut_ratio"] = front_hits / max(batch_size * frame_mask.shape[1], 1)
        stats["back_cut_ratio"] = back_hits / max(batch_size * frame_mask.shape[1], 1)

    return frames, frame_mask, input_mask, stats


class DDPv12Trainer:
    """DDP V12 训练器 — 纯动态重建基线."""

    def __init__(self, cfg: Config, local_rank: int = 0) -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"npu:{local_rank}")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Student 模型
        self.model = AEFModel(cfg).to(self.device)
        self.model = DistributedDataParallel(
            self.model, device_ids=[local_rank], find_unused_parameters=True
        )

        # EMA Teacher
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = getattr(cfg.training, "teacher_momentum", 0.996)

        # 优化器
        self.optimizer = build_optimizer(self.model.module, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # GradScaler
        self.scaler = torch_npu.npu.amp.GradScaler(init_scale=2**18)

        # 输出目录
        self.output_dir = Path(cfg.experiment.output_dir)
        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 源特定重建权重
        self.source_recon_weights = torch.tensor(
            getattr(cfg.training, 'source_recon_weights', DEFAULT_SOURCE_RECON_WEIGHTS),
            dtype=torch.float32,
            device=self.device,
        )

        # Memory Bank — 扩大 uniformity 的有效 batch
        emb_dim = getattr(cfg.model, 'embedding_dim', 128)
        self.memory_bank = EmbeddingMemoryBank(K=512, dim=emb_dim, device=self.device)

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.teacher_momentum
        for p_t, p_s in zip(self.teacher.parameters(), self.model.module.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)

    def _reduce_loss_dict(self, loss_dict: dict) -> dict:
        if not dist.is_initialized() or self.world_size <= 1:
            return loss_dict
        reduced = {}
        for k, v in loss_dict.items():
            t = torch.tensor(v, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            reduced[k] = t.item()
        return reduced

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        t = self.cfg.training
        accum_steps = getattr(t, "gradient_accumulation_steps", 2)
        loss_accum: dict[str, float] = {}
        n_steps = 0

        # Kappa
        kappa = getattr(t, 'kappa_start', 0.0) or getattr(self.cfg.model, 'vmf_kappa', 2000.0)
        self.model.module.bottleneck.kappa = kappa
        self.teacher.bottleneck.kappa = kappa

        recon_w = t.reconstruction_weight
        consist_w = t.consistency_weight
        uniform_w = t.batch_uniformity_weight
        recon_warmup = min(1.0, (epoch + 1) / max(getattr(t, 'recon_warmup_epochs', 10), 1))

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=True):
                # Teacher forward
                has_dual = all(k in batch for k in ['valid_start_w1', 'valid_end_w1', 'valid_start_w2', 'valid_end_w2'])
                with torch.no_grad():
                    teacher_out = self.teacher(
                        source_frames=batch["source_frames"],
                        source_timestamps_ms=batch["source_timestamps_ms"],
                        source_frame_mask=batch["source_frame_mask"],
                        source_input_mask=batch["source_input_mask"],
                        source_type_ids=batch["source_type_ids"],
                        valid_start_ms=batch["valid_start_w1"] if has_dual else batch["valid_start_ms"],
                        valid_end_ms=batch["valid_end_w1"] if has_dual else batch["valid_end_ms"],
                        target_relative_time=batch["target_relative_time"],
                        target_metadata=batch["target_metadata"],
                        target_loss_type=batch.get("target_loss_type"),
                        target_source_idx=batch.get("target_source_idx"),
                    )

                # Student forward: 扰动输入
                student_frames, student_frame_mask, student_input_mask, perturb_stats = _build_student_view(
                    batch["source_frames"],
                    batch["source_timestamps_ms"],
                    batch["source_frame_mask"],
                    batch["source_input_mask"],
                    drop_rate=getattr(t, 'student_frame_drop_rate', 0.5),
                    source_drop_rate=getattr(t, 'student_source_drop_rate', 0.3),
                    front_drop_prob=getattr(t, 'student_front_drop_prob', 0.15),
                    back_drop_prob=getattr(t, 'student_back_drop_prob', 0.15),
                )

                student_out = self.model(
                    source_frames=student_frames,
                    source_timestamps_ms=batch["source_timestamps_ms"],
                    source_frame_mask=student_frame_mask,
                    source_input_mask=student_input_mask,
                    source_type_ids=batch["source_type_ids"],
                    valid_start_ms=batch["valid_start_ms"],
                    valid_end_ms=batch["valid_end_ms"],
                    target_relative_time=batch["target_relative_time"],
                    target_metadata=batch["target_metadata"],
                    target_loss_type=batch.get("target_loss_type"),
                    target_source_idx=batch.get("target_source_idx"),
                )

                # Reconstruction
                recon = self._compute_recon_loss(student_out.reconstructions, batch)

                # Consistency (teacher vs student embedding)
                consist = consistency_loss(teacher_out.embedding, student_out.embedding)

            # Batch Uniformity (with Memory Bank)
            embedding = student_out.embedding
            if dist.is_initialized() and self.world_size > 1:
                gathered_emb = [torch.zeros_like(embedding) for _ in range(self.world_size)]
                dist.all_gather(gathered_emb, embedding)
                gathered_emb = torch.cat(gathered_emb, dim=0)
            else:
                gathered_emb = embedding

            # Enqueue 当前 batch 的 embedding
            self.memory_bank.enqueue(gathered_emb.detach())

            # 合并当前 batch + memory bank
            bank_emb = self.memory_bank.get_all()
            if bank_emb.shape[0] > 0:
                all_emb = torch.cat([gathered_emb, bank_emb], dim=0)
            else:
                all_emb = gathered_emb

            uniform = batch_uniformity_loss_l2(all_emb)

            # Dummy loss for unused heads (avoid DDP unused parameter error)
            dummy = 0.0
            for head in [self.model.module.classification_head,
                         self.model.module.aux_cls_head,
                         self.model.module.bottleneck_cls_head]:
                for p in head.parameters():
                    dummy = dummy + p.sum() * 0.0

            total = (
                recon_w * recon_warmup * recon
                + consist_w * consist
                + uniform_w * uniform
                + dummy
            )

            if torch.isnan(total) or torch.isinf(total):
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"  [WARNING] NaN/Inf total at step {step}, skipping.")
                self.optimizer.zero_grad()
                continue

            total = total / accum_steps
            self.scaler.scale(total).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    getattr(t, "grad_clip_norm", 1.0)
                )
                has_nan_grad = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    if self.global_rank == 0:
                        print(f"  [WARNING] NaN/Inf gradient detected, skipping optimizer step.")
                    self.optimizer.zero_grad()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                self.update_teacher()

            for k, v in {
                "total": total.item() * accum_steps,
                "recon": recon.item(),
                "consist": consist.item(),
                "uniform": uniform.item(),
                "lr": lr,
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

            if self.global_rank == 0 and step % 20 == 0:
                print(f"  [Step {step}] recon={recon.item():.3f} "
                      f"consist={consist.item():.3f} uniform={uniform.item():.3f} "
                      f"perturb=[fd={perturb_stats['frame_drop_ratio']:.2f} "
                      f"sd={perturb_stats['source_drop_ratio']:.2f}] "
                      f"lr={lr:.6f}")

        loss_accum = {k: v / n_steps for k, v in loss_accum.items()}
        loss_accum = self._reduce_loss_dict(loss_accum)
        loss_accum["lr"] = lr
        return loss_accum

    def _compute_recon_loss(self, predictions, batch):
        """计算重建损失，应用源特定权重."""
        from src.training.loops import compute_recon_loss
        base_loss = compute_recon_loss(
            predictions, batch["target_images"], batch["target_mask"],
            batch.get("target_loss_type"), self.cfg.data.num_classes,
        )
        target_source_idx = batch.get("target_source_idx")
        if target_source_idx is not None:
            weights = self.source_recon_weights[target_source_idx]
            weight_factor = weights.mean()
            return base_loss * weight_factor
        return base_loss

    def save_checkpoint(self, epoch: int, losses: dict) -> None:
        if self.global_rank != 0:
            return
        path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "losses": losses,
        }, path)
        print(f"[ddp_v12] Saved checkpoint to {path}")

        ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_*.pt") if not p.name.startswith("epoch_best")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[ddp_v12] Removed old checkpoint: {old_ckpt}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing and self.global_rank == 0:
            print(f"[load_checkpoint] Missing keys: {len(missing)}")
        if unexpected and self.global_rank == 0:
            print(f"[load_checkpoint] Unexpected keys: {len(unexpected)}")
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                if self.global_rank == 0:
                    print(f"[load_checkpoint] Optimizer mismatch: {e}")
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        epoch = ckpt.get("epoch", 0)
        if not isinstance(epoch, int):
            import re
            m = re.search(r'(\d+)', str(epoch))
            epoch = int(m.group(1)) if m else 0
        return epoch
