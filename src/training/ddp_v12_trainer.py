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
    raw_uniformity_loss,
    consistency_loss,
    consistency_loss_spatial,
    variance_regularizer,
    covariance_loss,
    decorrelation_loss,
    classification_loss,
    bottleneck_orthogonality_loss,
)
from src.training.memory_bank import EmbeddingMemoryBank
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


# ---------------------------------------------------------------------------
# 源 → 重建权重映射
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_RECON_WEIGHTS = [1.0, 1.0, 1.0, 0.05]  # S2, S1, Landsat, DEM


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
        # V13: 禁用 scaler，fp32 训练不需要
        self.scaler = None

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

        # Memory Bank — 扩大 pre-norm 有效 batch
        emb_dim = getattr(cfg.model, 'embedding_dim', 128)
        self.memory_bank = EmbeddingMemoryBank(K=512, dim=emb_dim, device=self.device)
        
        # 日志文件句柄（用于 step 日志同时写入文件）
        self.log_file = None
        if self.global_rank == 0:
            self.log_file = open(self.output_dir / "train.log", "a", buffering=1, encoding="utf-8")

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
        consist_w = getattr(t, 'consistency_weight', 0.0)
        uniform_w = getattr(t, 'batch_uniformity_weight', 0.0)
        recon_warmup = min(1.0, (epoch + 1) / max(getattr(t, 'recon_warmup_epochs', 10), 1))

        import itertools
        max_steps = getattr(t, 'max_steps_per_epoch', None)
        iterator = itertools.islice(dataloader, max_steps) if max_steps else dataloader
        for step, batch in enumerate(iterator):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            max_steps = getattr(t, 'max_steps_per_epoch', None)
            effective_steps = min(len(dataloader), max_steps) if max_steps else len(dataloader)
            lr = get_cosine_lr(epoch, step, effective_steps, self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=False):
                # Teacher forward
                has_dual = all(k in batch for k in ['valid_start_w1', 'valid_end_w1', 'valid_start_w2', 'valid_end_w2'])
                # V13: empty cache before teacher forward to avoid OOM
                if hasattr(torch.npu, 'empty_cache'):
                    torch.npu.empty_cache()
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

                # V13: Consistency (teacher vs student spatial map)
                if consist_w > 0:
                    consist = consistency_loss_spatial(
                        teacher_out.embedding_map.detach(),
                        student_out.embedding_map,
                    )
                else:
                    consist = torch.tensor(0.0, device=self.device)

                # Classification Loss (语义监督)
                cls_w = getattr(t, 'classification_weight', 0.0)
                cls = torch.tensor(0.0, device=self.device)
                dummy_cls = torch.tensor(0.0, device=self.device)
                if cls_w > 0 and "label" in batch:
                    labels = batch["label"]
                    if labels.unique().numel() > 1 or labels[0].item() != 0:
                        cls = classification_loss(student_out.logits.float(), labels)
                        if student_out.aux_logits is not None:
                            cls = cls + 0.5 * classification_loss(student_out.aux_logits.float(), labels)
                        if student_out.bottleneck_logits is not None:
                            cls = cls + 0.3 * classification_loss(student_out.bottleneck_logits.float(), labels)
                    else:
                        # Dummy loss 确保 DDP 所有参数参与梯度
                        if student_out.logits is not None:
                            dummy_cls = dummy_cls + student_out.logits.sum() * 0.0
                        if student_out.aux_logits is not None:
                            dummy_cls = dummy_cls + student_out.aux_logits.sum() * 0.0
                        if student_out.bottleneck_logits is not None:
                            dummy_cls = dummy_cls + student_out.bottleneck_logits.sum() * 0.0

            # === VICReg Variance + Covariance (Pre-norm 空间) + Memory Bank ===
            pre_norm = student_out.pre_norm_embedding  # [B, D] — 真正的 pre-norm
            pre_norm_map = student_out.pre_norm_map      # [B, D, H, W] — spatial pre-norm
            
            # V13: skip NaN/Inf check (fixed in model.encode_frames fallback)
            
            if dist.is_initialized() and self.world_size > 1:
                gathered_pre = [torch.zeros_like(pre_norm) for _ in range(self.world_size)]
                dist.all_gather(gathered_pre, pre_norm)
                gathered_pre = torch.cat(gathered_pre, dim=0)
            else:
                gathered_pre = pre_norm

            # Memory Bank: 扩大 pre-norm 有效 batch
            self.memory_bank.enqueue(gathered_pre.detach())
            bank_emb = self.memory_bank.get_all()
            if bank_emb.shape[0] > 0:
                all_pre = torch.cat([gathered_pre, bank_emb], dim=0)
            else:
                all_pre = gathered_pre

            # Pre-norm VICReg: variance + covariance
            var_w = getattr(t, 'variance_weight', 0.3)
            cov_w = getattr(t, 'covariance_weight', 0.1)
            vicreg_min_std = getattr(t, 'vicreg_min_std', 1.0)
            
            # V13-explore: Spatial VICReg — 在 [B*H*W, D] 上计算，绕过 GMP
            # ★ FIX: DDP 下必须 all_gather 所有卡的 pre_norm_map，否则 effective batch 不一致
            use_spatial_vicreg = getattr(t, 'use_spatial_vicreg', False)
            gathered_spatial_map = pre_norm_map
            if use_spatial_vicreg and pre_norm_map is not None:
                if dist.is_initialized() and self.world_size > 1:
                    gathered_maps = [torch.zeros_like(pre_norm_map) for _ in range(self.world_size)]
                    dist.all_gather(gathered_maps, pre_norm_map)
                    gathered_spatial_map = torch.cat(gathered_maps, dim=0)
                # [B_total, D, H, W] → [B_total*H*W, D]
                spatial_flat = gathered_spatial_map.permute(0, 2, 3, 1).reshape(-1, gathered_spatial_map.shape[1])
                var = variance_regularizer(spatial_flat.float(), min_std=vicreg_min_std) if var_w > 0 else torch.tensor(0.0, device=self.device)
                cov = covariance_loss(spatial_flat.float()) if cov_w > 0 else torch.tensor(0.0, device=self.device)
            else:
                var = variance_regularizer(all_pre.float(), min_std=vicreg_min_std) if var_w > 0 else torch.tensor(0.0, device=self.device)
                cov = covariance_loss(all_pre.float()) if cov_w > 0 else torch.tensor(0.0, device=self.device)
            
            # ★ 新增：样本间 VICReg variance（在 gathered_pre 上计算）
            inter_var_w = getattr(t, 'inter_variance_weight', 0.0)
            inter_min_std = getattr(t, 'inter_variance_min_std', 0.3)
            inter_var = torch.tensor(0.0, device=self.device)
            inter_cov = torch.tensor(0.0, device=self.device)
            if inter_var_w > 0 and gathered_pre is not None and gathered_pre.shape[0] >= 2:
                inter_var = variance_regularizer(gathered_pre.float(), min_std=inter_min_std)
                inter_cov = covariance_loss(gathered_pre.float())
                var = var + inter_var_w * inter_var
                cov = cov + inter_var_w * inter_cov

            # === Uniformity Loss（实验变体支持）===
            use_spatial_unif = getattr(t, 'use_spatial_uniformity', False)
            use_pre_norm_unif = getattr(t, 'use_pre_norm_uniform', False)
            use_spatial_raw_unif = getattr(t, 'use_spatial_raw_uniformity', False)
            l2_uniform_w = getattr(t, 'batch_uniformity_weight', 0.05)
            pre_norm_uniform_w = getattr(t, 'pre_norm_uniform_weight', 0.0)

            if use_spatial_raw_unif and pre_norm_uniform_w > 0:
                # V13-explore: Spatial raw uniformity on pre_norm_map [B, D, H, W] — 完全绕过 GMP + L2
                if dist.is_initialized() and self.world_size > 1:
                    gathered_map_unif = [torch.zeros_like(pre_norm_map) for _ in range(self.world_size)]
                    dist.all_gather(gathered_map_unif, pre_norm_map)
                    gathered_map_unif = torch.cat(gathered_map_unif, dim=0)
                else:
                    gathered_map_unif = pre_norm_map
                # [B, D, H, W] → [B*H*W, D]
                spatial_flat_unif = gathered_map_unif.permute(0, 2, 3, 1).reshape(-1, gathered_map_unif.shape[1])
                l2_uniform = raw_uniformity_loss(spatial_flat_unif.float())
                l2_uniform_w = pre_norm_uniform_w
            elif use_pre_norm_unif and pre_norm_uniform_w > 0:
                # Pre-norm raw uniformity（欧氏空间，无 L2 Jacobian 屏障）
                if dist.is_initialized() and self.world_size > 1:
                    gathered_pre_unif = [torch.zeros_like(pre_norm) for _ in range(self.world_size)]
                    dist.all_gather(gathered_pre_unif, pre_norm)
                    gathered_pre_unif = torch.cat(gathered_pre_unif, dim=0)
                else:
                    gathered_pre_unif = pre_norm
                l2_uniform = raw_uniformity_loss(gathered_pre_unif.float())
                l2_uniform_w = pre_norm_uniform_w
            elif use_spatial_unif and l2_uniform_w > 0:
                # Spatial uniformity on embedding_map [B, D, H, W]
                emb_map = student_out.embedding_map  # [B, D, H, W]
                if dist.is_initialized() and self.world_size > 1:
                    gathered_map = [torch.zeros_like(emb_map) for _ in range(self.world_size)]
                    dist.all_gather(gathered_map, emb_map)
                    gathered_map = torch.cat(gathered_map, dim=0)
                else:
                    gathered_map = emb_map
                l2_uniform = batch_uniformity_loss_l2(gathered_map.float())
            else:
                # Standard L2 uniformity on embedding vector
                embedding = student_out.embedding  # [B, D]
                if dist.is_initialized() and self.world_size > 1:
                    gathered_l2 = [torch.zeros_like(embedding) for _ in range(self.world_size)]
                    dist.all_gather(gathered_l2, embedding)
                    gathered_l2 = torch.cat(gathered_l2, dim=0)
                else:
                    gathered_l2 = embedding
                l2_uniform = batch_uniformity_loss_l2(gathered_l2.float()) if l2_uniform_w > 0 else torch.tensor(0.0, device=self.device)

            # Per-dim 诊断（日志用）
            # ★ FIX: 使用 gathered_spatial_map 计算 active，与 spatial_vicreg 一致
            with torch.no_grad():
                if use_spatial_vicreg and gathered_spatial_map is not None:
                    spatial_flat_diag = gathered_spatial_map.permute(0, 2, 3, 1).reshape(-1, gathered_spatial_map.shape[1])
                    std_per_dim = torch.sqrt(spatial_flat_diag.var(dim=0, unbiased=False) + 1e-6)
                else:
                    std_per_dim = torch.sqrt(all_pre.var(dim=0, unbiased=False) + 1e-6)
                std_min = std_per_dim.min().item()
                std_mean = std_per_dim.mean().item()
                std_max = std_per_dim.max().item()
                active_dims = (std_per_dim > 0.05).sum().item()
                cov_offdiag = cov.item() if cov_w > 0 else 0.0
                
                # ★ 新增：样本间多样性诊断
                inter_std = torch.sqrt(gathered_pre.var(dim=0, unbiased=False) + 1e-6)
                inter_active = (inter_std > 0.05).sum().item()
                inter_std_mean = inter_std.mean().item()

            # Decorrelation Loss (Barlow Twins)
            decorr_w = getattr(t, 'decorrelation_weight', 0.0)
            decorr = torch.tensor(0.0, device=self.device)
            if decorr_w > 0 and gathered_pre.shape[0] >= 2:
                decorr = decorrelation_loss(gathered_pre.float())

            # V13: Bottleneck Orthogonality Loss
            orth_w = getattr(t, 'orthogonality_weight', 0.0)
            orth = torch.tensor(0.0, device=self.device)
            if orth_w > 0:
                orth = bottleneck_orthogonality_loss(self.model.module.bottleneck.to_embedding.weight)

            total = (
                recon_w * recon_warmup * recon
                + consist_w * consist
                + cls_w * cls
                + var_w * var
                + cov_w * cov
                + l2_uniform_w * l2_uniform
                + decorr_w * decorr
                + orth_w * orth
                + dummy_cls
                + inter_var_w * inter_var
            )

            if torch.isnan(total) or torch.isinf(total):
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"  [WARNING] NaN/Inf total at step {step}, skipping.")
                    print(f"    recon={recon.item():.4f} consist={consist.item():.4f} "
                          f"cls={cls.item():.4f} var={var.item():.4f} cov={cov.item():.4f} "
                          f"decorr={decorr.item():.4f} l2unif={l2_uniform.item():.4f}")
                self.optimizer.zero_grad()
                continue

            total = total / accum_steps
            if self.scaler is not None:
                self.scaler.scale(total).backward()
            else:
                total.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                if self.scaler is not None:
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
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                self.update_teacher()
                
                # V13: weight health check (silent)
                # for name, p in self.model.named_parameters():
                #     if p is not None and (torch.isnan(p).any() or torch.isinf(p).any()):
                #         print(f"  [DEBUG] NaN/Inf weight after step: {name}")

            for k, v in {
                "total": total.item() * accum_steps,
                "recon": recon.item(),
                "consist": consist.item(),
                "cls": cls.item(),
                "var": var.item(),
                "cov": cov.item(),
                "l2unif": l2_uniform.item(),
                "decorr": decorr.item(),
                "orth": orth.item(),
                "lr": lr,
                "inter_var": inter_var.item() if inter_var_w > 0 else 0.0,
                "active_dims": float(active_dims),
                "std_mean": float(std_mean),
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

            if self.global_rank == 0:
                bank_size = self.memory_bank.size
                step_msg = (f"  [Step {step}] recon={recon.item():.4f} "
                      f"consist={consist.item():.4f} cls={cls.item():.4f} "
                      f"var={var.item():.4f} cov={cov.item():.4f} "
                      f"decorr={decorr.item():.4f} orth={orth.item():.4f} "
                      f"l2unif={l2_uniform.item():.4f} "
                      f"inter_var={inter_var.item():.4f} "
                      f"bank={bank_size}/{self.memory_bank.K} "
                      f"spatial=[{active_dims}/{pre_norm_map.shape[1] if pre_norm_map is not None else pre_norm.shape[1]}:{std_mean:.4f}] "
                      f"inter=[{inter_active}/{pre_norm.shape[1]}:{inter_std_mean:.4f}] "
                      f"perturb=[fd={perturb_stats['frame_drop_ratio']:.2f} "
                      f"sd={perturb_stats['source_drop_ratio']:.2f}] "
                      f"lr={lr:.6f}")
                print(step_msg)
                if self.log_file:
                    self.log_file.write(step_msg + "\n")
                    self.log_file.flush()

        loss_accum = {k: v / n_steps for k, v in loss_accum.items()}
        loss_accum = self._reduce_loss_dict(loss_accum)
        loss_accum["lr"] = lr
        # 添加 epoch 级别诊断指标
        # ★ FIX: 不再用 memory bank 计算 active（bank 可能包含历史坍缩嵌入）
        # 保留 step 累加的平均 active_dims 和 std_mean（已在 loss_accum 中）
        loss_accum["bank"] = float(self.memory_bank.size)
        return loss_accum

    def _compute_recon_loss(self, predictions, batch):
        """计算重建损失，应用源特定权重."""
        from src.training.loops import compute_recon_loss
        base_loss = compute_recon_loss(
            predictions, batch["target_images"], batch["target_mask"],
            batch.get("target_loss_type"), self.cfg.data.num_classes,
            recon_mask=batch.get("recon_mask"),
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
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else {},
            "losses": losses,
        }, path)
        print(f"[ddp_v12] Saved checkpoint to {path}")

        # 清理普通 checkpoint，只保留最近 3 个
        ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_*.pt") if not p.name.startswith("epoch_best")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[ddp_v12] Removed old checkpoint: {old_ckpt}")

        # 清理 best checkpoint，只保留最近 3 个
        best_ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_best_*.pt")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in best_ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[ddp_v12] Removed old best checkpoint: {old_ckpt}")

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
            if self.scaler is not None and "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        epoch = ckpt.get("epoch", 0)
        if not isinstance(epoch, int):
            import re
            m = re.search(r'(\d+)', str(epoch))
            epoch = int(m.group(1)) if m else 0
        return epoch
