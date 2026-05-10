"""DDP V11 训练器 — 数据扩展 + 架构优化.

核心改进 (vs V10):
- 源特定重建权重: 动态目标=1.0, DEM=0.3, 静态分类=0.5
- 简化损失: 移除 decorrelation + orthogonality (冗余)
- 强化教师-学生一致性: weight 0.05 → 0.2 (AEF 核心机制)
- 静态目标时间编码弱化: decoder 中 relative_time → 0
- 课程学习: Phase 1 (哈尔滨) → Phase 2 (全省混合)
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
    raw_uniformity_loss,
    variance_regularizer,
    consistency_loss,
    classification_loss,
    gap_aware_temporal_cosine_loss,
    pixel_change_supervision_loss,
    change_consistency_loss,
)
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


# ---------------------------------------------------------------------------
# 源 → 重建权重映射
# ---------------------------------------------------------------------------
# TARGET_SOURCES 顺序: s2, s1, landsat, dem, worldcover, dynamic_world, jrc_water
DEFAULT_SOURCE_RECON_WEIGHTS = [1.0, 1.0, 1.0, 0.3, 0.5, 1.0, 0.5]


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
    """构建 Student 的扰动输入视图."""
    frames = source_frames.clone()
    frame_mask = source_frame_mask.clone()
    input_mask = source_input_mask.clone()
    stats = {
        "source_drop_ratio": 0.0,
        "frame_drop_ratio": 0.0,
        "front_cut_ratio": 0.0,
        "back_cut_ratio": 0.0,
    }

    # 1. 随机丢源
    if source_drop_rate > 0:
        source_keep = (torch.rand(input_mask.shape, device=input_mask.device) > source_drop_rate)
        input_mask = input_mask & source_keep
        stats["source_drop_ratio"] = float((~source_keep).float().mean().item())
        for batch_index in range(input_mask.shape[0]):
            if not input_mask[batch_index].any():
                input_mask[batch_index, 0] = True

    # 2. 随机丢帧
    if drop_rate > 0:
        keep = (torch.rand(frame_mask.shape, device=frame_mask.device) > drop_rate) & frame_mask
        frame_mask = keep
        stats["frame_drop_ratio"] = float((~keep).float().mean().item())

    # 3. 截断前后段
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


def _get_kappa(epoch: int, cfg) -> float:
    """渐进 Kappa 调度."""
    kappa_start = getattr(cfg.training, 'kappa_start', 0.0)
    kappa_end = getattr(cfg.training, 'kappa_end', 0.0)
    warmup = getattr(cfg.training, 'kappa_warmup_epochs', 100)
    if kappa_start <= 0 or kappa_end <= 0:
        return getattr(cfg.model, 'vmf_kappa', 2000.0)
    progress = min(epoch / max(warmup, 1), 1.0)
    return kappa_start + (kappa_end - kappa_start) * progress


def _get_temporal_weight(epoch: int, cfg) -> float:
    """动态 Temporal Weight 调度."""
    t = cfg.training
    warmup = getattr(t, 'temporal_gap_aware_warmup_epochs', 30)
    if epoch < warmup:
        return 0.0
    
    w_start = getattr(t, 'temporal_gap_aware_weight', 0.02)
    w_end = getattr(t, 'temporal_gap_aware_weight_end', w_start)
    ramp_epochs = getattr(t, 'temporal_gap_aware_weight_ramp_epochs', 50)
    
    if ramp_epochs <= 0 or w_end <= w_start:
        return w_start
    
    progress = min((epoch - warmup) / ramp_epochs, 1.0)
    return w_start + (w_end - w_start) * progress


def _extract_window_images(
    source_frames: torch.Tensor,
    source_timestamps_ms: torch.Tensor,
    source_frame_mask: torch.Tensor,
    source_input_mask: torch.Tensor,
    valid_start: torch.Tensor,
    valid_end: torch.Tensor,
) -> torch.Tensor | None:
    """从 source_frames 中提取指定时间窗口的代表图像."""
    if source_frames.dim() != 6:
        return None
    
    B, S, T, C, H, W = source_frames.shape
    device = source_frames.device
    
    window_center = (valid_start.float() + valid_end.float()) / 2.0
    
    images = []
    for b in range(B):
        valid_src = source_input_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_src) == 0:
            images.append(source_frames.new_zeros(3, H, W))
            continue
        
        s_idx = valid_src[0].item()
        
        ts = source_timestamps_ms[b, s_idx].float()
        mask = source_frame_mask[b, s_idx]
        in_window = (ts >= valid_start[b]) & (ts <= valid_end[b]) & mask
        
        if not in_window.any():
            first_valid = mask.nonzero(as_tuple=True)[0]
            if len(first_valid) > 0:
                frame = source_frames[b, s_idx, first_valid[0].item(), :3]
            else:
                frame = source_frames.new_zeros(3, H, W)
            images.append(frame)
            continue
        
        valid_ts = ts[in_window]
        valid_indices = in_window.nonzero(as_tuple=True)[0]
        center = window_center[b]
        closest_idx = (valid_ts - center).abs().argmin()
        frame_idx = valid_indices[closest_idx].item()
        
        frame = source_frames[b, s_idx, frame_idx, :3]
        images.append(frame)
    
    return torch.stack(images)


class DDPv11Trainer:
    """DDP V11 训练器 — 数据扩展 + 架构优化."""

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

        # temporal loss 动态 weight
        self.temporal_weight_current = getattr(cfg.training, 'temporal_gap_aware_weight', 0.02)
        self.recon_history: list[float] = []
        self.pixel_change_history: list[float] = []

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
        accum_steps = getattr(t, "gradient_accumulation_steps", 8)
        loss_accum: dict[str, float] = {}
        n_steps = 0

        # 渐进 Kappa
        kappa = _get_kappa(epoch, self.cfg)
        self.model.module.bottleneck.kappa = kappa
        self.teacher.bottleneck.kappa = kappa

        # Temporal loss warmup & dynamic weight
        temporal_warmup_epochs = getattr(t, 'temporal_gap_aware_warmup_epochs', 30)
        temporal_enabled = epoch >= temporal_warmup_epochs
        self.temporal_weight_current = _get_temporal_weight(epoch, self.cfg)

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=True):
                # Teacher forward: 完整输入
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

                # Student forward: 扰动输入
                student_frames, student_frame_mask, student_input_mask, perturb_stats = _build_student_view(
                    batch["source_frames"],
                    batch["source_timestamps_ms"],
                    batch["source_frame_mask"],
                    batch["source_input_mask"],
                    drop_rate=getattr(t, 'student_frame_drop_rate', 0.4),
                    source_drop_rate=getattr(t, 'student_source_drop_rate', 0.25),
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

                # Reconstruction (只从 student 计算, 带源特定权重)
                recon = self._compute_recon_loss(student_out.reconstructions, batch)

            # Gather pre_norm embeddings across GPUs for uniformity
            pre_norm = student_out.pre_norm_embedding
            if dist.is_initialized() and self.world_size > 1:
                gathered_pre_norm = [torch.zeros_like(pre_norm) for _ in range(self.world_size)]
                dist.all_gather(gathered_pre_norm, pre_norm)
                gathered_pre_norm = torch.cat(gathered_pre_norm, dim=0)
            else:
                gathered_pre_norm = pre_norm

            # 1. Consistency Loss (V11: 强化到 0.2)
            consist_w = getattr(t, 'consistency_weight', 0.0)
            consist = torch.tensor(0.0, device=self.device)
            if consist_w > 0:
                consist = consistency_loss(
                    teacher_out.embedding.detach().float(),
                    student_out.embedding.float()
                )

            # 2. Classification Loss
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
                    if student_out.logits is not None:
                        dummy_cls = dummy_cls + student_out.logits.sum() * 0.0
                    if student_out.aux_logits is not None:
                        dummy_cls = dummy_cls + student_out.aux_logits.sum() * 0.0
                    if student_out.bottleneck_logits is not None:
                        dummy_cls = dummy_cls + student_out.bottleneck_logits.sum() * 0.0

            # 3. Raw Uniformity Loss
            uniform_w = getattr(t, 'uniformity_weight', 0.0)
            uniform = torch.tensor(0.0, device=self.device)
            if uniform_w > 0 and gathered_pre_norm.shape[0] >= 2:
                uniform = raw_uniformity_loss(gathered_pre_norm.float())

            # 4. Variance Regularizer (VICReg)
            var_w = getattr(t, 'variance_weight', 0.0)
            var = torch.tensor(0.0, device=self.device)
            if var_w > 0 and gathered_pre_norm.shape[0] >= 2:
                var = variance_regularizer(gathered_pre_norm.float(), min_std=1.0)

            # V11: 移除 decorrelation 和 orthogonality (已验证冗余)

            # 5. Gap-aware Temporal Loss + Difference Module
            temporal = torch.tensor(0.0, device=self.device)
            change_consist = torch.tensor(0.0, device=self.device)
            change_score = None
            if temporal_enabled and self.temporal_weight_current > 0:
                dual_keys = ['valid_start_w1', 'valid_end_w1', 'valid_start_w2', 'valid_end_w2']
                if all(k in batch for k in dual_keys):
                    try:
                        with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=True):
                            emb_w1, emb_w2, pre_w1, pre_w2, change_score, diff_feat = self.model.module.encode_dual_window_v10(
                                source_frames=student_frames,
                                source_timestamps_ms=batch["source_timestamps_ms"],
                                source_frame_mask=student_frame_mask,
                                source_input_mask=student_input_mask,
                                source_type_ids=batch["source_type_ids"],
                                valid_start_w1=batch["valid_start_w1"].float(),
                                valid_end_w1=batch["valid_end_w1"].float(),
                                valid_start_w2=batch["valid_start_w2"].float(),
                                valid_end_w2=batch["valid_end_w2"].float(),
                            )

                        w1_center = (batch["valid_start_w1"].float() + batch["valid_end_w1"].float()) / 2.0
                        w2_center = (batch["valid_start_w2"].float() + batch["valid_end_w2"].float()) / 2.0
                        time_gap_ms = torch.abs(w1_center - w2_center).to(self.device)
                        max_gap_ms = getattr(t, 'temporal_gap_max_months', 12) * 30 * 24 * 3600 * 1000

                        temporal = gap_aware_temporal_cosine_loss(
                            pre_w1, pre_w2, time_gap_ms,
                            max_gap_ms=max_gap_ms,
                            temperature=0.1,
                        )

                        change_w = getattr(t, 'change_consistency_weight', 0.0)
                        change_warmup = getattr(t, 'change_consistency_warmup_epochs', 40)
                        if change_w > 0 and epoch >= change_warmup and change_score is not None:
                            with torch.no_grad():
                                img_w1 = _extract_window_images(
                                    student_frames,
                                    batch["source_timestamps_ms"],
                                    student_frame_mask,
                                    student_input_mask,
                                    batch["valid_start_w1"].float(),
                                    batch["valid_end_w1"].float(),
                                )
                                img_w2 = _extract_window_images(
                                    student_frames,
                                    batch["source_timestamps_ms"],
                                    student_frame_mask,
                                    student_input_mask,
                                    batch["valid_start_w2"].float(),
                                    batch["valid_end_w2"].float(),
                                )
                            if img_w1 is not None and img_w2 is not None:
                                threshold = getattr(t, 'change_consistency_threshold', 0.1)
                                change_consist = change_consistency_loss(
                                    change_score, img_w1, img_w2, threshold=threshold
                                )
                    except Exception as e:
                        if self.global_rank == 0:
                            print(f"  [Temporal/V10] Error at step {step}: {e}")

            # 6. Pixel Change Supervision
            pixel_change = torch.tensor(0.0, device=self.device)
            pixel_change_w = getattr(t, 'pixel_change_supervision_weight', 0.0)
            pixel_change_warmup = getattr(t, 'pixel_change_supervision_warmup_epochs', 40)
            if pixel_change_w > 0 and epoch >= pixel_change_warmup and temporal_enabled:
                dual_keys = ['valid_start_w1', 'valid_end_w1', 'valid_start_w2', 'valid_end_w2']
                if all(k in batch for k in dual_keys):
                    try:
                        with torch.no_grad():
                            img_w1 = _extract_window_images(
                                student_frames,
                                batch["source_timestamps_ms"],
                                student_frame_mask,
                                student_input_mask,
                                batch["valid_start_w1"].float(),
                                batch["valid_end_w1"].float(),
                            )
                            img_w2 = _extract_window_images(
                                student_frames,
                                batch["source_timestamps_ms"],
                                student_frame_mask,
                                student_input_mask,
                                batch["valid_start_w2"].float(),
                                batch["valid_end_w2"].float(),
                            )
                        
                        if img_w1 is not None and img_w2 is not None:
                            threshold = getattr(t, 'pixel_change_threshold', 0.1)
                            pixel_change = pixel_change_supervision_loss(
                                pre_w1, pre_w2, img_w1, img_w2,
                                threshold=threshold,
                            )
                    except Exception as e:
                        if self.global_rank == 0:
                            print(f"  [PixelChange] Error at step {step}: {e}")

            # Recon warmup
            recon_warmup = min(1.0, (epoch + 1) / max(getattr(t, 'recon_warmup_epochs', 10), 1))
            recon_weight = t.reconstruction_weight * recon_warmup

            change_w = getattr(t, 'change_consistency_weight', 0.0)
            total = (
                recon_weight * recon
                + consist_w * consist
                + cls_w * cls
                + uniform_w * uniform
                + var_w * var
                + self.temporal_weight_current * temporal
                + pixel_change_w * pixel_change
                + change_w * change_consist
                + dummy_cls
            )
            
            # NaN/Inf 检测
            if torch.isnan(total) or torch.isinf(total):
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"  [WARNING] NaN/Inf total at step {step}, skipping. "
                          f"recon={recon.item():.3f} consist={consist.item():.3f} "
                          f"cls={cls.item():.3f} uniform={uniform.item():.3f} "
                          f"var={var.item():.3f} temporal={temporal.item():.3f}")
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
                "cls": cls.item(),
                "uniform": uniform.item(),
                "var": var.item(),
                "temporal": temporal.item(),
                "pixel_change": pixel_change.item(),
                "change_consist": change_consist.item(),
                "lr": lr,
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

            if self.global_rank == 0 and step % 20 == 0:
                temporal_status = f"temporal={temporal.item():.4f}" if temporal_enabled else "temporal=warmup"
                pixel_status = f" pc={pixel_change.item():.3f}" if pixel_change_w > 0 and epoch >= pixel_change_warmup else ""
                change_status = f" chg={change_consist.item():.3f}" if change_w > 0 and epoch >= getattr(t, 'change_consistency_warmup_epochs', 40) else ""
                print(f"  [Step {step}] recon={recon.item():.3f} {temporal_status}{pixel_status}{change_status} "
                      f"perturb=[fd={perturb_stats['frame_drop_ratio']:.2f} "
                      f"sd={perturb_stats['source_drop_ratio']:.2f} "
                      f"fc={perturb_stats['front_cut_ratio']:.2f} "
                      f"bc={perturb_stats['back_cut_ratio']:.2f}] "
                      f"temp_w={self.temporal_weight_current:.3f} kappa={kappa:.1f}")

        # 平均并跨卡同步
        loss_accum = {k: v / n_steps for k, v in loss_accum.items()}
        loss_accum = self._reduce_loss_dict(loss_accum)
        loss_accum["lr"] = lr

        # 熔断机制
        self.recon_history.append(loss_accum["recon"])
        if len(self.recon_history) >= 6:
            recent = self.recon_history[-5:]
            if all(recent[i] <= recent[i+1] for i in range(4)):
                old_w = self.temporal_weight_current
                self.temporal_weight_current *= 0.5
                if self.global_rank == 0:
                    print(f"  [FUSE] Recon rising for 5 epochs: {recent[0]:.4f} → {recent[-1]:.4f}. "
                          f"Reducing temporal weight: {old_w:.4f} → {self.temporal_weight_current:.4f}")
                self.recon_history.clear()

        return loss_accum

    def _compute_recon_loss(self, predictions, batch):
        """计算重建损失，应用源特定权重."""
        from src.training.loops import compute_recon_loss
        base_loss = compute_recon_loss(
            predictions, batch["target_images"], batch["target_mask"],
            batch.get("target_loss_type"), self.cfg.data.num_classes,
        )
        
        # 应用源特定权重
        target_source_idx = batch.get("target_source_idx")
        if target_source_idx is not None:
            # target_source_idx: [B, T_tgt]
            B, T = target_source_idx.shape
            # 为每个样本-目标对取权重
            weights = self.source_recon_weights[target_source_idx]  # [B, T]
            # 平均权重作为该 batch 的缩放因子
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
        print(f"[ddp_v11] Saved checkpoint to {path}")
        
        ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_*.pt") if not p.name.startswith("epoch_best")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[ddp_v11] Removed old checkpoint: {old_ckpt}")

        best_ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_best_*.pt")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in best_ckpts[:-2]:
            old_ckpt.unlink()
            print(f"[ddp_v11] Removed old best checkpoint: {old_ckpt}")

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
                if self.global_rank == 0:
                    print("[load_checkpoint] Optimizer state loaded.")
            except ValueError as e:
                if self.global_rank == 0:
                    print(f"[load_checkpoint] Optimizer mismatch: {e}")
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if self.global_rank == 0:
                print("[load_checkpoint] GradScaler state loaded.")
        epoch = ckpt.get("epoch", 0)
        if not isinstance(epoch, int):
            import re
            m = re.search(r'(\d+)', str(epoch))
            epoch = int(m.group(1)) if m else 0
        return epoch

    def soft_restart(self, path: str) -> None:
        """软重启: 只加载 encoder 权重, 重新初始化 bottleneck/decoder/head."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        old_state = ckpt["model_state_dict"]
        
        self.model.module.load_state_dict(old_state, strict=False)
        
        def _reset_module(m):
            for layer in m.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        _reset_module(self.model.module.bottleneck)
        if self.global_rank == 0:
            print("[soft_restart] Bottleneck reinitialized.")
        
        if hasattr(self.model.module, 'per_source_decoders'):
            for dec in self.model.module.per_source_decoders:
                _reset_module(dec)
        if hasattr(self.model.module, 'continuous_decoder'):
            _reset_module(self.model.module.continuous_decoder)
        if hasattr(self.model.module, 'categorical_decoder'):
            _reset_module(self.model.module.categorical_decoder)
        if self.global_rank == 0:
            print("[soft_restart] Decoders reinitialized.")
        
        if hasattr(self.model.module, 'classification_head'):
            _reset_module(self.model.module.classification_head)
        if hasattr(self.model.module, 'aux_cls_head'):
            _reset_module(self.model.module.aux_cls_head)
        if hasattr(self.model.module, 'bottleneck_cls_head'):
            _reset_module(self.model.module.bottleneck_cls_head)
        if self.global_rank == 0:
            print("[soft_restart] Classification heads reinitialized.")
        
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.optimizer = build_optimizer(self.model.module, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)
        if self.global_rank == 0:
            print("[soft_restart] Optimizer and scheduler reinitialized.")
        
        if self.global_rank == 0:
            print(f"[soft_restart] Soft restart from {path} complete. Encoder preserved, others reset.")
