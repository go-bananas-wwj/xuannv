"""DDP V5 Mixed Scale 训练器 — 混合尺度时间窗口 + Temporal Magnitude Loss.

核心机制:
- Teacher: 完整输入 → embedding
- Student: 扰动输入 (丢帧/丢源/截断) → embedding
- Consistency Loss: 强制 teacher/student embedding 一致
- Dual Window: encode_dual_window(w1, w2) → pre_w1, pre_w2
- Temporal Magnitude Loss: 约束 embedding distance ≤ time_gap_norm + margin
- Scale-aware: 长间隔 consistency 权重更高
- Classification Loss: WorldCover 众数类别监督
- Raw Uniformity + Variance + Decorrelation: 反坍缩三件套
- 渐进 VMF Kappa: 50→500
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
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
    decorrelation_loss,
    bottleneck_orthogonality_loss,
    consistency_loss,
    classification_loss,
    temporal_magnitude_loss,
)
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


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
        # 至少保留一个源
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
                
                # 按真实时间戳排序后截断
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


class DDPv5MixedScaleTrainer:
    """DDP V5 Mixed Scale 训练器 — 双卡 GPU 6/7."""

    def __init__(self, cfg: Config, local_rank: int = 0) -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Student 模型
        self.model = AEFModel(cfg).to(self.device)
        self.model = DistributedDataParallel(
            self.model, device_ids=[local_rank], find_unused_parameters=True
        )

        # EMA Teacher（独立，不包装 DDP）
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = getattr(cfg.training, "teacher_momentum", 0.996)

        # 优化器
        self.optimizer = build_optimizer(self.model.module, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # GradScaler — 更高初始 scale 防止 fp16 梯度溢出
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2**18)

        # 输出目录
        self.output_dir = Path(cfg.experiment.output_dir)
        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

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

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
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

                # Reconstruction (只从 student 计算)
                recon = self._compute_recon_loss(student_out.reconstructions, batch)

                # Dual Window forward: 用 w1/w2 分别编码 (用于 temporal magnitude loss)
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

            # Gather pre_norm embeddings across GPUs for uniformity
            pre_norm = student_out.pre_norm_embedding  # [B, D]
            if dist.is_initialized() and self.world_size > 1:
                gathered_pre_norm = [torch.zeros_like(pre_norm) for _ in range(self.world_size)]
                dist.all_gather(gathered_pre_norm, pre_norm)
                gathered_pre_norm = torch.cat(gathered_pre_norm, dim=0)
            else:
                gathered_pre_norm = pre_norm

            # 计算时间间隔 (毫秒) — 两个窗口中心点的差距
            center_w1 = (batch["valid_start_w1"] + batch["valid_end_w1"]) * 0.5
            center_w2 = (batch["valid_start_w2"] + batch["valid_end_w2"]) * 0.5
            time_gap_ms = torch.abs(center_w2 - center_w1).float()  # [B]

            # 1. Consistency Loss (论文核心) — fp32 for numerical stability
            consist_w = getattr(t, 'consistency_weight', 0.0)
            consist = torch.tensor(0.0, device=self.device)
            if consist_w > 0:
                # Scale-aware: 短间隔提高 consistency weight (细粒度变化需要更强约束)
                avg_gap_ms = time_gap_ms.mean().item()
                six_month_ms = 6 * 30 * 24 * 3600 * 1000
                if avg_gap_ms < six_month_ms:
                    scale_factor = 1.0 + 0.6 * (1.0 - avg_gap_ms / six_month_ms)
                    consist_w = consist_w * scale_factor
                consist = consistency_loss(teacher_out.embedding.detach().float(), student_out.embedding.float())

            # 2. Classification Loss (语义监督) — fp32 for softmax stability
            cls_w = getattr(t, 'classification_weight', 0.0)
            cls = torch.tensor(0.0, device=self.device)
            dummy_cls = torch.tensor(0.0, device=self.device)
            if cls_w > 0 and "label" in batch:
                labels = batch["label"]
                # 跳过全 0 标签 (未初始化)
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

            # 3. Raw Uniformity Loss — fp32 for numerical stability
            uniform_w = getattr(t, 'uniformity_weight', 0.0)
            uniform = torch.tensor(0.0, device=self.device)
            if uniform_w > 0 and gathered_pre_norm.shape[0] >= 2:
                uniform = raw_uniformity_loss(gathered_pre_norm.float())

            # 4. Variance Regularizer (VICReg)
            var_w = getattr(t, 'variance_weight', 0.0)
            var = torch.tensor(0.0, device=self.device)
            if var_w > 0 and gathered_pre_norm.shape[0] >= 2:
                var = variance_regularizer(gathered_pre_norm.float(), min_std=1.0)

            # 5. Decorrelation Loss (Barlow Twins) — fp32 for numerical stability
            decorr_w = getattr(t, 'decorrelation_weight', 0.0)
            decorr = torch.tensor(0.0, device=self.device)
            if decorr_w > 0 and gathered_pre_norm.shape[0] >= 2:
                decorr = decorrelation_loss(gathered_pre_norm.float())

            # 6. Bottleneck Orthogonality
            orth_w = getattr(t, 'orthogonality_weight', 0.0)
            orth = torch.tensor(0.0, device=self.device)
            if orth_w > 0:
                bnw = self.model.module.bottleneck.to_embedding.weight
                orth = bottleneck_orthogonality_loss(bnw)

            # 7. Temporal Magnitude Loss — 约束 embedding distance ≤ time_gap_norm + margin
            temporal_w = getattr(t, 'temporal_magnitude_weight', 0.0)
            temporal = torch.tensor(0.0, device=self.device)
            if temporal_w > 0:
                temporal = temporal_magnitude_loss(
                    pre_w1.float(), pre_w2.float(),
                    time_gap_ms=time_gap_ms,
                    max_gap_ms=getattr(t, 'temporal_max_gap_ms', six_month_ms),
                    margin=getattr(t, 'temporal_margin', 0.1),
                )

            # Recon warmup
            recon_warmup = min(1.0, (epoch + 1) / max(getattr(t, 'recon_warmup_epochs', 10), 1))
            recon_weight = t.reconstruction_weight * recon_warmup

            total = (
                recon_weight * recon
                + consist_w * consist
                + cls_w * cls
                + uniform_w * uniform
                + var_w * var
                + decorr_w * decorr
                + orth_w * orth
                + temporal_w * temporal
                + dummy_cls
            )
            
            # NaN/Inf 检测: 跳过有问题的 step
            if torch.isnan(total) or torch.isinf(total):
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"  [WARNING] NaN/Inf total at step {step}, skipping. "
                          f"recon={recon.item():.3f} consist={consist.item():.3f} "
                          f"cls={cls.item():.3f} uniform={uniform.item():.3f} "
                          f"var={var.item():.3f} decorr={decorr.item():.3f} "
                          f"temporal={temporal.item():.3f}")
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
                # 检查梯度是否 NaN，如果是则跳过更新
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
                "decorr": decorr.item(),
                "orth": orth.item(),
                "temporal": temporal.item(),
                "lr": lr,
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

            # 打印扰动统计 + gap 信息
            avg_gap_months = time_gap_ms.mean().item() / (30 * 24 * 3600 * 1000)
            if self.global_rank == 0 and step % 20 == 0:
                print(f"  [Perturb] frame_drop={perturb_stats['frame_drop_ratio']:.2f} "
                      f"source_drop={perturb_stats['source_drop_ratio']:.2f} "
                      f"front={perturb_stats['front_cut_ratio']:.2f} "
                      f"back={perturb_stats['back_cut_ratio']:.2f} "
                      f"kappa={kappa:.1f} "
                      f"gap={avg_gap_months:.1f}mo")

        # 平均并跨卡同步
        loss_accum = {k: v / n_steps for k, v in loss_accum.items()}
        loss_accum = self._reduce_loss_dict(loss_accum)
        loss_accum["lr"] = lr
        return loss_accum

    def _compute_recon_loss(self, predictions, batch):
        from src.training.loops import compute_recon_loss
        return compute_recon_loss(
            predictions, batch["target_images"], batch["target_mask"],
            batch.get("target_loss_type"), self.cfg.data.num_classes,
        )

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
        print(f"[ddp_v5_mixed_scale] Saved checkpoint to {path}")
        
        # 自动清理：只保留最新的 3 个数字 epoch checkpoint
        ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_*.pt") if not p.name.startswith("epoch_best")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[ddp_v5_mixed_scale] Removed old checkpoint: {old_ckpt}")

        best_ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_best_*.pt")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in best_ckpts[:-2]:
            old_ckpt.unlink()
            print(f"[ddp_v5_mixed_scale] Removed old best checkpoint: {old_ckpt}")

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
        
        # 1. 加载全部权重 (先恢复结构)
        self.model.module.load_state_dict(old_state, strict=False)
        
        # 2. 重新初始化 bottleneck
        def _reset_module(m):
            for layer in m.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        _reset_module(self.model.module.bottleneck)
        if self.global_rank == 0:
            print("[soft_restart] Bottleneck reinitialized.")
        
        # 3. 重新初始化 decoders
        if hasattr(self.model.module, 'per_source_decoders'):
            for dec in self.model.module.per_source_decoders:
                _reset_module(dec)
        if hasattr(self.model.module, 'continuous_decoder'):
            _reset_module(self.model.module.continuous_decoder)
        if hasattr(self.model.module, 'categorical_decoder'):
            _reset_module(self.model.module.categorical_decoder)
        if self.global_rank == 0:
            print("[soft_restart] Decoders reinitialized.")
        
        # 4. 重新初始化 classification heads
        if hasattr(self.model.module, 'classification_head'):
            _reset_module(self.model.module.classification_head)
        if hasattr(self.model.module, 'aux_cls_head'):
            _reset_module(self.model.module.aux_cls_head)
        if hasattr(self.model.module, 'bottleneck_cls_head'):
            _reset_module(self.model.module.bottleneck_cls_head)
        if self.global_rank == 0:
            print("[soft_restart] Classification heads reinitialized.")
        
        # 5. 同步 teacher
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # 6. 重新初始化优化器 (丢弃旧的 momentum/state)
        self.optimizer = build_optimizer(self.model.module, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)
        if self.global_rank == 0:
            print("[soft_restart] Optimizer and scheduler reinitialized.")
        
        if self.global_rank == 0:
            print(f"[soft_restart] Soft restart from {path} complete. Encoder preserved, others reset.")
