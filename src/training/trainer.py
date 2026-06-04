"""DDP 训练器 — Teacher-Student + 反坍缩损失体系.

核心设计:
- Teacher-Student EMA 一致性（对齐 AEF 原文）
- 损失: Reconstruction + Uniformity + Consistency + VICReg + Temporal Contrastive
- skip_l2_training=True 时在 pre-norm 欧氏空间计算 uniformity，避免梯度屏障
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
    hyperspherical_uniformity_loss,
    pairwise_cosine_diversity_loss,
    aef_batch_uniformity_loss,
    consistency_loss,
    consistency_loss_spatial,
    variance_regularizer,
    covariance_loss,
    decorrelation_loss,
    classification_loss,
    bottleneck_orthogonality_loss,
    latent_mim_loss,
    inter_patch_infonce_loss,
    erank_maximization_loss,
    coding_rate_loss,
    gap_aware_temporal_cosine_loss,
    temporal_contrastive_loss,
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
    """构建 Student 的扰动输入视图 — 对齐 AEF 原文 S2.2.5.

    ★ v41 FIX: 完全向量化改写，消除所有 GPU→CPU 同步（.item() 调用）。
       原实现每 step 触发数百次同步，训练速度下降 10-100 倍。
    """
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
        drop = torch.rand(B, S, device=device) < source_drop_rate
        drop[:, 0] = False  # S2 永不 drop
        input_mask = input_mask & ~drop
        stats["source_drop_ratio"] = float((~input_mask).float().mean().item())
        all_false = ~input_mask.any(dim=1)
        input_mask[all_false, 0] = True

    # Stage 2: 三种策略选一种
    strat = torch.randint(0, 3, (1,)).item()

    if strat == 0:
        # 向量化：每源独立 drop 概率
        frac = torch.where(torch.arange(S, device=device) == 0, 0.5, 0.3)
        drop_mask = torch.rand(B, S, T, device=device) < frac[None, :, None]
        drop_mask = drop_mask & input_mask[:, :, None] & source_frame_mask
        frame_mask = frame_mask & ~drop_mask
        stats["frame_drop_ratio"] = float((~frame_mask & source_frame_mask).float().mean().item())

    elif strat in (1, 2):
        # 向量化：按时间戳排序，保留前半或后半有效帧
        ts_masked = source_timestamps_ms.clone().float()
        ts_masked[~source_frame_mask] = float('inf')
        sorted_idx = ts_masked.argsort(dim=2).float()  # (B, S, T) — float32 for NPU
        valid_count = source_frame_mask.sum(dim=2).long()  # (B, S)
        keep_count = (valid_count / 2).long().clamp(min=1)

        rank = torch.argsort(sorted_idx, dim=2)  # 每个原始索引在排序后序列中的位置
        if strat == 1:
            keep_mask = rank < keep_count[:, :, None]
        else:
            keep_mask = rank >= (valid_count - keep_count)[:, :, None]

        keep_mask = keep_mask & source_frame_mask & input_mask[:, :, None]
        frame_mask = frame_mask & keep_mask

        if strat == 1:
            stats["back_cut_ratio"] = 0.25
        else:
            stats["front_cut_ratio"] = 0.25

    # Stage 3: 截断前后段
    if front_drop_prob > 0 or back_drop_prob > 0:
        ts_masked = source_timestamps_ms.clone().float()
        ts_masked[~frame_mask] = float('inf')
        sorted_idx = ts_masked.argsort(dim=2).float()  # float32 for NPU
        valid_count = frame_mask.sum(dim=2).long()

        do_front = (torch.rand(B, S, device=device) < front_drop_prob) if front_drop_prob > 0 else torch.zeros(B, S, dtype=torch.bool, device=device)
        do_back = (torch.rand(B, S, device=device) < back_drop_prob) if back_drop_prob > 0 else torch.zeros(B, S, dtype=torch.bool, device=device)

        front_cut = torch.where(do_front & (valid_count > 1), (valid_count / 4).long().clamp(min=1), 0)
        back_cut = torch.where(do_back & (valid_count > 1), (valid_count / 4).long().clamp(min=1), 0)

        rank = torch.argsort(sorted_idx, dim=2)
        keep_mask = (rank >= front_cut[:, :, None]) & (rank < (valid_count - back_cut)[:, :, None])
        keep_mask = keep_mask & frame_mask
        frame_mask = keep_mask

        stats["front_cut_ratio"] = float(do_front.float().mean().item())
        stats["back_cut_ratio"] = float(do_back.float().mean().item())

    return frames, frame_mask, input_mask, stats


class DDPv13Trainer:
    """DDP 训练器 — Teacher-Student + 反坍缩损失体系."""

    def __init__(self, cfg: Config, local_rank: int = 0, log_ts: str | None = None) -> None:
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

        # Memory Bank — 扩大 pre-norm 有效 batch（大小从 config 读取）
        emb_dim = getattr(cfg.model, 'embedding_dim', 128)
        bank_size = getattr(cfg.training, 'memory_bank_size', 512)
        self.memory_bank = EmbeddingMemoryBank(K=bank_size, dim=emb_dim, device=self.device)
        
        # 日志文件句柄（用于 step 日志同时写入文件），文件名含时间戳避免多次训练混写
        # log_ts 由 train.py 传入（rank 0 广播），确保与启动日志在同一文件
        self.log_file = None
        if self.global_rank == 0:
            from datetime import datetime
            ts = log_ts if log_ts else datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(self.output_dir / f"train_{ts}.log", "a", buffering=1, encoding="utf-8")

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.teacher_momentum
        for p_t, p_s in zip(self.teacher.parameters(), self.model.module.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)

    def _all_gather_diff(self, tensor: torch.Tensor) -> torch.Tensor:
        """可微 all_gather: 收集所有 rank 的 tensor 并在 dim=0 拼接.

        与 dist.all_gather 不同，此版本保留梯度回传路径，
        使依赖跨 rank 聚合张量的损失（VICReg、erank 等）在多卡下正常提供梯度。
        """
        if not dist.is_initialized() or self.world_size <= 1:
            return tensor
        from torch.distributed.nn.functional import all_gather as _diff_all_gather
        gathered = _diff_all_gather(tensor)
        return torch.cat(gathered, dim=0)

    def _reduce_loss_dict(self, loss_dict: dict) -> dict:
        if not dist.is_initialized() or self.world_size <= 1:
            return loss_dict
        reduced = {}
        for k, v in loss_dict.items():
            t = torch.tensor(v, device=self.device)
            # NPU 兼容性: 某些驱动不支持 ReduceOp.AVG，使用 SUM + 手动除法
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            reduced[k] = t.item() / self.world_size
        return reduced

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        t = self.cfg.training
        # ★ Projector Warmup: 前 N epoch 只训练投影头，冻结 backbone
        warmup_ep = getattr(t, 'distill_projector_warmup_epochs', 0)
        if warmup_ep > 0:
            if epoch < warmup_ep:
                for name, p in self.model.module.named_parameters():
                    p.requires_grad_('distill_head' in name)
            elif epoch == warmup_ep:
                for p in self.model.module.parameters():
                    p.requires_grad_(True)
                if self.global_rank == 0:
                    print(f'[Distill Stage2] epoch={epoch}: backbone 解冻，进入全量微调')
        accum_steps = getattr(t, "gradient_accumulation_steps", 2)
        loss_accum: dict[str, float] = {}
        n_steps = 0

        # Kappa
        kappa = getattr(t, 'kappa_start', 0.0) or getattr(self.cfg.model, 'vmf_kappa', 2000.0)
        self.model.module.bottleneck.kappa = kappa
        self.teacher.bottleneck.kappa = kappa

        # ★ v40: Curriculum Learning — 先推散(epoch 0~N)，后注入语义(epoch N+)
        # ★ v41: 扩展为双教师 Curriculum，蒸馏权重从 start → end 渐进
        curriculum_epochs = getattr(t, 'curriculum_epochs', 0)
        use_curriculum = getattr(t, 'use_curriculum', False)
        
        if curriculum_epochs > 0 and use_curriculum:
            alpha = min(1.0, epoch / max(curriculum_epochs, 1))  # 0 -> 1
            recon_w = (1 - alpha) * getattr(t, 'curriculum_recon_weight', 0.0) + alpha * t.reconstruction_weight
            pre_norm_uniform_w = (1 - alpha) * getattr(t, 'curriculum_pre_norm_uniform_weight', 3.0) + alpha * getattr(t, 'pre_norm_uniform_weight', 1.0)
            erank_loss_w = (1 - alpha) * getattr(t, 'curriculum_erank_loss_weight', 1.0) + alpha * getattr(t, 'erank_loss_weight', 0.2)
            
            # 双教师蒸馏 Curriculum
            distill_start = getattr(t, 'curriculum_start_weight', 0.3)
            distill_end = getattr(t, 'curriculum_end_weight', 1.0)
            distill_scale = (1 - alpha) * distill_start + alpha * distill_end
            
            aef_spatial_w = getattr(t, 'aef_spatial_distill_weight', 0.0) * distill_scale
            aef_global_w = getattr(t, 'aef_global_distill_weight', 0.0) * distill_scale
            olmo_spatial_w = getattr(t, 'olmoearth_spatial_distill_weight', 0.0) * distill_scale
            olmo_global_w = getattr(t, 'olmoearth_global_distill_weight', 0.0) * distill_scale
            
            if self.global_rank == 0:
                print(f"[Curriculum] epoch={epoch} alpha={alpha:.2f} distill_scale={distill_scale:.2f} "
                      f"aef_sp={aef_spatial_w:.2f} olmo_sp={olmo_spatial_w:.2f} erank_w={erank_loss_w:.2f}")
        else:
            recon_w = t.reconstruction_weight
            pre_norm_uniform_w = getattr(t, 'pre_norm_uniform_weight', 0.0)
            erank_loss_w = getattr(t, 'erank_loss_weight', 0.0)
            aef_spatial_w = getattr(t, 'aef_spatial_distill_weight', 0.0)
            aef_global_w = getattr(t, 'aef_global_distill_weight', 0.0)
            olmo_spatial_w = getattr(t, 'olmoearth_spatial_distill_weight', 0.0)
            olmo_global_w = getattr(t, 'olmoearth_global_distill_weight', 0.0)

        consist_w = getattr(t, 'consistency_weight', 0.0)
        temporal_w = getattr(t, 'temporal_contrastive_weight', 0.0)
        temporal_gap_aware_w = getattr(t, 'temporal_gap_aware_weight', 0.0)
        coding_rate_w = getattr(t, 'coding_rate_weight', 0.0)
        use_l2_vicreg = getattr(t, 'use_l2_space_vicreg', False)
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
            scale = getattr(self.cfg.training, 'backbone_lr_scale', 1.0)
            for pg in self.optimizer.param_groups:
                if scale != 1.0 and pg.get("name") == "backbone":
                    pg["lr"] = lr * scale
                else:
                    pg["lr"] = lr

            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=False):
                # Teacher forward
                has_dual = all(k in batch for k in ['valid_start_w1', 'valid_end_w1', 'valid_start_w2', 'valid_end_w2'])
                # V13: empty cache before teacher forward to avoid OOM
                # ★ FIX: 每 step 调用 empty_cache 导致严重性能下降和内存碎片化
                # 改为每 100 step 调用一次
                if hasattr(torch.npu, 'empty_cache') and step % 100 == 0:
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
                        target_valid_start_ms=batch.get("target_valid_start_ms"),
                        target_valid_end_ms=batch.get("target_valid_end_ms"),
                        dual_window=has_dual,
                        valid_start_w2=batch.get("valid_start_w2") if has_dual else None,
                        valid_end_w2=batch.get("valid_end_w2") if has_dual else None,
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

                if self.global_rank == 0 and step == 0:
                    n_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
                    n_no_grad = sum(1 for p in self.model.parameters() if not p.requires_grad)
                
                student_out = self.model(
                    source_frames=student_frames,
                    source_timestamps_ms=batch["source_timestamps_ms"],
                    source_frame_mask=student_frame_mask,
                    source_input_mask=student_input_mask,
                    source_type_ids=batch["source_type_ids"],
                    valid_start_ms=batch["valid_start_w1"] if has_dual else batch["valid_start_ms"],
                    valid_end_ms=batch["valid_end_w1"] if has_dual else batch["valid_end_ms"],
                    target_relative_time=batch["target_relative_time"],
                    target_metadata=batch["target_metadata"],
                    target_loss_type=batch.get("target_loss_type"),
                    target_source_idx=batch.get("target_source_idx"),
                    target_valid_start_ms=batch.get("target_valid_start_ms"),
                    target_valid_end_ms=batch.get("target_valid_end_ms"),
                    # V28 fix: student 不跑 dual_window（避免 HBM OOM）
                    # W2 目标用 teacher_out.dual_pre_w2（EMA，已在 no_grad 内）
                    dual_window=False,
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

                # V29: 时序对比损失（纯斥力 hinge loss，不含吸引力）
                # teacher W1 vs teacher W2（双窗口，EMA 无梯度）→ 时序不变性监督
                # student W1 vs teacher W2（有梯度）→ 学生端时序感知
                # 关键修复：v28 gap_aware MSE 在 small gap 时 target=1 → 引发坍缩
                # 新方案：纯斥力 hinge loss，只要 cos_sim > target_margin 就惩罚
                temporal_loss = torch.tensor(0.0, device=self.device)
                if temporal_w > 0 and has_dual and teacher_out.dual_pre_w2 is not None:
                    # 教师双窗口：teacher(W1) vs teacher(W2) — 提供稳定时序梯度
                    tc_teacher = temporal_contrastive_loss(
                        teacher_out.pre_norm_map.detach(),
                        teacher_out.dual_pre_w2.detach(),
                        temperature=1.0,
                        target_margin=0.2,
                    )
                    # 学生-教师跨窗口：student(W1) vs teacher(W2) — 学生端有梯度
                    tc_cross = temporal_contrastive_loss(
                        student_out.pre_norm_map,
                        teacher_out.dual_pre_w2.detach(),
                        temperature=1.0,
                        target_margin=0.2,
                    )
                    temporal_loss = (tc_teacher + tc_cross) / 2.0

                # gap_aware MSE 已废弃（v28 根因：small gap 时 target=1 引发坍缩）
                # 若需 gap-aware 行为，应改用 hinge-only 变体（target_margin = f(gap)）
                gap_aware_temporal_loss = torch.tensor(0.0, device=self.device)
                if temporal_gap_aware_w > 0 and has_dual and teacher_out.dual_pre_w2 is not None:
                    w1_center = (batch["valid_start_w1"] + batch["valid_end_w1"]) / 2.0
                    w2_center = (batch["valid_start_w2"] + batch["valid_end_w2"]) / 2.0
                    gap_ms = (w2_center - w1_center).abs().float()
                    # Hinge-only gap-aware：target_margin 随 gap 增大而减小（越大 gap 越要推开）
                    # 注意：不用 MSE；用 F.relu(cos_sim - margin) 不含吸引力
                    gap_norm = torch.clamp(gap_ms / (6 * 30 * 24 * 3600 * 1000), 0.0, 1.0)
                    margin_map = (0.5 * (1.0 - gap_norm)).mean().item()  # 标量 margin
                    gap_aware_temporal_loss = temporal_contrastive_loss(
                        student_out.pre_norm_map,
                        teacher_out.dual_pre_w2.detach(),
                        temperature=1.0,
                        target_margin=float(margin_map),
                    )

                # V19: Inter-Patch InfoNCE (NT-Xent) — 防止方向坍缩
                # 不同patch的teacher(key)和student(query)作为正负样本对
                # 对方向坍缩梯度强：完全坍缩时 loss=log(N)≈2.77（最大梯度）
                infonce_w = getattr(t, 'inter_patch_infonce_weight', 0.0)
                infonce_temp = getattr(t, 'inter_patch_infonce_temperature', 0.1)
                inter_infonce = torch.tensor(0.0, device=self.device)
                if infonce_w > 0:
                    inter_infonce = inter_patch_infonce_loss(
                        student_out.pre_norm_embedding,
                        teacher_out.pre_norm_embedding.detach(),
                        temperature=infonce_temp,
                    )

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

                # ★ 实例判别损失: 预测 patch 身份 (0 ~ N_patches-1)
                # 坍缩时 CE = log(N_patches) ≈ 5.77 (N=320)，梯度对每个 patch 方向不同 → 打破坍缩
                patch_id_w = getattr(t, 'patch_id_loss_weight', 0.0)
                patch_id_loss = torch.tensor(0.0, device=self.device)
                if patch_id_w > 0 and "patch_index" in batch and student_out.patch_id_logits is not None:
                    pid_logits = student_out.patch_id_logits.float()
                    pid_labels = batch["patch_index"]
                    hw = student_out.patch_id_spatial_hw
                    if hw > 1:
                        # spatial token PID: expand labels [B] → [B*HW]
                        _B_pid = pid_labels.shape[0]
                        pid_labels = pid_labels.unsqueeze(1).expand(_B_pid, hw).reshape(_B_pid * hw)
                    patch_id_loss = F.cross_entropy(pid_logits, pid_labels)
                elif student_out.patch_id_logits is not None:
                    # Dummy: 确保 patch_id_head 参数参与 backward
                    dummy_cls = dummy_cls + student_out.patch_id_logits.sum() * 0.0
                    if student_out.distill_map is not None:
                        dummy_cls = dummy_cls + student_out.distill_map.sum() * 0.0

            # === VICReg Variance + Covariance (L2-norm 或 Pre-norm 空间) + Memory Bank ===
            # V13: 支持在 L2-norm 空间计算 VICReg，强制模型在球面上学习时间方向
            if use_l2_vicreg:
                pre_norm = student_out.embedding           # [B, D] — L2 normalized
                pre_norm_map = student_out.embedding_map   # [B, D, H, W] — L2 normalized
            else:
                pre_norm = student_out.pre_norm_embedding  # [B, D] — 真正的 pre-norm
                pre_norm_map = student_out.pre_norm_map    # [B, D, H, W] — spatial pre-norm
            
            # V13: skip NaN/Inf check (fixed in model.encode_frames fallback)
            
            gathered_pre = self._all_gather_diff(pre_norm)

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
                gathered_spatial_map = self._all_gather_diff(pre_norm_map)
                # [B_total, D, H, W] → [B_total*H*W, D]
                spatial_flat = gathered_spatial_map.permute(0, 2, 3, 1).reshape(-1, gathered_spatial_map.shape[1])
                var = variance_regularizer(spatial_flat.float(), min_std=vicreg_min_std) if var_w > 0 else torch.tensor(0.0, device=self.device)
                cov = covariance_loss(spatial_flat.float()) if cov_w > 0 else torch.tensor(0.0, device=self.device)
            else:
                # ★ FIX (v29): 用 gathered_pre（当前 batch，全梯度）而非 all_pre（含 detach bank）
                # all_pre 中 bank 占 ~1024/1040 ≈ 99% 无梯度 → 方差梯度稀释 65×
                # gathered_pre N=16, D=64 → 方差估计噪声但梯度完整，是 VICReg 原设计
                var = variance_regularizer(gathered_pre.float(), min_std=vicreg_min_std) if var_w > 0 else torch.tensor(0.0, device=self.device)
                cov = covariance_loss(gathered_pre.float()) if cov_w > 0 else torch.tensor(0.0, device=self.device)
            
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

            # V22: Inter-Patch Decorrelation (Barlow Twins on gathered_pre)
            # 注意：N=16 < D=64 时 Barlow Twins 估计不稳定（rank 不足），已确认无效
            inter_decorr_w = getattr(t, 'inter_decorr_weight', 0.0)
            inter_decorr = torch.tensor(0.0, device=self.device)
            if inter_decorr_w > 0 and gathered_pre is not None and gathered_pre.shape[0] >= 2:
                inter_decorr = decorrelation_loss(gathered_pre.float())

            # V23: 直接 erank 最大化（SVD 奇异值熵）
            # ★ v39 FIX: 改在 all_pre（含 memory bank，N≈4112）上计算
            #   之前用 gathered_pre（N=16），max erank=16 << D=64，推散力严重不足
            #   改用 all_pre 后 N>>D，max erank≈64，直接优化列方差均匀分布
            # ★ v40: 使用 curriculum 动态 erank_loss_w（已在 train_epoch 开头计算）
            erank_loss_val = torch.tensor(0.0, device=self.device)
            if erank_loss_w > 0 and all_pre is not None and all_pre.shape[0] >= 2:
                erank_loss_val = erank_maximization_loss(all_pre.float())

            # V28: MCR² Coding Rate Loss — 直接 log-det 优化（绕过 VICReg 的 N<D 梯度上界问题）
            # 在 all_pre（含 memory bank，N >> D）上计算，保证协方差矩阵满秩
            # 梯度 ∂logdet/∂C = C⁻¹，弱维度梯度最大，直接均衡奇异值分布
            coding_rate_val = torch.tensor(0.0, device=self.device)
            if coding_rate_w > 0 and all_pre is not None and all_pre.shape[0] >= 2:
                coding_rate_val = coding_rate_loss(all_pre.float(), eps=0.5)

            # === Uniformity Loss（实验变体支持）===
            use_spatial_unif = getattr(t, 'use_spatial_uniformity', False)
            use_pre_norm_unif = getattr(t, 'use_pre_norm_uniform', False)
            use_spatial_raw_unif = getattr(t, 'use_spatial_raw_uniformity', False)
            l2_uniform_w = getattr(t, 'batch_uniformity_weight', 0.05)
            # ★ v40: pre_norm_uniform_w 已在 train_epoch 开头通过 curriculum 动态计算

            if use_spatial_raw_unif and pre_norm_uniform_w > 0:
                # V13-explore: Spatial raw uniformity on pre_norm_map [B, D, H, W] — 完全绕过 GMP + L2
                gathered_map_unif = self._all_gather_diff(pre_norm_map)
                # [B, D, H, W] → [B*H*W, D]
                spatial_flat_unif = gathered_map_unif.permute(0, 2, 3, 1).reshape(-1, gathered_map_unif.shape[1])
                l2_uniform = raw_uniformity_loss(spatial_flat_unif.float())
                l2_uniform_w = pre_norm_uniform_w
            elif use_pre_norm_unif and pre_norm_uniform_w > 0:
                # Pre-norm raw uniformity（欧氏空间，无 L2 Jacobian 屏障）
                # ★ 使用 all_pre（含 memory bank，~528样本）而非 gathered_pre（16样本）
                # 样本数更多 → 梯度信噪比高 ~30倍，uniformity 对坍缩的抵抗力更强
                # 对齐 archive/ddp_xuannv_v2_trainer.py（exp_v2 成功的实现）
                l2_uniform = raw_uniformity_loss(all_pre.float())
                l2_uniform_w = pre_norm_uniform_w
            elif use_spatial_unif and l2_uniform_w > 0:
                # Spatial uniformity on embedding_map [B, D, H, W]
                emb_map = student_out.embedding_map  # [B, D, H, W]
                gathered_map = self._all_gather_diff(emb_map)
                l2_uniform = batch_uniformity_loss_l2(gathered_map.float())
            else:
                # Standard L2 uniformity on embedding vector
                embedding = student_out.embedding  # [B, D]
                gathered_l2 = self._all_gather_diff(embedding)
                l2_uniform = batch_uniformity_loss_l2(gathered_l2.float()) if l2_uniform_w > 0 else torch.tensor(0.0, device=self.device)

            # ★ 球面 Uniformity Loss — 双级别：像素级 + 全局平均级
            #   像素级: 每个 patch 内随机采样 8 像素，防止像素方向坍缩
            #   全局级: pre_norm_embedding ([B,D]) 全局平均，防止 patch 间全局方向坍缩
            #   之前只用像素级 → 模型学到所有 patch 全局平均朝同一方向（erank≈2）而像素内多样
            hsph_uniform_w = getattr(t, 'hyperspherical_uniform_weight', 0.0)
            hsph_uniform = torch.tensor(0.0, device=self.device)
            hsph_pixel = torch.tensor(0.0, device=self.device)
            hsph_global = torch.tensor(0.0, device=self.device)
            if hsph_uniform_w > 0:
                # Part A: 像素级（pre_norm_map 随机采样，梯度不经全局平均稀释）
                if student_out.pre_norm_map is not None:
                    pre_norm_map_s = student_out.pre_norm_map  # [B, D, H, W]
                    B_s, D_s, H_s, W_s = pre_norm_map_s.shape
                    n_px = min(8, H_s * W_s)
                    perm = torch.randperm(H_s * W_s, device=self.device)[:n_px]
                    flat_px = pre_norm_map_s.permute(0, 2, 3, 1).reshape(B_s, H_s * W_s, D_s)
                    sampled_px = flat_px[:, perm, :].reshape(B_s * n_px, D_s)  # [B*n_px, D]
                    gathered_px = self._all_gather_diff(sampled_px)
                    hsph_pixel = hyperspherical_uniformity_loss(gathered_px.float())
                else:
                    hsph_pixel = torch.tensor(0.0, device=self.device)

                # Part B: 全局平均级 — 使用 pairwise cosine（坍缩时梯度非零，与 Wang-Isola 相比更稳健）
                # ★ 关键修复: Wang-Isola 在完全坍缩时 zi-zj≈0 → 梯度≈0（无法逃脱陷阱）
                #             pairwise_cosine 梯度 = 其他样本均值方向，坍缩时非零 ✓
                gathered_global = self._all_gather_diff(pre_norm)
                hsph_global = pairwise_cosine_diversity_loss(gathered_global.float())

                # 两级别加权合并
                # hsph_pixel ∈ [-∞,0]（更负=更好），hsph_global ∈ [-1,1]（更小=更好）
                # total += w * (0.5*px + 1.5*g)：坍缩时 g≈1 → 正值 → 优化器推散; 正常时 g≈0 → 负值 ✓
                hsph_uniform = 0.5 * hsph_pixel + 1.5 * hsph_global

            # ★ 球面方差正则化：在 L2 归一化后的 embedding 上强制各维度方差 ≥ min_std
            # 补充 Wang-Isola 损失，从"维度填充"角度防止坍缩（各维度信息量均等）
            spherical_var_w = getattr(t, 'spherical_variance_weight', 0.0)
            spherical_var = torch.tensor(0.0, device=self.device)
            if spherical_var_w > 0 and gathered_pre.shape[0] >= 2:
                z_sph = F.normalize(gathered_pre.float(), p=2, dim=1)  # [N, D] 单位球面
                sph_min_std = getattr(t, 'spherical_variance_min_std', 0.1)
                spherical_var = variance_regularizer(z_sph, min_std=sph_min_std)

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

                # 有效秩（erank）监控 — 检测 embedding 空间坍缩
                # 使用 all_pre（含 memory bank，N >> D=64），协方差矩阵满秩，可测真实维度利用率
                # 范围：1（完全坍缩）→ D=embedding_dim（理想均匀），目标 > D/2 = 32
                # ★ SPEED FIX: SVD 在 NPU 上每 step 0.45s，改为每 50 step 计算一次
                if step % 50 == 0:
                    try:
                        _z = all_pre.float()
                        _z = _z - _z.mean(dim=0)
                        _svs = torch.linalg.svdvals(_z.T @ _z)
                        _p = _svs / (_svs.sum() + 1e-9)
                        erank = torch.exp(-(_p * (_p + 1e-9).log()).sum()).item()
                    except Exception:
                        erank = 0.0
                    self._last_erank = erank
                else:
                    erank = getattr(self, '_last_erank', 0.0)

                # ★ 新增：样本间多样性诊断
                inter_std = torch.sqrt(gathered_pre.var(dim=0, unbiased=False) + 1e-6)
                inter_active = (inter_std > 0.05).sum().item()
                inter_std_mean = inter_std.mean().item()

            # Decorrelation Loss (Barlow Twins)
            # ★ v41: 改在 all_pre（含 memory bank，N≈4112）上计算
            #   之前用 spatial_flat（空间维度 decorrelation）或 gathered_pre（N=32<D）
            #   都偏离了 Barlow Twins 的本意：样本间协方差矩阵 → I
            #   N=4112 >> D=64，协方差矩阵满秩，硬约束有效
            decorr_w = getattr(t, 'decorrelation_weight', 0.0)
            decorr = torch.tensor(0.0, device=self.device)
            if decorr_w > 0 and all_pre is not None and all_pre.shape[0] >= 2:
                decorr = decorrelation_loss(all_pre.float())

            # V13: Bottleneck Orthogonality Loss
            orth_w = getattr(t, 'orthogonality_weight', 0.0)
            orth = torch.tensor(0.0, device=self.device)
            if orth_w > 0:
                orth = bottleneck_orthogonality_loss(self.model.module.bottleneck.to_embedding.weight)

            # V35: AEF Batch Rotation Uniformity Loss
            aef_uniform_w = getattr(t, 'aef_batch_uniformity_weight', 0.0)
            aef_uniform = torch.tensor(0.0, device=self.device)
            if aef_uniform_w > 0 and gathered_pre is not None and gathered_pre.shape[0] >= 2:
                aef_uniform = aef_batch_uniformity_loss(gathered_pre.float())

            # LMIM: 潜在空间预测损失（替代像素重建的语义增强，来源：OlmoEarth/AnySat）
            lmim_w = getattr(t, 'lmim_weight', 0.0)
            lmim = torch.tensor(0.0, device=self.device)
            if lmim_w > 0 and student_out.pre_norm_map is not None and teacher_out.pre_norm_map is not None:
                lmim = latent_mim_loss(student_out.pre_norm_map, teacher_out.pre_norm_map)

            # ── Teacher 1: AEF (64D) — 直接对齐 ──
            aef_spatial_val = torch.tensor(0.0, device=self.device)
            aef_global_val = torch.tensor(0.0, device=self.device)
            
            if (aef_spatial_w > 0 or aef_global_w > 0) and student_out.pre_norm_map is not None:
                aef_spatial_emb = batch.get("aef_spatial_emb", None)
                if aef_spatial_emb is not None:
                    aef_spatial_emb = aef_spatial_emb.to(self.device).float()  # (B, 64, H, W)
                    student_64 = student_out.pre_norm_map.float()  # (B, 64, H, W)
                    
                    # 分辨率匹配
                    if aef_spatial_emb.shape[2:] != student_64.shape[2:]:
                        aef_spatial_emb = F.adaptive_avg_pool2d(aef_spatial_emb, student_64.shape[2:])
                    
                    if aef_spatial_w > 0:
                        # 空间蒸馏：cosine similarity
                        aef_spatial_val = (1.0 - F.cosine_similarity(
                            student_64, aef_spatial_emb, dim=1, eps=1e-6)).mean()
                    
                    if aef_global_w > 0:
                        # 全局蒸馏：空间平均后对齐
                        aef_global_emb = batch.get("aef_global_emb", None)
                        if aef_global_emb is not None:
                            aef_global_emb = aef_global_emb.to(self.device).float()  # (B, 64)
                        else:
                            aef_global_emb = aef_spatial_emb.mean(dim=(2, 3))  # fallback
                        student_global = student_64.mean(dim=(2, 3))  # (B, 64)
                        aef_global_val = (1.0 - F.cosine_similarity(
                            student_global, aef_global_emb, dim=1, eps=1e-6)).mean()

            # ── Teacher 2: OlmoEarth (768D) — 投影头对齐 ──
            # 注意: olmo_spatial_w / olmo_global_w 已在 curriculum 逻辑中定义
            olmo_spatial_val = torch.tensor(0.0, device=self.device)
            olmo_global_val = torch.tensor(0.0, device=self.device)
            
            if (olmo_spatial_w > 0 or olmo_global_w > 0) and student_out.distill_map is not None:
                teacher_tok = batch.get("teacher_spatial_tokens", None)
                if teacher_tok is not None:
                    teacher_tok = teacher_tok.to(self.device)  # (B, 32, 32, 768)
                    teacher_raw = teacher_tok.permute(0, 3, 1, 2).float()  # (B, 768, 32, 32)
                    student_map = student_out.distill_map.float()          # (B, 768, h, w)
                    if olmo_spatial_w > 0:
                        # 空间蒸馏：teacher 池化到 student 原生分辨率
                        if teacher_raw.shape[2:] != student_map.shape[2:]:
                            teacher_sp = F.adaptive_avg_pool2d(teacher_raw, student_map.shape[2:])
                        else:
                            teacher_sp = teacher_raw
                        t_cent = teacher_sp - teacher_sp.mean(dim=(2, 3), keepdim=True)
                        s_cent = student_map - student_map.mean(dim=(2, 3), keepdim=True)
                        olmo_spatial_val = (1.0 - F.cosine_similarity(s_cent, t_cent, dim=1, eps=1e-6)).mean()
                    if olmo_global_w > 0:
                        teacher_global_ref = batch.get("teacher_global_emb", None)
                        if teacher_global_ref is not None:
                            teacher_global_ref = teacher_global_ref.to(self.device).float()
                        else:
                            teacher_global_ref = teacher_raw.mean(dim=(2, 3))
                        olmo_global_val = (1.0 - F.cosine_similarity(
                            student_out.distill_global.float(), teacher_global_ref, dim=1, eps=1e-6)).mean()

            total = (
                recon_w * recon_warmup * recon
                + consist_w * consist
                + temporal_w * temporal_loss
                + temporal_gap_aware_w * gap_aware_temporal_loss
                + infonce_w * inter_infonce
                + cls_w * cls
                + patch_id_w * patch_id_loss
                + var_w * var
                + cov_w * cov
                + l2_uniform_w * l2_uniform
                + hsph_uniform_w * hsph_uniform
                + spherical_var_w * spherical_var
                + decorr_w * decorr
                + orth_w * orth
                + aef_uniform_w * aef_uniform
                + dummy_cls
                + inter_var_w * inter_var
                + inter_decorr_w * inter_decorr
                + erank_loss_w * erank_loss_val
                + coding_rate_w * coding_rate_val
                + lmim_w * lmim
                + aef_spatial_w * aef_spatial_val
                + aef_global_w * aef_global_val
                + olmo_spatial_w * olmo_spatial_val
                + olmo_global_w * olmo_global_val
            )

            if torch.isnan(total) or torch.isinf(total):
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"  [WARNING] NaN/Inf total at step {step}, doing dummy backward.")
                # ★ FIX: 必须执行 dummy backward 以保持 DDP all_reduce 同步
                # 直接 continue 会导致缺失 backward 的 rank 不参与 all_reduce → DDP 死锁
                dummy_sync = total * 0.0
                if self.scaler is not None:
                    self.scaler.scale(dummy_sync).backward()
                else:
                    dummy_sync.backward()
                self.optimizer.zero_grad()
                # 继续执行后续梯度累积逻辑（不 step），保持所有 rank 同步

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
                    # ★ FIX: update_teacher 移入 else 分支
                    # 只有成功执行 optimizer.step() 的 rank 才更新 teacher
                    # 防止不同 rank 的 teacher 因 has_nan_grad 差异而 diverge
                    self.update_teacher()
                
                # V13: weight health check (silent)
                # for name, p in self.model.named_parameters():
                #     if p is not None and (torch.isnan(p).any() or torch.isinf(p).any()):

            # ========== 精简损失累加（只保留关键指标） ==========
            for k, v in {
                "total": total.item() * accum_steps,
                "recon": recon.item(),
                "cls": cls.item(),
                "var": var.item(),
                "cov": cov.item(),
                "l2unif": l2_uniform.item(),
                "erank": float(erank),
                "aef_sp": aef_spatial_val.item(),
                "aef_gl": aef_global_val.item(),
                "olmo_sp": olmo_spatial_val.item(),
                "olmo_gl": olmo_global_val.item(),
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

            # ========== 精简 Step 日志（每 10 步或最后一步打印） ==========
            if self.global_rank == 0:
                print_interval = 10
                if step == 0 or step == len(dataloader) - 1 or step % print_interval == 0:
                    step_msg = (f"\n[Step {step:3d}/{len(dataloader)}] "
                          f"total={total.item()*accum_steps:.3f} "
                          f"recon={recon.item():.3f} cls={cls.item():.3f} "
                          f"var={var.item():.3f} cov={cov.item():.3f} "
                          f"l2unif={l2_uniform.item():.3f} "
                          f"erank={erank:.1f} "
                          f"aef=[sp={aef_spatial_val.item():.3f},gl={aef_global_val.item():.3f}] "
                          f"olmo=[sp={olmo_spatial_val.item():.3f},gl={olmo_global_val.item():.3f}] "
                          f"lr={lr:.6f}")
                    print(step_msg)
                    if self.log_file:
                        self.log_file.write(step_msg + "\n")
                        self.log_file.flush()

        loss_accum = {k: v / max(n_steps, 1) for k, v in loss_accum.items()}
        # ★ FIX: temporal 损失需要在平均化之前累加，否则记录的是最后一步的值
        if temporal_w > 0:
            loss_accum["temporal"] = loss_accum.get("temporal", 0.0) / max(n_steps, 1)
        else:
            loss_accum["temporal"] = 0.0
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

    @torch.no_grad()
    def evaluate_knn(self, dataloader: DataLoader, max_batches: int | None = None) -> dict:
        """In-process kNN 探针评估：global embedding (64D) → patch-level WorldCover 众数类别.

        衡量蒸馏后底座表征的可分性（acc / mIoU）。各 rank 收集自身分片，
        all_gather_object 汇总到 rank0 计算，返回 {"acc", "miou"}（非 rank0 返回 0）。
        """
        was_training = self.model.training
        self.model.eval()
        embs: list = []
        labels: list = []
        import itertools
        iterator = itertools.islice(dataloader, max_batches) if max_batches else dataloader
        for batch in iterator:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if "label" not in batch:
                continue
            with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=False):
                has_dual = all(k in batch for k in ['valid_start_w1', 'valid_end_w1', 'valid_start_w2', 'valid_end_w2'])
                out = self.model(
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
                    target_valid_start_ms=batch.get("target_valid_start_ms"),
                    target_valid_end_ms=batch.get("target_valid_end_ms"),
                    dual_window=False,
                )
            emb = out.embedding.float()  # (B, 64)
            emb = F.normalize(emb, dim=1)
            embs.append(emb.cpu())
            labels.append(batch["label"].cpu())

        if was_training:
            self.model.train()

        import numpy as np
        if not embs:
            local_emb = np.array([], dtype=np.float32).reshape(0, 0)
            local_lab = np.array([], dtype=np.int64)
        else:
            local_emb = torch.cat(embs, dim=0).numpy()
            local_lab = torch.cat(labels, dim=0).numpy()

        # 跨 rank 汇总
        if self.world_size > 1 and dist.is_initialized():
            gathered: list = [None] * self.world_size
            dist.all_gather_object(gathered, (local_emb, local_lab))
            if self.global_rank != 0:
                return {"acc": 0.0, "miou": 0.0}
            valid = [g for g in gathered if g[0].shape[0] > 0]
            if not valid:
                return {"acc": 0.0, "miou": 0.0}
            all_emb = np.concatenate([g[0] for g in valid], axis=0)
            all_lab = np.concatenate([g[1] for g in valid], axis=0)
        else:
            if local_emb.shape[0] == 0:
                return {"acc": 0.0, "miou": 0.0}
            all_emb, all_lab = local_emb, local_lab

        return self._knn_metric(all_emb, all_lab)

    @staticmethod
    def _knn_metric(emb, lab, k: int = 5, seed: int = 42) -> dict:
        """patch-level kNN（cosine）train/test split → acc + mIoU."""
        import numpy as np
        n = len(emb)
        if n < 4:
            return {"acc": 0.0, "miou": 0.0}
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_tr = n // 2
        tr, te = perm[:n_tr], perm[n_tr:]
        Xtr, ytr = emb[tr], lab[tr]
        Xte, yte = emb[te], lab[te]
        sim = Xte @ Xtr.T  # 已 L2 归一化
        kk = min(k, len(tr))
        topk = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        preds = np.empty(len(te), dtype=lab.dtype)
        for i, idx in enumerate(topk):
            u, c = np.unique(ytr[idx], return_counts=True)
            preds[i] = u[np.argmax(c)]
        acc = float((preds == yte).mean())
        classes = np.unique(np.concatenate([yte, preds]))
        ious = []
        for cls in classes:
            inter = np.logical_and(preds == cls, yte == cls).sum()
            union = np.logical_or(preds == cls, yte == cls).sum()
            if union > 0:
                ious.append(inter / union)
        miou = float(np.mean(ious)) if ious else 0.0
        return {"acc": acc, "miou": miou}

    def save_checkpoint(self, epoch, losses: dict, miou: float | None = None,
                        tag: str | None = None) -> None:
        """保存 checkpoint.

        - tag="last": 覆盖式断点 epoch_last.pt（含 optimizer，用于 OOM 后续训）
        - miou 提供: best 权重 epoch_best_miou{miou}_ep{epoch}.pt，按 mIoU 仅保留最优 3 个
        - 否则: 普通 epoch_{epoch}.pt
        """
        if self.global_rank != 0:
            return
        import re
        if tag == "last":
            path = self.output_dir / "epoch_last.pt"
        elif miou is not None:
            path = self.output_dir / f"epoch_best_miou{miou:.4f}_ep{epoch}.pt"
        else:
            path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else {},
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else {},
            "losses": losses,
            "miou": miou,
        }, path)
        print(f"[trainer] Saved checkpoint to {path}")

        # 清理普通 checkpoint（非 best/last），只保留最近 1 个
        ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_*.pt")
             if not p.name.startswith("epoch_best") and p.name != "epoch_last.pt"],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in ckpts[:-1]:
            old_ckpt.unlink()
            print(f"[trainer] Removed old checkpoint: {old_ckpt}")

        # ★ best 权重：按文件名内 mIoU 数值排序，只保留最优 3 个
        def _miou_of(p):
            m = re.search(r"miou([0-9.]+)_", p.name)
            return float(m.group(1)) if m else -1.0
        best_ckpts = sorted(self.output_dir.glob("epoch_best_miou*.pt"), key=_miou_of)
        for old_ckpt in best_ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[trainer] Removed lower-mIoU checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
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
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except ValueError as e:
                if self.global_rank == 0:
                    print(f"[load_checkpoint] Scheduler mismatch: {e}")
        epoch = ckpt.get("epoch", 0)
        if not isinstance(epoch, int):
            import re
            m = re.search(r'(\d+)', str(epoch))
            epoch = int(m.group(1)) if m else 0
        return epoch

    def soft_restart(self, path: str) -> None:
        """从指定 checkpoint 加载模型权重，但不恢复 epoch/optimizer 状态.

        用途：跨域迁移学习（哈尔滨→海淀），保留权重、丢弃训练进度，从 epoch 0 开始。
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.module.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if self.global_rank == 0:
            print(f"[soft_restart] Loaded weights from {path}")
            print(f"[soft_restart] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        # 同步教师模型权重
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        if self.global_rank == 0:
            print("[soft_restart] Teacher synced. Training starts from epoch 0.")
