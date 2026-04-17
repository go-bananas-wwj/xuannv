"""单 GPU 训练器 — 支持 EMA Teacher、跨时相掩码重建、VICReg + KoLeo.

设计约束:
- 仅使用 GPU6 单卡
- batch_size=2, gradient_accumulation=8 模拟有效 batch=16
- EMA Teacher 提供稳定监督信号，补偿小 batch 统计噪声
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config
from src.models.model import AEFModel
from src.training.losses import reconstruction_loss
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


# ────────────────────────────────────────────
# KoLeo 正则化
# ────────────────────────────────────────────

def koleo_loss(x: torch.Tensor) -> torch.Tensor:
    """Kozachenko-Leonenko 熵估计正则化 — 强制 batch 内 embedding 均匀分散.

    Args:
        x: [N, D] embedding，已 L2 归一化或未归一化均可.
    Returns:
        标量损失，越小表示分布越均匀.
    """
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    eye = torch.eye(dists.shape[0], device=dists.device)
    dists = dists + eye * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()


# ────────────────────────────────────────────
# VICReg Loss (Teacher-Student Pair 版本)
# ────────────────────────────────────────────

def vicreg_loss(z_student: torch.Tensor, z_teacher: torch.Tensor,
                lambda_inv: float = 1.0, mu_var: float = 1.0, nu_cov: float = 0.04) -> torch.Tensor:
    """VICReg，利用 teacher-student pair 扩充有效 batch.

    Args:
        z_student: [B, D]
        z_teacher: [B, D]
    """
    # Invariance
    inv = F.mse_loss(z_student, z_teacher)

    # Variance: 合并 teacher + student 计算
    z_all = torch.cat([z_student, z_teacher], dim=0)  # [2B, D]
    std = torch.sqrt(z_all.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std))

    # Covariance
    z_all = z_all - z_all.mean(dim=0, keepdim=True)
    cov = (z_all.T @ z_all) / (z_all.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / cov.shape[0]

    return lambda_inv * inv + mu_var * var + nu_cov * cov_loss


# ────────────────────────────────────────────
# DINO Head
# ────────────────────────────────────────────

class DINOHead(nn.Module):
    """DINO projector + predictor head."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, bottleneck_dim: int = 128,
                 n_prototypes: int = 4096) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototypes = nn.Linear(bottleneck_dim, n_prototypes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = F.normalize(x, p=2, dim=-1)
        return self.prototypes(x)


class DINOLoss(nn.Module):
    """DINO 分布匹配损失."""

    def __init__(self, n_prototypes: int = 4096, teacher_temp: float = 0.07,
                 student_temp: float = 0.1, center_momentum: float = 0.9) -> None:
        super().__init__()
        self.register_buffer("center", torch.zeros(1, n_prototypes))
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: [B, K]
            teacher_logits: [B, K]
        """
        student_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)
        teacher_probs = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1)
        loss = -torch.sum(teacher_probs * student_probs, dim=-1).mean()

        # 更新 center (EMA)
        with torch.no_grad():
            batch_center = teacher_probs.mean(dim=0, keepdim=True)
            self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

        return loss


# ────────────────────────────────────────────
# Single GPU Trainer
# ────────────────────────────────────────────

class SingleGPUTrainer:
    """单 GPU 训练器 — GPU6 专用."""

    def __init__(self, cfg: Config, device_str: str = "cuda:6") -> None:
        self.cfg = cfg
        self.device = torch.device(device_str)

        # Student 模型
        self.model = AEFModel(cfg).to(self.device)

        # 加载预训练权重
        pretrained_path = getattr(cfg, "pretrained", None)
        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[single_gpu] Loaded pretrained weights from {pretrained_path}")

        # EMA Teacher
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = getattr(cfg.training, "teacher_momentum", 0.996)

        # DINO Head
        emb_dim = cfg.model.embedding_dim
        self.dino_head = DINOHead(in_dim=emb_dim).to(self.device)
        self.dino_loss = DINOLoss().to(self.device)

        # 优化器: 同时优化 backbone + DINO head
        params = list(self.model.parameters()) + list(self.dino_head.parameters())
        self.optimizer = build_optimizer_from_params(params, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # 输出目录
        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Expander for VICReg (可选)
        expander_dim = getattr(cfg.training, "expander_dim", 0)
        if expander_dim > 0:
            self.expander = nn.Sequential(
                nn.Linear(emb_dim, expander_dim),
                nn.GELU(),
                nn.Linear(expander_dim, expander_dim),
            ).to(self.device)
        else:
            self.expander = None

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA 更新 Teacher."""
        m = self.teacher_momentum
        for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        t = self.cfg.training
        accum_steps = getattr(t, "gradient_accumulation_steps", 8)
        loss_accum: dict[str, float] = {}
        n_steps = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 学习率
            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # ── Student 前向 (完整输入) ──
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                student_out = self.model(
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
                recon = self._compute_recon_loss(student_out.reconstructions, batch)

                # ── Teacher 前向 (完整输入，无 grad) ──
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
                    )

                # DINO Loss (全局 embedding)
                s_logits = self.dino_head(student_out.embedding)
                t_logits = self.dino_head(teacher_out.embedding)
                dino = self.dino_loss(s_logits, t_logits)

                # VICReg + KoLeo (使用 teacher-student pair)
                z_s = student_out.pre_norm_embedding
                z_t = teacher_out.pre_norm_embedding
                if self.expander is not None:
                    z_s = self.expander(z_s)
                    z_t = self.expander(z_t)
                vicreg = vicreg_loss(z_s, z_t)
                koleo = koleo_loss(torch.cat([z_s, z_t], dim=0))

                # 时序对比损失 (w1 vs w2)
                temporal = torch.tensor(0.0, device=self.device)
                temporal_w = getattr(t, 'temporal_contrastive_weight', 0.0)
                if temporal_w > 0 and "valid_start_w1" in batch:
                    try:
                        emb_w1, emb_w2, pre_w1, pre_w2 = self.model.encode_dual_window(
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
                        temporal = self._temporal_loss(pre_w1, pre_w2, t)
                    except Exception as e:
                        print(f"  [Temporal] Error: {e}")

                # 跨时相掩码重建损失
                ct_recon = torch.tensor(0.0, device=self.device)
                ct_recon_w = getattr(t, "ct_reconstruction_weight", 0.5)
                if ct_recon_w > 0 and "spatial_mask" in batch and "valid_start_w2" in batch:
                    ct_recon = self._cross_temporal_masked_recon(batch, student_out)

                # Recon warmup
                recon_warmup = min(1.0, (epoch + 1) / max(t.recon_warmup_epochs, 1))
                recon_weight = t.reconstruction_weight * recon_warmup

                # 总损失
                total = (
                    recon_weight * recon
                    + ct_recon_w * ct_recon
                    + getattr(t, "dino_weight", 0.1) * dino
                    + getattr(t, "vicreg_weight", 1.0) * vicreg
                    + getattr(t, "koleo_weight", 0.1) * koleo
                    + temporal_w * temporal
                )

                # 梯度累积缩放
                total = total / accum_steps

            # 反向传播
            total.backward()

            # 步进优化器
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_teacher()

            # 记录
            for k, v in {
                "total": total.item() * accum_steps,
                "recon": recon.item(),
                "ct_recon": ct_recon.item(),
                "dino": dino.item(),
                "vicreg": vicreg.item(),
                "koleo": koleo.item(),
                "temporal": temporal.item(),
                "lr": lr,
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

        return {k: v / n_steps for k, v in loss_accum.items()}

    def _compute_recon_loss(self, predictions, batch):
        from src.training.loops import compute_recon_loss
        return compute_recon_loss(
            predictions, batch["target_images"], batch["target_mask"],
            batch.get("target_loss_type"), self.cfg.data.num_classes,
        )

    def _temporal_loss(self, pre_w1, pre_w2, t_cfg):
        from src.training.losses import temporal_info_nce_loss, temporal_contrastive_loss
        loss_type = getattr(t_cfg, 'temporal_loss_type', 'hinge')
        temp = getattr(t_cfg, 'temporal_contrastive_temperature', 0.1)
        if loss_type == 'info_nce_antidiagonal':
            return temporal_info_nce_loss(pre_w1, pre_w2, temperature=temp)
        return temporal_contrastive_loss(pre_w1, pre_w2, temperature=temp)

    def _cross_temporal_masked_recon(self, batch, student_out) -> torch.Tensor:
        """跨时相掩码重建: 对 w2 的输入做空间掩码, 重建目标只计算掩码区域.

        实现: 用 spatial_mask 对 target_images 加权, 让模型关注被掩码区域.
        """
        spatial_mask = batch["spatial_mask"]  # [B, H, W] or [H, W]
        if spatial_mask.dim() == 2:
            spatial_mask = spatial_mask.unsqueeze(0)  # [1, H, W]

        # 目前只对 S2 目标做掩码重建 (target_source_idx 中找 s2)
        # 简单做法: 对所有 target 的重建做空间掩码加权
        pred = student_out.reconstructions  # [B, T_tgt, C, H, W]
        tgt = batch["target_images"]
        mask = batch["target_mask"]  # [B, T_tgt]

        # 空间掩码上采样到 target 分辨率 (H/2)
        B, T, C, H, W = pred.shape
        sm = F.interpolate(spatial_mask.unsqueeze(1), size=(H, W), mode="nearest")  # [B, 1, H, W]
        sm = sm.unsqueeze(1)  # [B, 1, 1, H, W]

        # 加权 MSE: 掩码区域权重高
        pixel_valid = (~torch.isnan(tgt)).float()
        tgt_mask = mask[:, :, None, None, None].float() * pixel_valid
        diff = torch.abs(pred - tgt) * tgt_mask * sm
        denom = torch.clamp((tgt_mask * sm).sum(), min=1.0)
        return diff.sum() / denom

    def save_checkpoint(self, epoch: int, losses: dict) -> None:
        path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": losses,
        }, path)
        print(f"[single_gpu] Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.teacher = copy.deepcopy(self.model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt.get("epoch", 0)


def build_optimizer_from_params(params, cfg):
    """从参数列表构建优化器."""
    import torch
    t = cfg.training
    return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
