"""DDP V4 训练器 — EMA Teacher + DINO + VICReg + KoLeo + CT Reconstruction.

设计约束:
- 仅使用 GPU 6,7 双卡
- 每卡 batch_size=2, gradient_accumulation=4 模拟有效 batch=16
- EMA Teacher 每 rank 独立维护（因 DDP 已同步 student 梯度）
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
from src.training.losses import reconstruction_loss
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


def koleo_loss(x: torch.Tensor) -> torch.Tensor:
    # Force fp32 for cdist / log numerical stability under autocast
    x = x.float()
    x = F.normalize(x, p=2, dim=-1)
    dists = torch.cdist(x, x, p=2)
    eye = torch.eye(dists.shape[0], device=dists.device)
    dists = dists + eye * 1e6
    nn_dists = dists.min(dim=1)[0]
    return -torch.log(nn_dists + 1e-8).mean()


def vicreg_loss(z_student: torch.Tensor, z_teacher: torch.Tensor,
                lambda_inv: float = 1.0, mu_var: float = 1.0, nu_cov: float = 0.04) -> torch.Tensor:
    inv = F.mse_loss(z_student, z_teacher)
    z_all = torch.cat([z_student, z_teacher], dim=0)
    std = torch.sqrt(z_all.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std))
    z_all = z_all - z_all.mean(dim=0, keepdim=True)
    cov = (z_all.T @ z_all) / (z_all.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / cov.shape[0]
    return lambda_inv * inv + mu_var * var + nu_cov * cov_loss


def build_optimizer_from_params(params, cfg):
    t = cfg.training
    # eps=1e-4 for fp16 stability (default 1e-8 underflows in half precision)
    return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay, eps=1e-4)


class DDPv4Trainer:
    """DDP V4 训练器 — 双卡 GPU 6/7."""

    def __init__(self, cfg: Config, local_rank: int = 0) -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Student 模型
        self.model = AEFModel(cfg).to(self.device)
        self.model = DistributedDataParallel(
            self.model, device_ids=[local_rank], find_unused_parameters=False
        )

        # 加载预训练权重
        pretrained_path = getattr(cfg, "pretrained", None)
        if pretrained_path and self.global_rank == 0:
            ckpt = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.module.load_state_dict(state_dict, strict=False)
            print(f"[ddp_v4] Loaded pretrained weights from {pretrained_path}")
        if dist.is_initialized():
            dist.barrier()

        # EMA Teacher（独立，不包装 DDP）
        self.teacher = copy.deepcopy(self.model.module)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = getattr(cfg.training, "teacher_momentum", 0.996)

        # 优化器: backbone only
        params = list(self.model.parameters())
        self.optimizer = build_optimizer_from_params(params, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # GradScaler for mixed-precision training stability
        self.scaler = torch.cuda.amp.GradScaler()

        # 输出目录
        self.output_dir = Path(cfg.experiment.output_dir)
        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        emb_dim = cfg.model.embedding_dim

        # Expander for VICReg
        expander_dim = getattr(cfg.training, "expander_dim", 0)
        if expander_dim > 0:
            self.expander = nn.Sequential(
                nn.Linear(emb_dim, expander_dim), nn.GELU(),
                nn.Linear(expander_dim, expander_dim),
            ).to(self.device)
            # expander 不需要 DDP，因为它只在 student/teacher pair 上使用，不涉及跨卡通信
            # 但为了安全，如果希望同步其梯度，也可以包装
            self.expander = DistributedDataParallel(
                self.expander, device_ids=[local_rank], find_unused_parameters=False
            )
        else:
            self.expander = None

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
        if self.expander is not None:
            self.expander.train()
        t = self.cfg.training
        accum_steps = getattr(t, "gradient_accumulation_steps", 4)
        loss_accum: dict[str, float] = {}
        n_steps = 0

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            lr = get_cosine_lr(epoch, step, len(dataloader), self.cfg)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                # Student forward
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

                recon = self._compute_recon_loss(student_out.reconstructions, batch)

                # Teacher forward (no grad)
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

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):

                # VICReg + KoLeo
                z_s = student_out.pre_norm_embedding
                z_t = teacher_out.pre_norm_embedding
                if self.expander is not None:
                    z_s = self.expander(z_s)
                    z_t = self.expander(z_t)
                vicreg = vicreg_loss(z_s, z_t)
                koleo = koleo_loss(torch.cat([z_s, z_t], dim=0))

                # Temporal
                temporal = torch.tensor(0.0, device=self.device)
                temporal_w = getattr(t, 'temporal_contrastive_weight', 0.0)
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
                        temporal = self._temporal_loss(pre_w1, pre_w2, t)
                        if self.global_rank == 0 and temporal.item() == 0.0:
                            # Debug: print why temporal is zero
                            f1 = F.adaptive_avg_pool2d(pre_w1, 1).view(pre_w1.shape[0], -1)
                            f2 = F.adaptive_avg_pool2d(pre_w2, 1).view(pre_w2.shape[0], -1)
                            f1 = F.normalize(f1, p=2, dim=-1)
                            f2 = F.normalize(f2, p=2, dim=-1)
                            sim = (f1 * f2).sum(dim=-1).mean().item()
                            print(f"  [Temporal DEBUG] sim={sim:.4f} pre_w1_mean={pre_w1.mean().item():.4f} pre_w2_mean={pre_w2.mean().item():.4f}")
                    except Exception as e:
                        if self.global_rank == 0:
                            print(f"  [Temporal] Error: {e}")

                # CT reconstruction
                ct_recon = torch.tensor(0.0, device=self.device)
                ct_recon_w = getattr(t, "ct_reconstruction_weight", 0.5)
                if ct_recon_w > 0 and "spatial_mask" in batch and "valid_start_w2" in batch:
                    ct_recon = self._cross_temporal_masked_recon(batch, student_out)

                # Dummy loss for unused classification heads — 让 DDP 所有参数都参与梯度
                dummy_cls = torch.tensor(0.0, device=self.device)
                if student_out.logits is not None:
                    dummy_cls = dummy_cls + student_out.logits.sum() * 0.0
                if student_out.aux_logits is not None:
                    dummy_cls = dummy_cls + student_out.aux_logits.sum() * 0.0
                if student_out.bottleneck_logits is not None:
                    dummy_cls = dummy_cls + student_out.bottleneck_logits.sum() * 0.0

                # Recon warmup
                recon_warmup = min(1.0, (epoch + 1) / max(t.recon_warmup_epochs, 1))
                recon_weight = t.reconstruction_weight * recon_warmup

                total = (
                    recon_weight * recon
                    + ct_recon_w * ct_recon
                    + getattr(t, "vicreg_weight", 1.0) * vicreg
                    + getattr(t, "koleo_weight", 0.1) * koleo
                    + temporal_w * temporal
                    + dummy_cls
                )
                total = total / accum_steps

            self.scaler.scale(total).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + (list(self.expander.parameters()) if self.expander is not None else []),
                    getattr(t, "grad_clip_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.update_teacher()

            for k, v in {
                "total": total.item() * accum_steps,
                "recon": recon.item(),
                "ct_recon": ct_recon.item(),
                "dino": 0.0,
                "vicreg": vicreg.item(),
                "koleo": koleo.item(),
                "temporal": temporal.item(),
                "lr": lr,
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

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

    def _temporal_loss(self, pre_w1, pre_w2, t_cfg):
        from src.training.losses import temporal_info_nce_loss, temporal_contrastive_loss
        loss_type = getattr(t_cfg, 'temporal_loss_type', 'hinge')
        temp = getattr(t_cfg, 'temporal_contrastive_temperature', 0.1)
        if loss_type == 'info_nce_antidiagonal':
            return temporal_info_nce_loss(pre_w1, pre_w2, temperature=temp)
        return temporal_contrastive_loss(pre_w1, pre_w2, temperature=temp)

    def _cross_temporal_masked_recon(self, batch, student_out) -> torch.Tensor:
        spatial_mask = batch["spatial_mask"]
        if spatial_mask.dim() == 2:
            spatial_mask = spatial_mask.unsqueeze(0)
        pred = student_out.reconstructions
        tgt = batch["target_images"]
        mask = batch["target_mask"]
        B, T, C, H, W = pred.shape
        sm = F.interpolate(spatial_mask.unsqueeze(1), size=(H, W), mode="nearest")
        sm = sm.unsqueeze(1)
        pixel_valid = (~torch.isnan(tgt)).float()
        tgt_mask = mask[:, :, None, None, None].float() * pixel_valid
        diff = torch.abs(pred - tgt) * tgt_mask * sm
        denom = torch.clamp((tgt_mask * sm).sum(), min=1.0)
        return diff.sum() / denom

    def save_checkpoint(self, epoch: int, losses: dict) -> None:
        if self.global_rank != 0:
            return
        path = self.output_dir / f"epoch_{epoch}.pt"
        
        # 收集所有需要保存的 state dict
        extra_state = {}
        if self.expander is not None:
            extra_state["expander"] = self.expander.state_dict()
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "losses": losses,
            "extra_state": extra_state,
        }, path)
        print(f"[ddp_v4] Saved checkpoint to {path}")
        
        # 自动清理：只保留最新的 3 个数字 epoch checkpoint
        ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_*.pt") if not p.name.startswith("epoch_best")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in ckpts[:-3]:
            old_ckpt.unlink()
            print(f"[ddp_v4] Removed old checkpoint: {old_ckpt}")

        # 自动清理：只保留最新的 2 个 best checkpoint
        best_ckpts = sorted(
            [p for p in self.output_dir.glob("epoch_best_*.pt")],
            key=lambda p: p.stat().st_mtime
        )
        for old_ckpt in best_ckpts[:-2]:
            old_ckpt.unlink()
            print(f"[ddp_v4] Removed old best checkpoint: {old_ckpt}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing and self.global_rank == 0:
            print(f"[load_checkpoint] Missing keys: {len(missing)}")
        if unexpected and self.global_rank == 0:
            print(f"[load_checkpoint] Unexpected keys: {len(unexpected)}")
        
        # 加载 expander
        if "extra_state" in ckpt:
            es = ckpt["extra_state"]
            if "expander" in es and self.expander is not None:
                self.expander.load_state_dict(es["expander"])
                if self.global_rank == 0:
                    print("[load_checkpoint] Expander loaded.")
        
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
        # 兼容 "best_epoch47" 等非整数字符串 epoch
        if not isinstance(epoch, int):
            import re
            m = re.search(r'(\d+)', str(epoch))
            epoch = int(m.group(1)) if m else 0
        return epoch
