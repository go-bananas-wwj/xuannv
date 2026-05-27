"""DDP Unified Trainer — AEF 对齐版本.

核心设计 (对齐 AEF 原文):
- 损失: Reconstruction(1.0) + BatchUniformity(0.05) + Consistency(0.02)
- 无 VICReg, 无 Decorr, 无 Orth, 无 CLS
- Teacher-Student 一致性为核心机制
- Decoder 条件注入恢复 (时间条件化重建)
- Bottleneck 训练时 L2+VMF (skip_l2=false)
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
    batch_uniformity_cyclic_shift,
    consistency_loss_spatial,
)
from src.training.memory_bank import EmbeddingMemoryBank
from src.training.optimizer import build_optimizer, build_scheduler, get_cosine_lr


# ---------------------------------------------------------------------------
# Student View 构建 (从 v12 复制)
# ---------------------------------------------------------------------------
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


class DDPUnifiedTrainer:
    """AEF 对齐 Unified Trainer — 只保留核心损失."""

    def __init__(self, cfg: Config, local_rank: int = 0, wandb_group: str = "aef_v1") -> None:
        self.cfg = cfg
        self.local_rank = local_rank
        self.device = torch.device(f"npu:{local_rank}")
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Student 模型
        self.model = AEFModel(cfg).to(self.device)
        if dist.is_initialized() and self.world_size > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[local_rank], find_unused_parameters=True
            )

        # EMA Teacher
        model_for_copy = self.model.module if hasattr(self.model, 'module') else self.model
        self.teacher = copy.deepcopy(model_for_copy)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = getattr(cfg.training, "teacher_momentum", 0.996)

        # 优化器
        self.optimizer = build_optimizer(model_for_copy, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        self.scaler = None

        self.output_dir = Path(cfg.experiment.output_dir)
        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 源特定重建权重
        self.source_recon_weights = torch.tensor(
            getattr(cfg.training, 'source_recon_weights', [1.0, 1.0, 1.0, 0.1]),
            dtype=torch.float32,
            device=self.device,
        )

        self.log_file = None
        if self.global_rank == 0:
            self.log_file = open(self.output_dir / "train.log", "a", buffering=1, encoding="utf-8")
        
        # Memory Bank — 扩大 batch uniformity 有效 batch
        emb_dim = getattr(cfg.model, "embedding_dim", 64)
        self.memory_bank = EmbeddingMemoryBank(K=512, dim=emb_dim, device=self.device)
        
        # Wandb（仅 rank 0）
        self.use_wandb = False
        if self.global_rank == 0:
            try:
                import wandb
                wandb.init(
                    project="aef-alignment",
                    name=cfg.experiment.name,
                    group=wandb_group or "xuannv_round1",
                    config={
                        "experiment": cfg.experiment.name,
                        "embedding_dim": getattr(cfg.model, "embedding_dim", 64),
                        "skip_l2": getattr(cfg.model, "skip_l2_norm_training", False),
                        "consistency_weight": getattr(cfg.training, "consistency_weight", 0.02),
                        "batch_uniformity_weight": getattr(cfg.training, "batch_uniformity_weight", 0.05),
                        "reconstruction_weight": getattr(cfg.training, "reconstruction_weight", 1.0),
                        "source_recon_weights": getattr(cfg.training, "source_recon_weights", [1.0]*4),
                        "vmf_kappa": getattr(cfg.model, "vmf_kappa", 50.0),
                    },
                )
                self.use_wandb = True
            except Exception as e:
                print(f"[WARN] Wandb init failed: {e}")

    @torch.no_grad()
    def update_teacher(self) -> None:
        m = self.teacher_momentum
        model_params = self.model.module.parameters() if hasattr(self.model, 'module') else self.model.parameters()
        for p_t, p_s in zip(self.teacher.parameters(), model_params):
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
        accum_steps = getattr(t, "gradient_accumulation_steps", 1)
        loss_accum: dict[str, float] = {}
        n_steps = 0

        # Kappa
        kappa = getattr(t, 'kappa_start', 0.0) or getattr(self.cfg.model, 'vmf_kappa', 50.0)
        model_for_kappa = self.model.module if hasattr(self.model, 'module') else self.model
        model_for_kappa.bottleneck.kappa = kappa
        self.teacher.bottleneck.kappa = kappa

        recon_w = t.reconstruction_weight
        consist_w = getattr(t, 'consistency_weight', 0.02)
        uniform_w = getattr(t, 'batch_uniformity_weight', 0.05)
        recon_warmup = min(1.0, (epoch + 1) / max(getattr(t, 'recon_warmup_epochs', 1), 1))

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
                if hasattr(torch.npu, 'empty_cache'):
                    torch.npu.empty_cache()
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

                # Consistency (teacher vs student spatial map)
                if consist_w > 0:
                    consist = consistency_loss_spatial(
                        teacher_out.embedding_map.detach(),
                        student_out.embedding_map,
                    )
                else:
                    consist = torch.tensor(0.0, device=self.device)

                # Batch Uniformity (L2 space) + Memory Bank
                if uniform_w > 0:
                    embedding = student_out.embedding  # [B, D]
                    if dist.is_initialized() and self.world_size > 1:
                        gathered = [torch.zeros_like(embedding) for _ in range(self.world_size)]
                        dist.all_gather(gathered, embedding)
                        gathered = torch.cat(gathered, dim=0)
                    else:
                        gathered = embedding
                    
                    # Memory Bank: 扩大有效 batch
                    self.memory_bank.enqueue(gathered.detach())
                    bank_emb = self.memory_bank.get_all()
                    if bank_emb.shape[0] > 0:
                        all_emb = torch.cat([gathered, bank_emb], dim=0)
                    else:
                        all_emb = gathered
                    
                    unif_type = getattr(t, "batch_uniformity_type", "all_pairs")
                    if unif_type == "cyclic_shift":
                        l2_uniform = batch_uniformity_cyclic_shift(all_emb.float())
                    else:
                        l2_uniform = batch_uniformity_loss_l2(all_emb.float())
                else:
                    l2_uniform = torch.tensor(0.0, device=self.device)

            # === 总损失: 只有 Recon + Consist + Uniform ===
            total = (
                recon_w * recon_warmup * recon
                + consist_w * consist
                + uniform_w * l2_uniform
            )

            if torch.isnan(total) or torch.isinf(total):
                if self.global_rank == 0 and step % 10 == 0:
                    print(f"  [WARNING] NaN/Inf total at step {step}, skipping.")
                self.optimizer.zero_grad()
                continue

            total = total / accum_steps
            total.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    getattr(t, "grad_clip_norm", 1.0)
                )
                has_nan_grad = False
                model_params = self.model.module.parameters() if hasattr(self.model, 'module') else self.model.parameters()
                for p in model_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    if self.global_rank == 0:
                        print(f"  [WARNING] NaN/Inf gradient detected, skipping optimizer step.")
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.update_teacher()

            # 日志
            for k, v in {
                "total": total.item() * accum_steps,
                "recon": recon.item(),
                "consist": consist.item(),
                "l2unif": l2_uniform.item(),
                "lr": lr,
            }.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_steps += 1

            if self.global_rank == 0:
                bank_size = self.memory_bank.size
                step_msg = (f"  [Step {step}] recon={recon.item():.4f} "
                      f"consist={consist.item():.4f} "
                      f"l2unif={l2_uniform.item():.4f} "
                      f"bank={bank_size}/512 "
                      f"perturb=[fd={perturb_stats['frame_drop_ratio']:.2f} "
                      f"sd={perturb_stats['source_drop_ratio']:.2f}] "
                      f"lr={lr:.6f}")
                print(step_msg)
                if self.log_file:
                    self.log_file.write(step_msg + "\n")
                    self.log_file.flush()
                
                if self.use_wandb and step % 5 == 0:
                    import wandb
                    wandb.log({
                        "step/recon": recon.item(),
                        "step/consist": consist.item(),
                        "step/l2unif": l2_uniform.item(),
                        "step/bank_size": bank_size,
                        "step/lr": lr,
                        "step/frame_drop": perturb_stats["frame_drop_ratio"],
                        "step/source_drop": perturb_stats["source_drop_ratio"],
                    }, step=epoch * 1000 + step)

        loss_accum = {k: v / n_steps for k, v in loss_accum.items()}
        loss_accum = self._reduce_loss_dict(loss_accum)
        loss_accum["lr"] = lr
        loss_accum["bank"] = float(self.memory_bank.size)
        
        if self.global_rank == 0 and self.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "epoch/total": loss_accum["total"],
                "epoch/recon": loss_accum["recon"],
                "epoch/consist": loss_accum["consist"],
                "epoch/l2unif": loss_accum["l2unif"],
                "epoch/bank": loss_accum["bank"],
                "epoch/lr": loss_accum["lr"],
            })
        
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
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": losses,
        }, path)
        print(f"[unified] Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model_for_load = self.model.module if hasattr(self.model, 'module') else self.model
        missing, unexpected = model_for_load.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing and self.global_rank == 0:
            print(f"[load_checkpoint] Missing keys: {len(missing)}")
        if unexpected and self.global_rank == 0:
            print(f"[load_checkpoint] Unexpected keys: {len(unexpected)}")
        self.teacher = copy.deepcopy(model_for_load)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as e:
                if self.global_rank == 0:
                    print(f"[load_checkpoint] Optimizer mismatch: {e}")
        epoch = ckpt.get("epoch", 0)
        if not isinstance(epoch, int):
            import re
            m = re.search(r'(\d+)', str(epoch))
            epoch = int(m.group(1)) if m else 0
        return epoch
