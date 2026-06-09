"""AEF 训练器 — 适配 NPU + DDP."""
from __future__ import annotations

import contextlib
import copy
import itertools
import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.loss_function import AEFLoss


class Trainer:
    """AEF Trainer with DDP support."""

    def __init__(
        self,
        model: AlphaEarthFoundations,
        dataloader,
        val_dataloader,
        device: str = "npu",
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        max_steps: int = 100000,
        warmup_steps: int = 2000,
        log_every: int = 50,
        save_every: int = 500,
        eval_every: int = 5000,
        output_dir: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        distill_warmup_steps: int = 1000,
        grad_accum_steps: int = 1,
        resume_step: int = 0,
        resume_optimizer_state: dict | None = None,
        resume_scheduler_state: dict | None = None,
        resume_ema_state: dict | None = None,
        seed: int = 42,
        viz_patch_ids: list[str] | None = None,
        reconstruction_weight: float = 1.0,
        distill_weight: float = 0.2,
        spatial_distill_weight: float = 0.0,
        uniformity_weight: float = 0.05,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/aef_haidian")
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.distill_warmup_steps = distill_warmup_steps
        self.grad_accum_steps = grad_accum_steps
        self.viz_patch_ids = viz_patch_ids

        self.loss_fn = AEFLoss(
            reconstruction_weight=reconstruction_weight,
            distill_weight=distill_weight,
            spatial_distill_weight=spatial_distill_weight,
            uniformity_weight=uniformity_weight,
        )

        # Optimizer
        params = list(self.model.parameters())
        self.optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # LR scheduler with warmup
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

        # Resume optimizer / scheduler 状态
        if resume_optimizer_state is not None:
            self.optim.load_state_dict(resume_optimizer_state)
        if resume_scheduler_state is not None:
            self.scheduler.load_state_dict(resume_scheduler_state)
            self.scheduler.last_epoch = resume_step

        # EMA Teacher 模型
        self.ema_model = None
        if self.world_size > 1 or True:  # 始终启用 EMA
            model_to_copy = model.module if hasattr(model, "module") else model
            self.ema_model = copy.deepcopy(model_to_copy)
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self.ema_model.eval()
            if resume_ema_state is not None:
                self.ema_model.load_state_dict(resume_ema_state)
                if self.rank == 0:
                    print("[Resume] Loaded EMA state")

        # 缓存可视化用的 batch，避免反复实例化 dataloader
        self._viz_batch = None
        if val_dataloader is not None:
            try:
                self._viz_batch = next(iter(val_dataloader))
            except StopIteration:
                pass

        self.loss_history = {
            "steps": [],
            "total": [],
            "reconstruction": [],
            "uniformity": [],
            "consistency": [],
            "distill": [],
            "magnitude": [],
        }

        self.seed = seed
        self.step = resume_step

    @staticmethod
    def _embed_to_rgb(emb: np.ndarray) -> np.ndarray:
        """64D -> 3D PCA RGB. 输入 (B, H, W, D) 或 (H, W, D)."""
        if emb.ndim == 3:
            emb = emb[np.newaxis, ...]
        B, H, W, D = emb.shape
        emb_flat = emb.reshape(-1, D)
        mean = emb_flat.mean(axis=0, keepdims=True)
        emb_centered = emb_flat - mean
        u, s, vt = np.linalg.svd(emb_centered, full_matrices=False)
        rgb = u[:, :3] * s[:3]
        rgb_min, rgb_max = rgb.min(axis=0, keepdims=True), rgb.max(axis=0, keepdims=True)
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
        return rgb.reshape(B, H, W, 3)

    @staticmethod
    def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
        """将 (H, W, C) tensor 转为归一化 RGB numpy. 对每通道单独 min-max.
        单通道数据复制为灰度图."""
        arr = t.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        C = arr.shape[-1]
        rgb = np.zeros((*arr.shape[:2], 3), dtype=np.float32)
        if C == 1:
            # 单通道 -> 灰度
            ch = arr[..., 0]
            ch_min, ch_max = ch.min(), ch.max()
            gray = (ch - ch_min) / (ch_max - ch_min + 1e-8)
            rgb[..., 0] = gray
            rgb[..., 1] = gray
            rgb[..., 2] = gray
        else:
            for c in range(min(C, 3)):
                ch = arr[..., c]
                ch_min, ch_max = ch.min(), ch.max()
                rgb[..., c] = (ch - ch_min) / (ch_max - ch_min + 1e-8)
        return rgb

    @staticmethod
    def _embed_to_rgb_shared(student_emb: np.ndarray, aef_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """用 AEF embedding 的 SVD 基做统一 PCA，保证两者颜色空间可比.
        输入: (H, W, 64) numpy. 输出: (H, W, 3) RGB, 范围 [0,1]."""
        H, W, D = aef_emb.shape
        # 以 AEF 为参考计算 PCA 基
        aef_flat = aef_emb.reshape(-1, D)
        mean = aef_flat.mean(axis=0)
        centered = aef_flat - mean
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:3].T  # (64, 3)

        # AEF 投影
        aef_centered = aef_flat - mean
        aef_proj = aef_centered @ basis
        aef_proj = (aef_proj - aef_proj.min(axis=0)) / (aef_proj.max(axis=0) - aef_proj.min(axis=0) + 1e-8)
        aef_rgb = aef_proj.reshape(H, W, 3)

        # Student 用同一套 mean + basis 投影
        student_flat = student_emb.reshape(-1, D)
        student_centered = student_flat - mean
        student_proj = student_centered @ basis
        student_proj = (student_proj - student_proj.min(axis=0)) / (student_proj.max(axis=0) - student_proj.min(axis=0) + 1e-8)
        student_rgb = student_proj.reshape(H, W, 3)

        return student_rgb, aef_rgb

    def _log(self, msg: str) -> None:
        if self.rank == 0:
            print(msg)

    def _prepare_reconstruction_targets(
        self, batch: dict[str, Any], pred: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """从 batch 中准备重建目标（取时间中点最近的一帧）."""
        targets: dict[str, torch.Tensor] = {}
        for src_key in pred.keys():
            if src_key not in batch["source_data"]:
                continue
            x = batch["source_data"][src_key].to(self.device)  # (B, T, H, W, C)
            ts = batch["timestamps"][src_key].to(self.device)  # (B, T)
            B, T, H, W, C = x.shape
            center = ts.mean(dim=1, keepdim=True)  # (B, 1)
            idx = (ts - center).abs().argmin(dim=1)  # (B,)
            batch_indices = torch.arange(B, device=self.device)
            target = x[batch_indices, idx]  # (B, H, W, C)

            # 下采样到重建分辨率（decoder 直接输出全分辨率，无需额外下采样）
            H2, W2 = pred[src_key].shape[1], pred[src_key].shape[2]
            target_2d = rearrange(target, "b h w c -> b c h w")
            # 分类目标用 nearest 插值，避免破坏类别语义；需先转 float 再转回 long
            config = self.loss_fn.source_configs.get(src_key, {"loss_fn": F.l1_loss})
            if config["loss_fn"] == F.cross_entropy:
                target_2d_float = target_2d.float()
                target_2d_float = F.interpolate(target_2d_float, size=(H2, W2), mode="nearest")
                target_2d = target_2d_float.long()
            else:
                target_2d = F.interpolate(target_2d, size=(H2, W2), mode="bilinear", align_corners=False)
            target = rearrange(target_2d, "b c h w -> b h w c")
            targets[src_key] = target
        return targets

    def train(self) -> None:
        self.model.train()

        # 获取 sampler 以便每个 epoch 调用 set_epoch
        train_sampler = getattr(self.dataloader, "sampler", None)
        epoch = 0
        resume_step = self.step  # 记录起始 step，用于计算速度

        pbar = tqdm(
            range(self.step + 1, self.max_steps + 1),
            desc="Training",
            unit="step",
            disable=self.rank != 0,
            initial=self.step,
            total=self.max_steps,
        )
        start_time = time.time()

        data_iter = iter(self.dataloader)

        for step in pbar:
            self.step = step

            # Stage switching for distill alignment
            if step <= self.distill_warmup_steps:
                self.loss_fn.set_stage("distill_align")
                if step == 1 or step == self.distill_warmup_steps:
                    self._log(f"[Stage] distill_align at step {step}")
            else:
                self.loss_fn.set_stage("normal")
                if step == self.distill_warmup_steps + 1:
                    self._log(f"[Stage] normal at step {step}")

            # 每个 epoch 开始时调用 set_epoch（基于 step 数估算）
            steps_per_epoch = len(self.dataloader) if hasattr(self.dataloader, "__len__") else 1000
            current_epoch = (step - 1) // steps_per_epoch
            if current_epoch != epoch and train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                epoch = current_epoch
                train_sampler.set_epoch(epoch)

            # 迭代 dataloader，遇到 StopIteration 时重新创建迭代器
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            source_data = {
                k: v.to(self.device) for k, v in batch["source_data"].items()
            }
            timestamps = {
                k: v.to(self.device) for k, v in batch["timestamps"].items()
            }
            valid_periods = batch["valid_periods"]

            out = self.model(source_data, timestamps, valid_periods)

            # EMA Teacher: 用 EMA 模型重新计算 teacher embedding，覆盖当前模型的 teacher
            if self.ema_model is not None:
                with torch.no_grad():
                    out_ema = self.ema_model(source_data, timestamps, valid_periods)
                out["teacher_embeddings"] = out_ema["embeddings"]

            # predictions: 取第一个 sample（AEF decoder 输出 (B, S, H, W, C)，S=1 时就是 deterministic）
            predictions: dict[str, torch.Tensor] = {}
            for src, rec in out["reconstructions"].items():
                predictions[src] = rec[:, 0]  # (B, H, W, C)

            targets = self._prepare_reconstruction_targets(batch, predictions)

            masks = {
                k: torch.ones_like(v[..., :1], device=self.device)
                for k, v in predictions.items()
            }

            outputs_for_loss: dict[str, Any] = {
                "embeddings": out["embeddings"],
                "teacher_embeddings": out["teacher_embeddings"],
                "student_embeddings": out["student_embeddings"],
                "image_embeddings": out["image_embeddings"],
                "predictions": predictions,
                "targets": targets,
                "masks": masks,
            }

            # AEF 官方 embedding 蒸馏
            if "aef_embedding" in batch:
                aef_emb = batch["aef_embedding"].to(self.device)  # (B, 64, 128, 128)
                valid_mask = batch.get("aef_embedding_valid")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device)
                outputs_for_loss["aef_embedding_pred"] = out["embeddings"]
                outputs_for_loss["aef_embedding_target"] = aef_emb
                outputs_for_loss["aef_embedding_valid"] = valid_mask

            losses = self.loss_fn(outputs_for_loss)
            loss = losses["total"] / self.grad_accum_steps

            # DDP no_sync: 非同步 step 跳过 all_reduce，避免梯度被错误平均
            is_sync_step = step % self.grad_accum_steps == 0
            ctx = (
                self.model.no_sync()
                if (self.world_size > 1 and hasattr(self.model, "no_sync") and not is_sync_step)
                else contextlib.nullcontext()
            )
            with ctx:
                loss.backward()

            # Gradient clipping + optimizer step only after grad_accum_steps
            if is_sync_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                self.scheduler.step()
                # EMA 更新
                if self.ema_model is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
                            p_ema.data.mul_(0.996).add_(p.data, alpha=0.004)
                self.optim.zero_grad(set_to_none=True)

            # Sync losses across ranks (detach before all_reduce)
            if self.world_size > 1:
                with torch.no_grad():
                    for k in losses:
                        tensor = losses[k].detach()
                        if tensor.device != self.device:
                            tensor = tensor.to(self.device)
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        losses[k] = tensor / self.world_size

            # Logging
            if self.rank == 0:
                self.loss_history["steps"].append(step)
                for k in ["total", "reconstruction", "uniformity", "consistency", "distill", "magnitude"]:
                    self.loss_history[k].append(float(losses[k]))

                if step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    actual_steps = step - resume_step
                    steps_per_sec = actual_steps / elapsed if elapsed > 0 else 0
                    eta = (self.max_steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
                    stage_tag = "[Align]" if step <= self.distill_warmup_steps else "[Normal]"
                    spatial_distill_val = losses.get('spatial_distill', torch.tensor(0.0)).item()
                    print(
                        f"[Step {step}] {stage_tag} "
                        f"loss={losses['total']:.4f} "
                        f"recon={losses['reconstruction']:.4f} "
                        f"uniform={losses['uniformity']:.4f} "
                        f"consist={losses['consistency']:.4f} "
                        f"distill={losses['distill']:.4f} "
                        f"spat_dist={spatial_distill_val:.4f} "
                        f"mag={losses['magnitude']:.4f} "
                        f"stripe={losses.get('anti_stripe_embed', torch.tensor(0.0)).item():.4f} "
                        f"lr={lr:.6f} "
                        f"({steps_per_sec:.2f} step/s, ETA {eta:.1f}h)"
                    )

                # Progress bar
                recon = float(losses["reconstruction"])
                pbar.set_postfix({
                    "recon": f"{recon:.4f}",
                    "total": f"{float(loss):.4f}",
                })

            # Save checkpoint + eval + visualize
            if step % self.save_every == 0:
                if self.rank == 0:
                    self._save_checkpoint(step)
                self._eval()
                if self.rank == 0:
                    self._visualize(step)

        pbar.close()
        if self.rank == 0:
            print("\nTraining completed!")

    @torch.no_grad()
    def _eval(self) -> None:
        self.model.eval()
        total_recon = 0.0
        total_uniform = 0.0
        total_var = 0.0
        total_cov = 0.0
        total_distill = 0.0
        total_mag = 0.0
        count = 0

        for batch in self.val_dataloader:
            source_data = {k: v.to(self.device) for k, v in batch["source_data"].items()}
            timestamps = {k: v.to(self.device) for k, v in batch["timestamps"].items()}
            valid_periods = batch["valid_periods"]

            out = self.model(source_data, timestamps, valid_periods)

            predictions = {src: rec[:, 0] for src, rec in out["reconstructions"].items()}
            targets = self._prepare_reconstruction_targets(batch, predictions)
            masks = {k: torch.ones_like(v[..., :1], device=self.device) for k, v in predictions.items()}

            outputs_for_loss = {
                "embeddings": out["embeddings"],
                "teacher_embeddings": out["teacher_embeddings"],
                "student_embeddings": out["student_embeddings"],
                "image_embeddings": out["image_embeddings"],
                "predictions": predictions,
                "targets": targets,
                "masks": masks,
            }

            if "aef_embedding" in batch:
                aef_emb = batch["aef_embedding"].to(self.device)
                valid_mask = batch.get("aef_embedding_valid")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device)
                outputs_for_loss["aef_embedding_pred"] = out["embeddings"]
                outputs_for_loss["aef_embedding_target"] = aef_emb
                outputs_for_loss["aef_embedding_valid"] = valid_mask

            losses = self.loss_fn(outputs_for_loss)
            total_recon += float(losses["reconstruction"])
            total_uniform += float(losses["uniformity"])
            total_var += float(losses["variance"])
            total_cov += float(losses["covariance"])
            total_distill += float(losses["distill"])
            total_mag += float(losses["magnitude"])
            count += 1

        self.model.train()

        # DDP 同步：先同步 count，避免某些 rank count==0 提前 return 导致 deadlock
        count_tensor = torch.tensor(count, device=self.device, dtype=torch.float32)
        if self.world_size > 1:
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        if count_tensor.item() == 0:
            if self.rank == 0:
                print("[Val] No valid batches")
            return

        # DDP 同步：gather sum，再求平均
        if self.world_size > 1:
            recon_tensor = torch.tensor(total_recon, device=self.device, dtype=torch.float32)
            uniform_tensor = torch.tensor(total_uniform, device=self.device, dtype=torch.float32)
            var_tensor = torch.tensor(total_var, device=self.device, dtype=torch.float32)
            cov_tensor = torch.tensor(total_cov, device=self.device, dtype=torch.float32)
            distill_tensor = torch.tensor(total_distill, device=self.device, dtype=torch.float32)
            mag_tensor = torch.tensor(total_mag, device=self.device, dtype=torch.float32)
            for t in [recon_tensor, uniform_tensor, var_tensor, cov_tensor, distill_tensor, mag_tensor]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_recon = float(recon_tensor / count_tensor)
            avg_uniform = float(uniform_tensor / count_tensor)
            avg_var = float(var_tensor / count_tensor)
            avg_cov = float(cov_tensor / count_tensor)
            avg_distill = float(distill_tensor / count_tensor)
            avg_mag = float(mag_tensor / count_tensor)
        else:
            avg_recon = total_recon / count
            avg_uniform = total_uniform / count
            avg_var = total_var / count
            avg_cov = total_cov / count
            avg_distill = total_distill / count
            avg_mag = total_mag / count

        if self.rank == 0:
            print(f"[Val] recon={avg_recon:.4f} uniform={avg_uniform:.4f} var={avg_var:.4f} cov={avg_cov:.4f} distill={avg_distill:.4f} mag={avg_mag:.4f}")

    def _save_checkpoint(self, step: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 保存原始模型的 state_dict（去除 DDP 的 module. 前缀）
        model_state = self.model.state_dict()
        if hasattr(self.model, "module"):
            model_state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in model_state.items()}
        checkpoint = {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "ema_state_dict": self.ema_model.state_dict() if self.ema_model else None,
        }
        torch.save(checkpoint, self.output_dir / f"step_{step:06d}_seed{self.seed}.pt")
        if self.rank == 0:
            print(f"[Checkpoint] Saved at step {step}")

    def _embed_to_rgb(self, emb: np.ndarray) -> np.ndarray:
        """将高维 embedding 用 PCA 降到 3 维并归一化为 RGB."""
        H, W, D = emb.shape
        flat = emb.reshape(-1, D)
        mean = flat.mean(axis=0)
        centered = flat - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        comps = vh[:3]
        proj = centered @ comps.T
        proj = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-8)
        return proj.reshape(H, W, 3)

    @torch.no_grad()
    def _visualize(self, step: int) -> None:
        """2行合并大图：Row0=5源输入, Row1=Student PCA | AEF PCA | Diff.
        对 viz_patch_ids 中指定的每个 patch 各生成一张图."""
        if self.rank != 0:
            return

        self.model.eval()
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.gridspec as gridspec
        from src.aef.data.haidian_dataset import collate_fn

        viz_patch_ids = getattr(self, "viz_patch_ids", None)
        target_patches = set(viz_patch_ids) if viz_patch_ids else set()

        # 从 val_dataset 收集目标 patch，若找不到则从 train_dataset 搜索
        samples_to_viz = []
        if target_patches:
            # 先在 val_dataset 中查找
            val_dataset = self.val_dataloader.dataset
            found_in_val = []
            for idx in range(len(val_dataset)):
                sample = val_dataset[idx]
                if sample["patch_id"] in target_patches:
                    found_in_val.append(sample)
            found_ids = {s["patch_id"] for s in found_in_val}
            samples_to_viz.extend(found_in_val)
            # 对 val 中缺失的 patch，到 train_dataset 中搜索
            missing = target_patches - found_ids
            if missing and hasattr(self, "dataloader") and self.dataloader is not None:
                train_dataset = self.dataloader.dataset
                for idx in range(len(train_dataset)):
                    sample = train_dataset[idx]
                    if sample["patch_id"] in missing:
                        samples_to_viz.append(sample)
        else:
            # 回退到 _viz_batch
            batch = self._viz_batch
            if batch is None:
                batch = next(iter(self.val_dataloader))
            samples_to_viz = [{k: v[0] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}]

        for sample in samples_to_viz:
            patch_id = sample["patch_id"]
            # 组装 batch_size=1
            batch = collate_fn([sample])
            source_data = {k: v.to(self.device) for k, v in batch["source_data"].items()}
            timestamps = {k: v.to(self.device) for k, v in batch["timestamps"].items()}
            valid_periods = batch["valid_periods"]
            out = self.model(source_data, timestamps, valid_periods)

            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.2,
                                   height_ratios=[1, 1.3])
            gs_top = gs[0].subgridspec(1, 5, wspace=0.15)
            gs_bottom = gs[1].subgridspec(1, 3, wspace=0.15)

            # ---- Row 0: 5个输入源 ----
            input_sources = ["s1", "s2", "tianyi_sar", "landsat", "planet"]
            for i, src in enumerate(input_sources):
                ax = fig.add_subplot(gs_top[0, i])
                if src not in batch["source_data"]:
                    ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", fontsize=12)
                    ax.axis("off")
                    continue

                data = batch["source_data"][src][0]  # (T, H, W, C)
                ts = batch["timestamps"][src][0]
                T = data.shape[0]

                if src == "planet":
                    valid_idx = None
                    for t in range(T):
                        if data[t].abs().max() > 0.001:
                            valid_idx = t
                            break
                    if valid_idx is None:
                        ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", fontsize=12)
                        ax.axis("off")
                        continue
                    frame = data[valid_idx]
                else:
                    center = ts.mean()
                    t_idx = (ts - center).abs().argmin().item()
                    frame = data[t_idx]

                rgb = self._tensor_to_rgb(frame)
                ax.imshow(rgb)
                title = src.upper()
                if src == "landsat":
                    title += " [30m→10m]"
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.axis("off")

            # ---- Row 1: Student PCA | AEF PCA | Diff ----
            ax_student = fig.add_subplot(gs_bottom[0, 0])
            ax_aef = fig.add_subplot(gs_bottom[0, 1])
            ax_diff = fig.add_subplot(gs_bottom[0, 2])

            student_emb = out["student_embeddings"][0].detach().cpu().numpy()
            if "aef_embedding" in batch and batch["aef_embedding"] is not None:
                aef_emb = batch["aef_embedding"][0].detach().permute(1, 2, 0).cpu().numpy()
                student_rgb, aef_rgb = self._embed_to_rgb_shared(student_emb, aef_emb)
                diff = np.abs(student_rgb - aef_rgb).mean(axis=-1)

                ax_student.imshow(student_rgb)
                ax_student.set_title("Student (PCA RGB)", fontsize=12, fontweight="bold")
                ax_student.axis("off")

                ax_aef.imshow(aef_rgb)
                ax_aef.set_title("AEF Official (PCA RGB)", fontsize=12, fontweight="bold")
                ax_aef.axis("off")

                im = ax_diff.imshow(diff, cmap="hot")
                ax_diff.set_title("|Student - AEF|", fontsize=12, fontweight="bold")
                ax_diff.axis("off")
                plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
            else:
                for ax in (ax_student, ax_aef, ax_diff):
                    ax.text(0.5, 0.5, "AEF N/A", ha="center", va="center", fontsize=12)
                    ax.axis("off")

            plt.suptitle(f"{patch_id} @ Step {step}", fontsize=14, fontweight="bold", y=0.98)
            plt.savefig(viz_dir / f"viz_step_{step:06d}_{patch_id}_seed{self.seed}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

        self.model.train()
        print(f"[Viz] Saved {len(samples_to_viz)} patch visualizations at step {step}")

