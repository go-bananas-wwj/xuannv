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
        seed: int = 42,
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

        self.loss_fn = AEFLoss()

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
            self.ema_model = copy.deepcopy(model)
            for p in self.ema_model.parameters():
                p.requires_grad = False
            self.ema_model.eval()

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
        }

        self.seed = seed
        self.step = resume_step

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
            # 分类目标用 nearest 插值，避免破坏类别语义
            config = self.loss_fn.source_configs.get(src_key, {"loss_fn": F.l1_loss})
            if config["loss_fn"] == F.cross_entropy:
                target_2d = F.interpolate(target_2d, size=(H2, W2), mode="nearest")
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
                out["embeddings"] = out_ema["embeddings"]

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
                for k in ["total", "reconstruction", "uniformity", "consistency", "distill"]:
                    self.loss_history[k].append(float(losses[k]))

                if step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    actual_steps = step - resume_step
                    steps_per_sec = actual_steps / elapsed if elapsed > 0 else 0
                    eta = (self.max_steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
                    stage_tag = "[Align]" if step <= self.distill_warmup_steps else "[Normal]"
                    print(
                        f"[Step {step}] {stage_tag} "
                        f"loss={losses['total']:.4f} "
                        f"recon={losses['reconstruction']:.4f} "
                        f"uniform={losses['uniformity']:.4f} "
                        f"consist={losses['consistency']:.4f} "
                        f"distill={losses['distill']:.4f} "
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
            for t in [recon_tensor, uniform_tensor, var_tensor, cov_tensor, distill_tensor]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_recon = float(recon_tensor / count_tensor)
            avg_uniform = float(uniform_tensor / count_tensor)
            avg_var = float(var_tensor / count_tensor)
            avg_cov = float(cov_tensor / count_tensor)
            avg_distill = float(distill_tensor / count_tensor)
        else:
            avg_recon = total_recon / count
            avg_uniform = total_uniform / count
            avg_var = total_var / count
            avg_cov = total_cov / count
            avg_distill = total_distill / count

        if self.rank == 0:
            print(f"[Val] recon={avg_recon:.4f} uniform={avg_uniform:.4f} var={avg_var:.4f} cov={avg_cov:.4f} distill={avg_distill:.4f}")

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
        """保存可视化：teacher/student embedding 对比 + PCA RGB + reconstruction."""
        if self.rank != 0:
            return

        self.model.eval()
        batch = self._viz_batch
        if batch is None:
            batch = next(iter(self.val_dataloader))
        source_data = {k: v.to(self.device) for k, v in batch["source_data"].items()}
        timestamps = {k: v.to(self.device) for k, v in batch["timestamps"].items()}
        valid_periods = batch["valid_periods"]

        out = self.model(source_data, timestamps, valid_periods)

        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        patch_ids = batch.get("patch_ids", [f"b{i}" for i in range(len(out["teacher_embeddings"]))])

        # ---- 1. 单维度 Embedding 对比（batch 前 2 个 patch）----
        n_show = min(2, len(out["teacher_embeddings"]))
        fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, 4)
        dims = np.random.choice(64, 4, replace=False)
        for b in range(n_show):
            emb_t = out["teacher_embeddings"][b].cpu().numpy()
            emb_s = out["student_embeddings"][b].cpu().numpy()
            for i, d in enumerate(dims):
                vmin = min(emb_t[..., d].min(), emb_s[..., d].min())
                vmax = max(emb_t[..., d].max(), emb_s[..., d].max())
                if b == 0:
                    axes[b, i].set_title(f"Dim {d}")
                im = axes[b, i].imshow(emb_s[..., d], vmin=vmin, vmax=vmax, cmap="viridis")
                axes[b, i].axis("off")
                plt.colorbar(im, ax=axes[b, i], fraction=0.046)
            axes[b, 0].set_ylabel(f"{patch_ids[b]}", fontsize=12, fontweight="bold")
        plt.suptitle(f"Student Embedding Channels @ Step {step}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"embedding_step_{step:06d}_seed{self.seed}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ---- 2. PCA RGB 全域可视化（batch 前 3 个 patch）----
        n_show = min(3, len(out["teacher_embeddings"]))
        fig, axes = plt.subplots(n_show, 3, figsize=(15, 5 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, 3)
        for b in range(n_show):
            emb_t = out["teacher_embeddings"][b].cpu().numpy()
            emb_s = out["student_embeddings"][b].cpu().numpy()
            rgb_t = self._embed_to_rgb(emb_t)
            rgb_s = self._embed_to_rgb(emb_s)
            axes[b, 0].imshow(rgb_t)
            axes[b, 0].set_title(f"Teacher PCA RGB")
            axes[b, 0].axis("off")
            axes[b, 1].imshow(rgb_s)
            axes[b, 1].set_title(f"Student PCA RGB")
            axes[b, 1].axis("off")
            diff = np.abs(rgb_t - rgb_s).mean(axis=-1)
            im = axes[b, 2].imshow(diff, cmap="hot")
            axes[b, 2].set_title(f"|Teacher - Student|")
            axes[b, 2].axis("off")
            plt.colorbar(im, ax=axes[b, 2], fraction=0.046)
            axes[b, 0].set_ylabel(f"{patch_ids[b]}", fontsize=12, fontweight="bold")
        plt.suptitle(f"PCA RGB Embedding @ Step {step}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"pca_rgb_step_{step:06d}_seed{self.seed}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ---- 3. Reconstruction 对比 ----
        predictions = {src: rec[:, 0] for src, rec in out["reconstructions"].items()}
        targets = self._prepare_reconstruction_targets(batch, predictions)
        src_list = list(predictions.keys())[:3]
        if src_list:
            fig, axes = plt.subplots(2, len(src_list), figsize=(5 * len(src_list), 8))
            if len(src_list) == 1:
                axes = axes.reshape(2, 1)
            for i, src in enumerate(src_list):
                pred = predictions[src][0].cpu().numpy()
                tgt = targets[src][0].cpu().numpy()
                pred_rgb = pred[..., :3]
                tgt_rgb = tgt[..., :3]
                pred_min, pred_max = pred_rgb.min(), pred_rgb.max()
                tgt_min, tgt_max = tgt_rgb.min(), tgt_rgb.max()
                pred_rgb = np.clip((pred_rgb - pred_min) / (pred_max - pred_min + 1e-8), 0, 1)
                tgt_rgb = np.clip((tgt_rgb - tgt_min) / (tgt_max - tgt_min + 1e-8), 0, 1)
                axes[0, i].imshow(tgt_rgb)
                axes[0, i].set_title(f"{src} Target")
                axes[0, i].axis("off")
                axes[1, i].imshow(pred_rgb)
                axes[1, i].set_title(f"{src} Pred")
                axes[1, i].axis("off")
            patch_id = patch_ids[0] if patch_ids else "b0"
            plt.suptitle(f"Reconstruction {patch_id} @ Step {step}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(viz_dir / f"recon_step_{step:06d}_seed{self.seed}.png", dpi=150, bbox_inches="tight")
            plt.close()

        self.model.train()
        print(f"[Viz] Saved visualizations at step {step}")

    def _visualize(self, step: int) -> None:
        """保存可视化：teacher/student embedding 对比 + reconstruction 对比."""
        if self.rank != 0:
            return

        self.model.eval()
        # 使用缓存的可视化 batch，避免反复实例化 dataloader
        batch = self._viz_batch
        if batch is None:
            batch = next(iter(self.val_dataloader))
        source_data = {k: v.to(self.device) for k, v in batch["source_data"].items()}
        timestamps = {k: v.to(self.device) for k, v in batch["timestamps"].items()}
        valid_periods = batch["valid_periods"]

        out = self.model(source_data, timestamps, valid_periods)

        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Embedding 对比 ----
        emb_t = out["teacher_embeddings"][0].cpu().numpy()  # (H, W, 64)
        emb_s = out["student_embeddings"][0].cpu().numpy()  # (H, W, 64)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        dims = np.random.choice(64, 4, replace=False)
        for i, d in enumerate(dims):
            vmin = min(emb_t[..., d].min(), emb_s[..., d].min())
            vmax = max(emb_t[..., d].max(), emb_s[..., d].max())
            axes[0, i].imshow(emb_t[..., d], vmin=vmin, vmax=vmax, cmap="viridis")
            axes[0, i].set_title(f"Teacher dim {d}")
            axes[0, i].axis("off")
            axes[1, i].imshow(emb_s[..., d], vmin=vmin, vmax=vmax, cmap="viridis")
            axes[1, i].set_title(f"Student dim {d}")
            axes[1, i].axis("off")
        plt.suptitle(f"Embedding Comparison @ Step {step}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(viz_dir / f"embedding_step_{step:06d}_seed{self.seed}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ---- 2. Reconstruction 对比 ----
        predictions = {src: rec[:, 0] for src, rec in out["reconstructions"].items()}
        targets = self._prepare_reconstruction_targets(batch, predictions)

        # 选最多 3 个源展示
        src_list = list(predictions.keys())[:3]
        if src_list:
            fig, axes = plt.subplots(2, len(src_list), figsize=(5 * len(src_list), 8))
            if len(src_list) == 1:
                axes = axes.reshape(2, 1)
            for i, src in enumerate(src_list):
                pred = predictions[src][0].cpu().numpy()
                tgt = targets[src][0].cpu().numpy()
                # 归一化到 [0,1] 用于显示
                pred_rgb = pred[..., :3]
                tgt_rgb = tgt[..., :3]
                pred_min, pred_max = pred_rgb.min(), pred_rgb.max()
                tgt_min, tgt_max = tgt_rgb.min(), tgt_rgb.max()
                pred_rgb = np.clip((pred_rgb - pred_min) / (pred_max - pred_min + 1e-8), 0, 1)
                tgt_rgb = np.clip((tgt_rgb - tgt_min) / (tgt_max - tgt_min + 1e-8), 0, 1)
                axes[0, i].imshow(tgt_rgb)
                axes[0, i].set_title(f"{src} Target")
                axes[0, i].axis("off")
                axes[1, i].imshow(pred_rgb)
                axes[1, i].set_title(f"{src} Pred")
                axes[1, i].axis("off")
            plt.suptitle(f"Reconstruction @ Step {step}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(viz_dir / f"recon_step_{step:06d}_seed{self.seed}.png", dpi=150, bbox_inches="tight")
            plt.close()

        self.model.train()
        print(f"[Viz] Saved visualizations at step {step}")
