"""AEF 训练器 — 适配 NPU + DDP."""
from __future__ import annotations

import itertools
import math
import os
import time
from pathlib import Path
from typing import Any

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
        save_every: int = 5000,
        eval_every: int = 5000,
        output_dir: str | None = None,
        rank: int = 0,
        world_size: int = 1,
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

        self.loss_history = {
            "steps": [],
            "total": [],
            "reconstruction": [],
            "uniformity": [],
            "consistency": [],
        }

        self.step = 0

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

            # 下采样到重建分辨率（AEF 输出是 1/2L）
            H2, W2 = pred[src_key].shape[2], pred[src_key].shape[3]
            target_2d = rearrange(target, "b h w c -> b c h w")
            target_2d = F.interpolate(target_2d, size=(H2, W2), mode="bilinear", align_corners=False)
            target = rearrange(target_2d, "b c h w -> b h w c")
            targets[src_key] = target
        return targets

    def train(self) -> None:
        self.model.train()
        data_iter = itertools.cycle(self.dataloader)

        pbar = tqdm(
            range(1, self.max_steps + 1),
            desc="Training",
            unit="step",
            disable=self.rank != 0,
        )
        start_time = time.time()

        for step in pbar:
            self.step = step
            batch = next(data_iter)

            source_data = {
                k: v.to(self.device) for k, v in batch["source_data"].items()
            }
            timestamps = {
                k: v.to(self.device) for k, v in batch["timestamps"].items()
            }
            valid_periods = batch["valid_periods"]

            out = self.model(source_data, timestamps, valid_periods)

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

            losses = self.loss_fn(outputs_for_loss)
            loss = losses["total"]

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optim.step()
            self.scheduler.step()

            # Sync losses across ranks
            if self.world_size > 1:
                for k in losses:
                    if losses[k].device != self.device:
                        losses[k] = losses[k].to(self.device)
                    dist.all_reduce(losses[k], op=dist.ReduceOp.AVG)

            # Logging
            if self.rank == 0:
                self.loss_history["steps"].append(step)
                for k in ["total", "reconstruction", "uniformity", "consistency"]:
                    self.loss_history[k].append(float(losses[k]))

                if step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    eta = (self.max_steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
                    print(
                        f"[Step {step}] "
                        f"loss={losses['total']:.4f} "
                        f"recon={losses['reconstruction']:.4f} "
                        f"uniform={losses['uniformity']:.4f} "
                        f"consist={losses['consistency']:.4f} "
                        f"lr={lr:.6f} "
                        f"({steps_per_sec:.2f} step/s, ETA {eta:.1f}h)"
                    )

                # Progress bar
                recon = float(losses["reconstruction"])
                pbar.set_postfix({
                    "recon": f"{recon:.4f}",
                    "total": f"{float(loss):.4f}",
                })

            # Save checkpoint
            if step % self.save_every == 0 and self.rank == 0:
                self._save_checkpoint(step)

            # Eval
            if step % self.eval_every == 0:
                self._eval()

        pbar.close()
        if self.rank == 0:
            print("\nTraining completed!")

    @torch.no_grad()
    def _eval(self) -> None:
        self.model.eval()
        total_recon = 0.0
        total_uniform = 0.0
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
            losses = self.loss_fn(outputs_for_loss)
            total_recon += float(losses["reconstruction"])
            total_uniform += float(losses["uniformity"])
            count += 1

        self.model.train()

        if count == 0:
            if self.rank == 0:
                print("[Val] No valid batches")
            return

        if self.world_size > 1:
            recon_tensor = torch.tensor(total_recon / count, device=self.device)
            uniform_tensor = torch.tensor(total_uniform / count, device=self.device)
            dist.all_reduce(recon_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(uniform_tensor, op=dist.ReduceOp.AVG)
            avg_recon = float(recon_tensor)
            avg_uniform = float(uniform_tensor)
        else:
            avg_recon = total_recon / count
            avg_uniform = total_uniform / count

        if self.rank == 0:
            print(f"[Val] recon_loss={avg_recon:.4f} uniform={avg_uniform:.4f}")

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
        torch.save(checkpoint, self.output_dir / f"step_{step:06d}.pt")
        if self.rank == 0:
            print(f"[Save] Checkpoint saved to {self.output_dir}/step_{step:06d}.pt")
