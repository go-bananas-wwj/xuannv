"""四层嵌套Mask策略 — 让重建变得极端困难."""
from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class FourLayerMask(nn.Module):
    """
    四层嵌套Mask策略：
    Layer 1: 模态级角色分配（随机dropout整个模态）
    Layer 2: 时间步mask（随机mask 50%时间步）
    Layer 3: Patch级block mask（75% mask ratio）
    Layer 4: 通道级mask（随机mask 50%通道）
    """

    def __init__(
        self,
        source_names: list[str],
        image_size: int = 128,
        patch_size: int = 8,
        modality_probs: list[float] | None = None,
        temporal_keep_ratio: float = 0.5,
        spatial_visible_ratio: float = 0.25,
        channel_keep_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.source_names = source_names
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches_h = image_size // patch_size
        self.n_patches_w = image_size // patch_size
        self.total_patches = self.n_patches_h * self.n_patches_w

        self.modality_probs = modality_probs or [0.1, 0.25, 0.25, 0.4]
        self.roles = ["NOT_SELECTED", "ENCODE_ONLY", "DECODE_ONLY", "ENCODE_AND_DECODE"]
        self.temporal_keep_ratio = temporal_keep_ratio
        self.spatial_visible_ratio = spatial_visible_ratio
        self.channel_keep_ratio = channel_keep_ratio

    def forward(
        self,
        batch: dict[str, torch.Tensor | None],
    ) -> tuple[dict[str, torch.Tensor | None], dict]:
        """
        对batch应用四层嵌套mask。

        Args:
            batch: {source_name: [B, T, C, H, W] tensor 或 None}

        Returns:
            masked_batch: mask后的输入（None表示该模态不输入encoder）
            mask_info: mask元信息
        """
        # 获取batch shape
        first_valid = None
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() == 5:
                first_valid = v
                break
        if first_valid is None:
            raise ValueError("All sources are None in batch")

        B, T = first_valid.shape[:2]
        device = first_valid.device

        mask_info = {
            "modality_roles": {},
            "decode_sources": [],
            "encode_sources": [],
            "temporal_mask": {},
            "spatial_mask": {},
            "channel_mask": {},
            "original_batch": {},
            "_patch_size": self.patch_size,
        }

        # Layer 1: 模态级角色分配
        for source_name in self.source_names:
            x = batch.get(source_name)
            mask_info["original_batch"][source_name] = x

            # 若数据为None（如Planet缺失），强制设为NOT_SELECTED
            if x is None:
                mask_info["modality_roles"][source_name] = "NOT_SELECTED"
                continue

            # 检查valid_mask（来自collate_fn），若全部无效则设为NOT_SELECTED
            valid_key = f"{source_name}_valid"
            if valid_key in batch:
                valid_mask = batch[valid_key]
                if not valid_mask.any():
                    mask_info["modality_roles"][source_name] = "NOT_SELECTED"
                    continue

            # 使用 torch 随机流替代 np.random，避免 CPU↔NPU 同步 + DDP 随机状态一致
            probs = torch.tensor(self.modality_probs, device=device)
            role_idx = torch.multinomial(probs, 1).item()
            role = self.roles[role_idx]
            mask_info["modality_roles"][source_name] = role

        # 约束检查：确保至少1个encode源和1个decode源
        encode_sources = [
            s for s in self.source_names
            if mask_info["modality_roles"].get(s) in ("ENCODE_ONLY", "ENCODE_AND_DECODE")
        ]
        decode_sources = [
            s for s in self.source_names
            if mask_info["modality_roles"].get(s) in ("DECODE_ONLY", "ENCODE_AND_DECODE")
        ]

        # 若缺少encode源，强制选一个可用模态为ENCODE_AND_DECODE
        if len(encode_sources) == 0:
            available = [s for s in self.source_names if batch.get(s) is not None]
            if available:
                idx = torch.randint(0, len(available), (1,), device=device).item()
                forced = available[idx]
                mask_info["modality_roles"][forced] = "ENCODE_AND_DECODE"
                encode_sources.append(forced)
                if forced not in decode_sources:
                    decode_sources.append(forced)

        # 若缺少decode源，强制选一个可用模态为DECODE_ONLY
        if len(decode_sources) == 0:
            available = [s for s in self.source_names if batch.get(s) is not None]
            if available:
                idx = torch.randint(0, len(available), (1,), device=device).item()
                forced = available[idx]
                if forced in encode_sources:
                    mask_info["modality_roles"][forced] = "ENCODE_AND_DECODE"
                else:
                    mask_info["modality_roles"][forced] = "DECODE_ONLY"
                decode_sources.append(forced)

        mask_info["encode_sources"] = [
            s for s in self.source_names
            if mask_info["modality_roles"].get(s) in ("ENCODE_ONLY", "ENCODE_AND_DECODE")
        ]
        mask_info["decode_sources"] = [
            s for s in self.source_names
            if mask_info["modality_roles"].get(s) in ("DECODE_ONLY", "ENCODE_AND_DECODE")
        ]

        # Layer 2-4: 对encode源应用时间/空间/通道mask
        masked_batch = {}
        for source_name in self.source_names:
            role = mask_info["modality_roles"][source_name]
            x = batch.get(source_name)

            if role == "NOT_SELECTED" or x is None:
                masked_batch[source_name] = None
                continue

            if role == "DECODE_ONLY":
                masked_batch[source_name] = None
                continue

            # role为ENCODE_ONLY或ENCODE_AND_DECODE
            x_masked = x.clone()
            _, T_src, C, H, W = x_masked.shape

            # 对batch中无效sample（该源缺失）的数据全部置零，避免污染encoder
            valid_key = f"{source_name}_valid"
            if valid_key in batch:
                valid_mask = batch[valid_key]  # [B]
                if not valid_mask.all():
                    x_masked[~valid_mask] = 0.0

            # Layer 2: 时间步mask（向量化）
            n_keep = max(1, int(T_src * self.temporal_keep_ratio))
            keep_indices = torch.randperm(T_src, device=device)[:n_keep]
            temporal_mask = torch.ones(T_src, dtype=torch.bool, device=device)
            temporal_mask[keep_indices] = False
            mask_info["temporal_mask"][source_name] = temporal_mask

            # 被mask的时间步置零 [1, T, 1, 1, 1]
            temporal_exp = temporal_mask[None, :, None, None, None].expand_as(x_masked)
            x_masked = x_masked * (~temporal_exp).float()

            # Layer 3: Patch级block mask
            n_visible = max(1, int(self.total_patches * self.spatial_visible_ratio))
            visible_indices = torch.randperm(self.total_patches, device=device)[:n_visible]
            spatial_mask_flat = torch.zeros(self.total_patches, dtype=torch.float32, device=device)
            spatial_mask_flat[visible_indices] = 1.0
            spatial_mask = spatial_mask_flat.reshape(self.n_patches_h, self.n_patches_w)
            # 上采样到image_size
            spatial_mask_full = spatial_mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(
                self.patch_size, dim=1
            )  # [H, W]
            mask_info["spatial_mask"][source_name] = spatial_mask

            # 只对可见时间步应用空间mask
            visible = ~temporal_mask
            if visible.any():
                spatial_exp = spatial_mask_full[None, None, None, :, :]
                x_masked[:, visible] = x_masked[:, visible] * spatial_exp

            # Layer 4: 通道级mask（仅在可见时间步上应用）
            n_keep_ch = max(1, int(C * self.channel_keep_ratio))
            keep_channels = torch.randperm(C, device=device)[:n_keep_ch]
            channel_mask = torch.zeros(C, dtype=torch.float32, device=device)
            channel_mask[keep_channels] = 1.0
            mask_info["channel_mask"][source_name] = channel_mask

            if visible.any():
                channel_exp = channel_mask[None, :, None, None]
                x_masked[:, visible] = x_masked[:, visible] * channel_exp

            masked_batch[source_name] = x_masked

        return masked_batch, mask_info

    @staticmethod
    def compute_decode_mask(mask_info: dict, source_name: str) -> torch.Tensor:
        """
        计算某个decode_source需要重建的目标mask。
        对于DECODE_ONLY：返回全1
        对于ENCODE_AND_DECODE：返回被Layer 2-4 mask掉的位置
        """
        role = mask_info["modality_roles"][source_name]
        original = mask_info["original_batch"][source_name]
        if original is None:
            # fallback: 尝试构造一个全零mask
            for v in mask_info["original_batch"].values():
                if v is not None:
                    B, T, C, H, W = v.shape
                    device = v.device
                    return torch.zeros(B, T, C, H, W, dtype=torch.float32, device=device)
            raise ValueError(f"Cannot determine decode mask for {source_name}")

        B, T, C, H, W = original.shape
        device = original.device

        if role == "DECODE_ONLY":
            return torch.ones(B, T, C, H, W, dtype=torch.float32, device=device)

        # ENCODE_AND_DECODE: 被mask掉的位置是目标
        temporal_mask = mask_info["temporal_mask"][source_name]  # [T], True=被mask
        spatial_mask = mask_info["spatial_mask"][source_name]  # [grid_h, grid_w]
        channel_mask = mask_info["channel_mask"][source_name]  # [C]
        patch_size = mask_info.get("_patch_size", 8)

        # 构建组合mask: 被mask的位置为1（需要重建），可见位置为0
        decode_mask = torch.zeros(B, T, C, H, W, dtype=torch.float32, device=device)

        # 上采样spatial mask
        spatial_full = spatial_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(
            patch_size, dim=1
        )  # [H, W]

        # 向量化计算
        temporal_exp = temporal_mask[None, :, None, None, None].expand(B, T, C, H, W)
        spatial_masked = (spatial_full < 0.5)[None, None, None, :, :].expand(B, T, C, H, W)
        channel_masked = (channel_mask < 0.5)[None, None, :, None, None].expand(B, T, C, H, W)

        decode_mask = (temporal_exp | spatial_masked | channel_masked).float()
        return decode_mask
