"""像素级重建损失."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(
    reconstructions: dict[str, torch.Tensor],
    original_batch: dict[str, torch.Tensor | None],
    mask_info: dict,
) -> torch.Tensor:
    """
    计算像素级L1重建损失。

    Args:
        reconstructions: {source_name: [B, T, C, H, W]}
        original_batch: {source_name: [B, T, C, H, W] 或 None}
        mask_info: mask元信息

    Returns:
        loss: scalar tensor
    """
    from haidian_recon.data.masking import FourLayerMask

    total_loss = 0.0
    total_pixels = 0
    device_ref = None

    for source_name in mask_info["decode_sources"]:
        if source_name not in reconstructions:
            continue

        pred = reconstructions[source_name]
        target = original_batch.get(source_name)
        if target is None:
            continue

        device_ref = pred.device
        role = mask_info["modality_roles"][source_name]

        if role == "DECODE_ONLY":
            # 整个图像都是目标
            loss = F.l1_loss(pred, target, reduction="sum")
            n_pixels = target.numel()
        else:  # ENCODE_AND_DECODE
            # 仅计算被mask区域
            decode_mask = FourLayerMask.compute_decode_mask(mask_info, source_name)
            # decode_mask: [B, T, C, H, W], 1=需要重建的位置
            loss = F.l1_loss(pred * decode_mask, target * decode_mask, reduction="sum")
            n_pixels = decode_mask.sum().item()

        total_loss += loss
        total_pixels += n_pixels

    if total_pixels > 0:
        return total_loss / total_pixels
    elif device_ref is not None:
        return torch.tensor(0.0, device=device_ref)
    else:
        # 全部decode_sources都没有reconstruction，返回0
        for v in original_batch.values():
            if v is not None:
                return v.new_tensor(0.0)
        return torch.tensor(0.0)
