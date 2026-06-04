"""训练辅助循环函数.

为 trainer.py 提供 compute_recon_loss 等辅助函数，
直接复用 losses.py 中的 reconstruction_loss。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_recon_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    target_loss_type: torch.Tensor | None = None,
    num_classes: int = 11,
    recon_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算重建损失，支持连续 (L1) 和分类 (CE) 两种目标.

    Args:
        predictions: [B, T_tgt, C, H, W] 预测输出
        targets:     [B, T_tgt, C, H, W] 目标值（分类目标为 one-hot 或类别索引）
        target_mask: [B, T_tgt] bool，True 表示该目标有效
        target_loss_type: [B, T_tgt] int，0=L1，1=CE
        num_classes: 分类目标的类别数
        recon_mask:  [B, H, W] 空间掩码（MAE 风格，1=计算损失）

    Returns:
        scalar 损失值
    """
    if predictions is None or targets is None:
        return predictions.new_tensor(0.0) if predictions is not None else torch.tensor(0.0)

    B, T, C, H, W = predictions.shape
    total_loss = torch.tensor(0.0, device=predictions.device)
    n_valid = 0

    pixel_valid = (~torch.isnan(targets)).float()

    for t_idx in range(T):
        # 有效 batch 样本
        valid_b = target_mask[:, t_idx].bool() if target_mask is not None else torch.ones(B, dtype=torch.bool)
        if not valid_b.any():
            # ★ DDP safety: dummy loss ensures decoder parameters always have gradients
            total_loss = total_loss + predictions[:, t_idx].sum() * 0.0
            continue

        pred_t = predictions[valid_b, t_idx]   # [Bv, C, H, W]
        tgt_t  = targets[valid_b, t_idx]        # [Bv, C, H, W]
        pv_t   = pixel_valid[valid_b, t_idx]    # [Bv, C, H, W]

        loss_type = 0
        if target_loss_type is not None:
            lt = target_loss_type[valid_b, t_idx]
            loss_type = int(lt[0].item()) if lt.numel() > 0 else 0

        if loss_type == 1:
            # 分类损失：CE，tgt 是 one-hot [Bv, num_classes, H, W]
            tgt_labels = tgt_t.argmax(dim=1).long()   # [Bv, H, W]
            loss_t = F.cross_entropy(pred_t[:, :num_classes], tgt_labels, reduction='mean')
        else:
            # 连续 L1 损失
            mask = pv_t
            if recon_mask is not None:
                rm = recon_mask[valid_b]  # [Bv, H, W]
                mask = mask * rm[:, None, :, :]
            diff = torch.abs(pred_t - tgt_t) * mask
            denom = mask.sum().clamp(min=1.0)
            loss_t = diff.sum() / denom

        total_loss = total_loss + loss_t
        n_valid += 1

    return total_loss / max(n_valid, 1)
