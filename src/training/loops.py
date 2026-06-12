"""训练辅助循环函数.

为 trainer.py 提供 compute_recon_loss 等辅助函数，
直接复用 losses.py 中的 reconstruction_loss。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_recon_loss(
    predictions: torch.Tensor | list[torch.Tensor],
    targets: torch.Tensor | list[torch.Tensor],
    target_mask: torch.Tensor,
    target_loss_type: torch.Tensor | None = None,
    num_classes: int = 11,
    recon_mask: torch.Tensor | None = None,
    source_recon_weights: list[float] | None = None,
) -> torch.Tensor:
    """计算重建损失，支持连续 (L1) 和分类 (CE) 两种目标.

    Args:
        predictions: [B, T_tgt, C, H, W] 或 per-source list
        targets:     [B, T_tgt, C, H, W] 或 per-source list
        target_mask: [B, T_tgt] bool，True 表示该目标有效
        target_loss_type: [B, T_tgt] int，0=L1，1=CE
        num_classes: 分类目标的类别数
        recon_mask:  [B, H, W] 空间掩码（MAE 风格，1=计算损失）
        source_recon_weights: per-source 重建权重

    Returns:
        scalar 损失值
    """
    is_list = isinstance(predictions, list)
    if is_list:
        assert isinstance(targets, list)
        T = len(predictions)
        preds_list = predictions
        tgts_list = targets
        device = predictions[0].device
    else:
        if predictions is None or targets is None:
            return predictions.new_tensor(0.0) if predictions is not None else torch.tensor(0.0)
        B, T, C, H, W = predictions.shape
        preds_list = [predictions[:, t_idx] for t_idx in range(T)]
        tgts_list = [targets[:, t_idx] for t_idx in range(T)]
        device = predictions.device

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for t_idx in range(T):
        pred_t = preds_list[t_idx]
        tgt_t = tgts_list[t_idx]
        B = pred_t.shape[0]
        valid_b = target_mask[:, t_idx].bool() if target_mask is not None else torch.ones(B, dtype=torch.bool, device=device)
        if not valid_b.any():
            total_loss = total_loss + pred_t.sum() * 0.0
            continue

        pred_t = pred_t[valid_b]
        tgt_t = tgt_t[valid_b]
        pixel_valid = (~torch.isnan(tgt_t)).float()

        loss_type = 0
        if target_loss_type is not None:
            lt = target_loss_type[valid_b, t_idx]
            loss_type = int(lt[0].item()) if lt.numel() > 0 else 0

        if loss_type == 1:
            tgt_labels = tgt_t.argmax(dim=1).long()
            loss_t = F.cross_entropy(pred_t[:, :num_classes], tgt_labels, reduction='mean')
        else:
            mask = pixel_valid
            if recon_mask is not None:
                rm = recon_mask[valid_b]
                mask = mask * rm[:, None, :, :]
            diff = torch.abs(pred_t - tgt_t) * mask
            denom = mask.sum().clamp(min=1.0)
            loss_t = diff.sum() / denom

        weight = source_recon_weights[t_idx] if source_recon_weights is not None else 1.0
        total_loss = total_loss + loss_t * weight
        n_valid += 1

    return total_loss / max(n_valid, 1)
