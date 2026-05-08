"""训练循环中的独立可复用函数."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: torch.Tensor | None,
    num_classes: int,
) -> torch.Tensor:
    """计算重建损失: L1 (连续源) / CE (分类源).

    Args:
        pred: [B, T_tgt, C, H, W] 模型重建输出.
        target: [B, T_tgt, C', H, W] 目标图像.
        mask: [B, T_tgt] bool — 只计算 mask=True 的目标.
        loss_type: [B, T_tgt] or [B*T_tgt] int — 0=continuous, 1=categorical.
        num_classes: 分类目标的总类别数.

    Returns:
        标量损失值.
    """
    total_loss = torch.tensor(0.0, device=pred.device)
    count = 0

    B = pred.shape[0]
    T_tgt = pred.shape[1]
    if loss_type is not None and loss_type.dim() == 1:
        loss_type = loss_type.reshape(B, T_tgt)

    for t_idx in range(T_tgt):
        batch_mask = mask[:, t_idx]  # [B] bool
        if not batch_mask.any():
            continue

        p = pred[:, t_idx]  # [B, C, H, W]
        tgt = target[:, t_idx]  # [B, C', H, W]

        is_categorical = False
        if loss_type is not None:
            lt = loss_type[:, t_idx][batch_mask]  # [N_valid]
            if lt.numel() > 0 and (lt == 1).all().item():
                is_categorical = True

        if is_categorical:
            # 从 one-hot 编码恢复类别索引
            tgt_onehot = tgt[batch_mask]  # [N_valid, C, H, W]
            tgt_cls = tgt_onehot.argmax(dim=1).long()  # [N_valid, H, W]
            # 排除 no-data（所有通道和接近 0）
            has_data = tgt_onehot.sum(dim=1) > 0.5
            valid_pixels = has_data & (tgt_cls >= 0) & (tgt_cls < num_classes)
            p_valid = p[batch_mask]  # [N_valid, C, H, W]
            if valid_pixels.sum() > 0:
                if p_valid.shape[-2:] != tgt_cls.shape[-2:]:
                    p_aligned = F.interpolate(p_valid, size=tgt_cls.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    p_aligned = p_valid
                ce = F.cross_entropy(p_aligned, tgt_cls, reduction="none")  # [N_valid, H, W]
                total_loss = total_loss + (ce * valid_pixels.float()).sum() / valid_pixels.float().sum().clamp(min=1)
                count += 1
        else:
            p_valid = p[batch_mask]
            tgt_valid = tgt[batch_mask]
            valid = ~torch.isnan(tgt_valid)
            if valid.sum() > 0:
                total_loss = total_loss + torch.abs(p_valid[valid] - tgt_valid[valid]).mean()
                count += 1

    return total_loss / max(count, 1)
