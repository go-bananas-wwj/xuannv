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
        pred_src = preds_list[t_idx]
        tgt_t = tgts_list[t_idx]
        B = pred_src.shape[0]
        if target_mask is not None:
            valid_b = target_mask[:, t_idx].bool().to(device)
        else:
            valid_b = torch.ones(B, dtype=torch.bool, device=device)
        if not valid_b.any():
            total_loss = total_loss + pred_src.sum() * 0.0
            continue

        tgt_t = tgt_t[valid_b]
        # 多分辨率模式下 target 可能被 pad 到统一通道数，按 prediction 通道截断
        pred_c = pred_src.shape[2] if pred_src.dim() == 5 else pred_src.shape[1]
        if tgt_t.shape[1] > pred_c:
            tgt_t = tgt_t[:, :pred_c]
        pixel_valid = (~torch.isnan(tgt_t)).float()

        loss_type = 0
        if target_loss_type is not None:
            lt = target_loss_type[valid_b, t_idx]
            loss_type = int(lt[0].item()) if lt.numel() > 0 else 0

        # 多分辨率下每个源可能预测多个时间步 (B, T_inner, C, H, W)，目标只有单帧，对 T_inner 求平均
        inner_steps = pred_src.shape[1] if pred_src.dim() == 5 else 1
        inner_loss = torch.tensor(0.0, device=device)
        for inner_idx in range(inner_steps):
            pred_t = pred_src[:, inner_idx][valid_b] if pred_src.dim() == 5 else pred_src[valid_b]

            if loss_type == 1:
                tgt_labels = tgt_t.argmax(dim=1).long()
                loss_t = F.cross_entropy(pred_t[:, :num_classes], tgt_labels, reduction='mean')
            else:
                mask = pixel_valid
                if recon_mask is not None:
                    rm = recon_mask[valid_b]
                    # 多分辨率模式下 recon_mask 可能与目标分辨率不一致，需要插值
                    if rm.shape[-2:] != tgt_t.shape[-2:]:
                        rm = torch.nn.functional.interpolate(
                            rm.unsqueeze(1).float(),
                            size=tgt_t.shape[-2:],
                            mode="nearest",
                        ).squeeze(1)
                    mask = mask * rm[:, None, :, :]
                diff = torch.abs(pred_t - tgt_t) * mask
                denom = mask.sum().clamp(min=1.0)
                loss_t = diff.sum() / denom
            inner_loss = inner_loss + loss_t

        loss_t = inner_loss / max(inner_steps, 1)
        weight = source_recon_weights[t_idx] if source_recon_weights is not None else 1.0
        total_loss = total_loss + loss_t * weight
        n_valid += 1

    return total_loss / max(n_valid, 1)
