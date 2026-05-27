"""从已有 checkpoint 重新计算"真实" active_dims（使用正确的阈值）.

用法:
    python scripts/eval/compute_real_active_dims.py \
        --config configs/round7_8gpu/exp1_pre_norm_strong.yaml \
        --checkpoint /workspace/outputs/round7_exp1_pre_norm_strong/epoch_best_*.pt \
        --n-batches 50 \
        --threshold 0.15

说明:
    - 加载 checkpoint，跑 n-batches 个 forward
    - 收集 pre_norm_embedding，计算每维 std
    - 用指定阈值计算 active_dims
    - 支持 pre-norm (0.15) 和 L2 (0.05) 两种阈值
"""
from __future__ import annotations

import argparse
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.builder import build_dataloader
from src.models.model import AEFModel
from src.utils.device import get_device
from src.utils.checkpoint import load_checkpoint


def compute_active_dims(
    config_path: str,
    checkpoint_path: str,
    n_batches: int = 50,
    threshold: float = 0.15,
    device_str: str = "cpu",  # 默认 CPU，避免抢占训练 NPU
) -> dict:
    """计算指定 checkpoint 的 active_dims."""
    cfg = load_config(config_path)
    cfg.data.preload = False  # 避免长时间预加载
    device = torch.device(device_str)

    # 构建模型
    model = AEFModel(cfg).to(device)

    # 加载 checkpoint
    state = load_checkpoint(checkpoint_path, device=device_str)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # 构建 dataloader（只用 1 个 worker，小 batch）
    from src.data.dataset import HarbinPatchDataset
    dataset = HarbinPatchDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    all_pre_norm = []
    all_spatial = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            # 移动到设备
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Forward（单窗口）
            outputs = model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
            )

            # 收集 pre-norm
            pre_norm_embedding = outputs.pre_norm_embedding  # [B, D]
            pre_norm_map = outputs.pre_norm_map  # [B, D, H, W]

            all_pre_norm.append(pre_norm_embedding.cpu())
            all_spatial.append(pre_norm_map.cpu())

    # 合并
    all_pre = torch.cat(all_pre_norm, dim=0)  # [N, D]
    all_spatial_map = torch.cat(all_spatial, dim=0)  # [N, D, H, W]

    # Inter-level: 样本间 std
    inter_std = torch.sqrt(all_pre.var(dim=0, unbiased=False) + 1e-6)  # [D]
    inter_active = (inter_std > threshold).sum().item()
    inter_std_mean = inter_std.mean().item()
    inter_std_min = inter_std.min().item()
    inter_std_max = inter_std.max().item()

    # Spatial-level: 空间 std
    spatial_flat = all_spatial_map.permute(0, 2, 3, 1).reshape(-1, all_spatial_map.shape[1])  # [N*H*W, D]
    spatial_std = torch.sqrt(spatial_flat.var(dim=0, unbiased=False) + 1e-6)  # [D]
    spatial_active = (spatial_std > threshold).sum().item()
    spatial_std_mean = spatial_std.mean().item()
    spatial_std_min = spatial_std.min().item()
    spatial_std_max = spatial_std.max().item()

    # 同时计算旧阈值 (0.05) 作为对比
    inter_active_old = (inter_std > 0.05).sum().item()
    spatial_active_old = (spatial_std > 0.05).sum().item()

    return {
        "threshold": threshold,
        "threshold_old": 0.05,
        "n_samples": all_pre.shape[0],
        "embedding_dim": all_pre.shape[1],
        # Inter-level
        "inter_active": inter_active,
        "inter_active_old": inter_active_old,
        "inter_std_mean": inter_std_mean,
        "inter_std_min": inter_std_min,
        "inter_std_max": inter_std_max,
        # Spatial-level
        "spatial_active": spatial_active,
        "spatial_active_old": spatial_active_old,
        "spatial_std_mean": spatial_std_mean,
        "spatial_std_min": spatial_std_min,
        "spatial_std_max": spatial_std_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    results = compute_active_dims(
        args.config,
        args.checkpoint,
        n_batches=args.n_batches,
        threshold=args.threshold,
        device_str=args.device,
    )

    print("\n" + "=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold (new): {results['threshold']}")
    print(f"Threshold (old): {results['threshold_old']}")
    print(f"Samples: {results['n_samples']}, Dim: {results['embedding_dim']}")
    print("-" * 60)
    print(f"{'Metric':<25} {'New (≥'+str(results['threshold'])+')':<15} {'Old (≥0.05)':<15}")
    print("-" * 60)
    print(f"{'Inter active_dims':<25} {results['inter_active']:<15} {results['inter_active_old']:<15}")
    print(f"{'Spatial active_dims':<25} {results['spatial_active']:<15} {results['spatial_active_old']:<15}")
    print("-" * 60)
    print(f"Inter   std: min={results['inter_std_min']:.4f}, mean={results['inter_std_mean']:.4f}, max={results['inter_std_max']:.4f}")
    print(f"Spatial std: min={results['spatial_std_min']:.4f}, mean={results['spatial_std_mean']:.4f}, max={results['spatial_std_max']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
