#!/usr/bin/env python3
"""重建效果评估脚本 — PSNR / SSIM.

用法:
    python scripts/eval/evaluate_reconstruction.py \
        --config configs/config.yaml \
        --checkpoint /workspace/xuannv/outputs/exp/epoch_10.pt \
        --output /workspace/xuannv/outputs/exp/recon_epoch_10.json \
        --device npu:0 \
        --num-samples 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.data.multi_region_dataset import MultiRegionPatchDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="训练配置 YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--device", default="npu:0", help="设备")
    parser.add_argument("--num-samples", type=int, default=30, help="评估样本数")
    parser.add_argument("--months", type=str, default="4,5,6,7,8,9,10", help="评估月份列表（逗号分隔）")
    return parser.parse_args()


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float | None = None) -> float:
    """计算 PSNR."""
    if max_val is None:
        max_val = max(abs(pred.max()), abs(target.min()), abs(target.max()), abs(target.min()))
        max_val = max(max_val, 1e-6)
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """简化 SSIM 计算（单通道）."""
    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1 = pred.std()
    sigma2 = target.std()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denom = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2)
    if denom < 1e-10:
        return 1.0
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / denom
    return float(np.clip(ssim, -1.0, 1.0))


def month_to_window(year: int, month: int) -> tuple[int, int]:
    import calendar
    import time as time_mod
    start_s = int(time_mod.mktime((year, month, 1, 0, 0, 0, 0, 0, 0)))
    last_d = calendar.monthrange(year, month)[1]
    end_s = int(time_mod.mktime((year, month, last_d, 23, 59, 59, 0, 0, 0)))
    return start_s * 1000, end_s * 1000


def load_model_and_dataset(config_path: str, checkpoint_path: str, device: str):
    """加载模型和数据集."""
    cfg = load_config(config_path)
    cfg.data.preload = True
    cfg.data.num_workers = 0

    if getattr(cfg.data, "multi_region_manifest", None):
        dataset = MultiRegionPatchDataset(cfg=cfg)
    else:
        dataset = HarbinPatchDataset(cfg=cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    dev = torch.device(device)
    model = AEFModel(cfg).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, dataset, cfg, dev


@torch.no_grad()
def evaluate_reconstruction(model, dataset, cfg, device, months: list[int], num_samples: int = 30) -> dict:
    """评估重建效果.

    Args:
        months: 要评估的月份列表（1-12）
        num_samples: 采样的 patch 数量

    Returns:
        dict: {source_name: {"psnr_mean": ..., "psnr_std": ..., "ssim_mean": ..., "ssim_std": ..., "n_samples": ...}}
    """
    target_sources = cfg.data.target_sources
    source_names = [t["name"] if isinstance(t, dict) else t for t in target_sources]
    loss_types = [t.get("loss_type", 0) if isinstance(t, dict) else 0 for t in target_sources]

    # 收集所有可用的 (patch_id, year, month) 样本
    # monthly_samples 格式: list[(patch_id, year, month), ...]
    samples = []
    for sample_idx, (pid, year, month) in enumerate(dataset.monthly_samples):
        if month in months:
            samples.append((sample_idx, pid, year, month))

    # 随机采样
    rng = np.random.RandomState(42)
    if len(samples) > num_samples:
        indices = rng.choice(len(samples), size=num_samples, replace=False)
        samples = [samples[i] for i in indices]

    # 结果收集
    results: dict[str, list] = {name: {"psnr": [], "ssim": []} for name in source_names}

    t0 = time.time()
    for idx, (sample_idx, pid, year, month) in enumerate(samples):
        item = dataset[sample_idx]
        vs, ve = month_to_window(year, month)

        def _to(x):
            return x.unsqueeze(0).to(device)

        use_bf16 = getattr(cfg.training, "use_bf16", True)
        with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(
                source_frames=_to(item["source_frames"]),
                source_timestamps_ms=_to(item["source_timestamps_ms"]),
                source_frame_mask=_to(item["source_frame_mask"]),
                source_input_mask=_to(item["source_input_mask"]),
                source_type_ids=_to(item["source_type_ids"]),
                valid_start_ms=torch.tensor([vs], dtype=torch.int64, device=device),
                valid_end_ms=torch.tensor([ve], dtype=torch.int64, device=device),
                target_relative_time=_to(item["target_relative_time"]),
                target_metadata=_to(item["target_metadata"]),
                skip_decoder=False,
            )

        recon = out.reconstructions.float()  # [1, T_tgt, C, H, W]
        target = item["target_images"].to(device).float()  # [T_tgt, C, H, W]
        target_mask = item["target_mask"].to(device).bool()  # [T_tgt]
        target_loss_type = item["target_loss_type"].to(device).long()  # [T_tgt]

        T = recon.shape[1]
        for t_idx in range(T):
            if not target_mask[t_idx].item():
                continue

            src_name = source_names[t_idx]
            loss_type = loss_types[t_idx]

            pred_t = recon[0, t_idx].cpu().numpy()  # [C, H, W]
            tgt_t = target[t_idx].cpu().numpy()  # [C, H, W]

            if loss_type == 1:
                # 分类目标: pred 是 logits [num_classes, H, W]，取 argmax
                pred_t = pred_t.argmax(axis=0)  # [H, W]
                tgt_t = tgt_t.argmax(axis=0)  # [H, W]
                # 转为 float 计算 PSNR/SSIM（类别索引作为强度值）
                pred_t = pred_t.astype(np.float32)
                tgt_t = tgt_t.astype(np.float32)
                psnr = compute_psnr(pred_t, tgt_t, max_val=max(pred_t.max(), tgt_t.max(), 1.0))
                ssim = compute_ssim(pred_t, tgt_t)
            else:
                # 连续目标: 逐通道计算后取平均
                psnr_list = []
                ssim_list = []
                C = pred_t.shape[0]
                for c in range(C):
                    p = pred_t[c]
                    t = tgt_t[c]
                    # 跳过全 NaN 通道
                    if np.isnan(p).all() or np.isnan(t).all():
                        continue
                    # mask NaN
                    valid = ~(np.isnan(p) | np.isnan(t))
                    if valid.sum() < 10:
                        continue
                    p_valid = p[valid]
                    t_valid = t[valid]
                    psnr_list.append(compute_psnr(p_valid, t_valid))
                    ssim_list.append(compute_ssim(p_valid, t_valid))
                if len(psnr_list) == 0:
                    continue
                psnr = float(np.mean(psnr_list))
                ssim = float(np.mean(ssim_list))

            results[src_name]["psnr"].append(psnr)
            results[src_name]["ssim"].append(ssim)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"  [ReconEval] {idx + 1}/{len(samples)} samples done ({time.time()-t0:.1f}s)")

    # 汇总
    summary = {}
    for src_name, metrics in results.items():
        if len(metrics["psnr"]) > 0:
            summary[src_name] = {
                "psnr_mean": float(np.mean(metrics["psnr"])),
                "psnr_std": float(np.std(metrics["psnr"])),
                "ssim_mean": float(np.mean(metrics["ssim"])),
                "ssim_std": float(np.std(metrics["ssim"])),
                "n_samples": len(metrics["psnr"]),
            }
        else:
            summary[src_name] = {
                "psnr_mean": 0.0,
                "psnr_std": 0.0,
                "ssim_mean": 0.0,
                "ssim_std": 0.0,
                "n_samples": 0,
            }

    return summary


def main():
    args = parse_args()
    months = [int(m.strip()) for m in args.months.split(",")]

    print(f"[ReconEval] Loading model from {args.checkpoint}")
    model, dataset, cfg, device = load_model_and_dataset(args.config, args.checkpoint, args.device)

    print(f"[ReconEval] Evaluating {args.num_samples} samples for months {months}")
    summary = evaluate_reconstruction(model, dataset, cfg, device, months, args.num_samples)

    print("\n=== Reconstruction Evaluation Results ===")
    for src_name, metrics in summary.items():
        print(
            f"  {src_name}: PSNR={metrics['psnr_mean']:.2f}±{metrics['psnr_std']:.2f} "
            f"SSIM={metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.4f} "
            f"(n={metrics['n_samples']})"
        )

    # 保存 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ReconEval] Results saved to {output_path}")


if __name__ == "__main__":
    main()
