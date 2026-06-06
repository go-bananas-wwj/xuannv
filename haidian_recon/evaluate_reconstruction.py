"""重建效果评估 — PSNR/SSIM/可视化."""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch_npu
from torch.utils.data import DataLoader

from haidian_recon.config import Config
from haidian_recon.data.dataset import HaidianReconDataset, collate_fn
from haidian_recon.data.masking import FourLayerMask
from haidian_recon.models.hre_model import HREModel


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 6.0) -> float:
    """计算PSNR."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """简化SSIM计算."""
    mu1 = pred.mean()
    mu2 = target.mean()
    sigma1 = pred.std()
    sigma2 = target.std()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
    return float(ssim)


def evaluate_reconstruction(checkpoint_path: str, config_path: str, num_samples: int = 50) -> dict:
    """评估重建效果."""
    device = torch.device("npu:0")

    # 加载配置
    cfg = Config()
    source_channels = {s["name"]: s["channels"] for s in cfg.data.sources}

    # 加载模型
    model = HREModel(
        source_channels=source_channels,
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        use_gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 数据集
    val_dataset = HaidianReconDataset(
        data_root=cfg.data.data_root,
        planet_root=cfg.data.planet_root,
        stats_dir=cfg.data.stats_dir,
        split="val",
        image_size=cfg.data.image_size,
        source_names=list(source_channels.keys()),
        cache_dir=cfg.data.cache_dir,
    )
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)

    # Masking（评估时不使用困难mask，只做简单的block mask）
    masking = FourLayerMask(
        source_names=list(source_channels.keys()),
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        modality_probs=[0.0, 0.0, 0.0, 1.0],  # 全部ENCODE_AND_DECODE
        temporal_keep_ratio=1.0,  # 不mask时间步
        spatial_visible_ratio=0.25,  # 75% block mask
        channel_keep_ratio=1.0,  # 不mask通道
    ).to(device)

    # 评估
    results = {name: {"psnr": [], "ssim": []} for name in source_channels.keys()}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if total_samples >= num_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            masked_batch, mask_info = masking(batch)
            output = model(masked_batch, mask_info)

            for source_name in source_channels.keys():
                if source_name not in output["reconstructions"]:
                    continue

                pred = output["reconstructions"][source_name].cpu().numpy()
                target = batch[source_name].cpu().numpy()

                # 计算每个batch element的PSNR/SSIM（跳过无效样本）
                valid_mask = batch.get("valid_mask")
                B, T, C, H, W = pred.shape
                for b in range(B):
                    if valid_mask is not None and not valid_mask[b].item():
                        continue
                    for t in range(T):
                        for c in range(C):
                            p = pred[b, t, c]
                            tgt = target[b, t, c]
                            psnr = compute_psnr(p, tgt)
                            ssim = compute_ssim(p, tgt)
                            results[source_name]["psnr"].append(psnr)
                            results[source_name]["ssim"].append(ssim)

            total_samples += B
            if batch_idx % 10 == 0:
                print(f"Evaluated {total_samples}/{num_samples} samples")

    # 汇总
    summary = {}
    for source_name, metrics in results.items():
        if len(metrics["psnr"]) > 0:
            summary[source_name] = {
                "psnr_mean": float(np.mean(metrics["psnr"])),
                "psnr_std": float(np.std(metrics["psnr"])),
                "ssim_mean": float(np.mean(metrics["ssim"])),
                "ssim_std": float(np.std(metrics["ssim"])),
                "n_samples": len(metrics["psnr"]),
            }
        else:
            summary[source_name] = {"psnr_mean": 0.0, "ssim_mean": 0.0, "n_samples": 0}

    return summary


def visualize_reconstruction(checkpoint_path: str, config_path: str, output_dir: str, num_viz: int = 5) -> None:
    """可视化重建结果."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("npu:0")

    cfg = Config()
    source_channels = {s["name"]: s["channels"] for s in cfg.data.sources}

    model = HREModel(
        source_channels=source_channels,
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        use_gradient_checkpointing=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()

    val_dataset = HaidianReconDataset(
        data_root=cfg.data.data_root,
        planet_root=cfg.data.planet_root,
        stats_dir=cfg.data.stats_dir,
        split="val",
        image_size=cfg.data.image_size,
        source_names=list(source_channels.keys()),
        cache_dir=cfg.data.cache_dir,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=True)

    masking = FourLayerMask(
        source_names=list(source_channels.keys()),
        image_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        modality_probs=[0.0, 0.0, 0.0, 1.0],
        temporal_keep_ratio=1.0,
        spatial_visible_ratio=0.25,
        channel_keep_ratio=1.0,
    ).to(device)

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= num_viz:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None and not valid_mask[0].item():
                continue

            masked_batch, mask_info = masking(batch)
            output = model(masked_batch, mask_info)

            for source_name in source_channels.keys():
                if source_name not in output["reconstructions"]:
                    continue

                pred = output["reconstructions"][source_name][0, 0].cpu().numpy()
                target = batch[source_name][0, 0].cpu().numpy()

                # 可视化第一个通道
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(target[0], cmap="gray")
                axes[0].set_title(f"Original {source_name}")
                axes[0].axis("off")

                axes[1].imshow(pred[0], cmap="gray")
                axes[1].set_title(f"Reconstructed {source_name}")
                axes[1].axis("off")

                diff = np.abs(pred[0] - target[0])
                im = axes[2].imshow(diff, cmap="hot")
                axes[2].set_title(f"Absolute Diff (max={diff.max():.3f})")
                axes[2].axis("off")
                plt.colorbar(im, ax=axes[2])

                plt.tight_layout()
                plt.savefig(f"{output_dir}/viz_{count:03d}_{source_name}.png", dpi=150)
                plt.close()

            count += 1
            print(f"Saved visualization {count}/{num_viz}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="haidian_recon/config.yaml")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-num", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/hre_eval")
    args = parser.parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    summary = evaluate_reconstruction(args.checkpoint, args.config, args.num_samples)

    print("\n=== Reconstruction Evaluation Results ===")
    for source_name, metrics in summary.items():
        print(f"\n{source_name}:")
        print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"  Samples: {metrics['n_samples']}")

    if args.viz:
        print(f"\nGenerating visualizations to {args.output_dir}...")
        visualize_reconstruction(args.checkpoint, args.config, args.output_dir, args.viz_num)


if __name__ == "__main__":
    main()
