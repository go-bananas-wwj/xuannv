#!/usr/bin/env python3
"""V8 Clean 下游重建任务评估 — 多卡并行评估7个目标源的重建质量.

用法:
    cd /workspace/xuannv
    python scripts/eval/eval_reconstruction_v8.py \
        --config configs/xuannv_v8_clean.yaml \
        --checkpoint /workspace/outputs/xuannv_backbone_v8_clean/epoch_best_epoch223.pt \
        --devices npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7
"""
import sys, time, argparse, multiprocessing as mp
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# 目标源定义 (与 dataset.py 一致)
TARGET_SOURCES = ["s2", "s1", "landsat", "dem", "worldcover", "dynamic_world", "jrc_water"]
CATEGORICAL = {"worldcover", "dynamic_world", "jrc_water"}


def load_model(cfg_path, ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    cfg = load_config(cfg_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, cfg


def worker_process(device, cfg_path, ckpt_path, patch_ids, return_dict):
    """子进程：在指定 NPU 上评估一批 patch."""
    torch.npu.set_device(device)
    model, dataset, cfg = load_model(cfg_path, ckpt_path, device)

    # 统计每个目标源
    results = {src: {"mae": [], "mse": [], "acc": [], "count": 0} for src in TARGET_SOURCES}

    for pid in patch_ids:
        try:
            idx = dataset.patches.index(pid)
            item = dataset[idx]
        except ValueError:
            continue

        # 构建 batch
        batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

        with torch.no_grad():
            out = model(
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

        # reconstructions: [B, T_tgt, C, H, W]
        recon = out.reconstructions
        if recon is None:
            continue

        target_images = batch["target_images"]  # [B, T_tgt, C, H, W]
        target_mask = batch.get("target_mask")  # [B, T_tgt]
        target_loss_type = batch.get("target_loss_type")  # [B, T_tgt]

        B, T_tgt = target_images.shape[:2]

        for t_idx in range(T_tgt):
            if target_mask is not None and not target_mask[0, t_idx].item():
                continue

            src_name = TARGET_SOURCES[t_idx] if t_idx < len(TARGET_SOURCES) else None
            if src_name is None:
                continue

            pred = recon[0, t_idx]  # [C, H, W]
            tgt = target_images[0, t_idx]  # [C, H, W]

            # 处理 NaN
            valid_mask = ~torch.isnan(tgt)
            if valid_mask.sum() == 0:
                continue

            pred_valid = pred[valid_mask]
            tgt_valid = tgt[valid_mask]

            if src_name in CATEGORICAL:
                # 分类源：计算 accuracy
                if tgt.ndim == 3 and tgt.shape[0] > 1:
                    # one-hot target
                    tgt_class = tgt.argmax(dim=0)  # [H, W]
                else:
                    tgt_class = tgt[0].long() if tgt.ndim == 3 else tgt.long()

                pred_class = pred.argmax(dim=0)  # [H, W]
                valid_mask_2d = valid_mask[0] if valid_mask.ndim == 3 else valid_mask

                correct = (pred_class[valid_mask_2d] == tgt_class[valid_mask_2d]).float().sum().item()
                total = valid_mask_2d.sum().item()
                if total > 0:
                    acc = correct / total
                    results[src_name]["acc"].append(acc)
                    results[src_name]["count"] += 1
            else:
                # 连续源：MAE, MSE
                mae = (pred_valid - tgt_valid).abs().mean().item()
                mse = ((pred_valid - tgt_valid) ** 2).mean().item()
                results[src_name]["mae"].append(mae)
                results[src_name]["mse"].append(mse)
                results[src_name]["count"] += 1

    return_dict[device] = results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xuannv_v8_clean.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--devices", type=str, default="npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7")
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]

    print("=" * 70)
    print("  V8 Clean 下游重建任务评估 — 多卡并行")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Devices:    {devices}")
    print("=" * 70)

    # 加载 dataset 获取 patch 列表
    from src.config import load_config
    from src.data.dataset import HarbinPatchDataset
    cfg = load_config(args.config)
    cfg.data.preload = False
    ds = HarbinPatchDataset(cfg)
    all_patches = ds.patches[:]

    print(f"\n总 patch 数: {len(all_patches)}")

    # 多卡并行
    print(f"\n多卡并行评估 ({len(devices)} 卡)...")
    start = time.time()

    manager = mp.Manager()
    return_dict = manager.dict()

    n = len(all_patches)
    chunk_size = (n + len(devices) - 1) // len(devices)
    chunks = [all_patches[i:i + chunk_size] for i in range(0, n, chunk_size)]

    processes = []
    for device, chunk in zip(devices, chunks):
        p = mp.Process(target=worker_process, args=(device, args.config, args.checkpoint, chunk, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start
    print(f"评估完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 汇总结果
    print("\n" + "=" * 70)
    print("  重建质量汇总")
    print("=" * 70)

    # 合并各设备结果
    merged = {src: {"mae": [], "mse": [], "acc": [], "count": 0} for src in TARGET_SOURCES}
    for device in devices:
        if device not in return_dict:
            continue
        dev_results = return_dict[device]
        for src in TARGET_SOURCES:
            merged[src]["mae"].extend(dev_results[src]["mae"])
            merged[src]["mse"].extend(dev_results[src]["mse"])
            merged[src]["acc"].extend(dev_results[src]["acc"])
            merged[src]["count"] += dev_results[src]["count"]

    for src in TARGET_SOURCES:
        data = merged[src]
        if data["count"] == 0:
            print(f"\n  {src}: 无有效数据")
            continue

        if src in CATEGORICAL:
            mean_acc = np.mean(data["acc"])
            print(f"\n  {src}: count={data['count']} | Accuracy={mean_acc:.4f}")
        else:
            mean_mae = np.mean(data["mae"])
            mean_mse = np.mean(data["mse"])
            rmse = np.sqrt(mean_mse)
            print(f"\n  {src}: count={data['count']} | MAE={mean_mae:.4f} | RMSE={rmse:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
