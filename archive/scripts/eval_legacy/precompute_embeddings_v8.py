#!/usr/bin/env python3
"""预提取所有 patch 的 before/after embedding map — 8 卡并行，保存到磁盘.

用法:
    cd /workspace/xuannv
    python scripts/eval/precompute_embeddings_v8.py \
        --config configs/xuannv_v8_clean.yaml \
        --checkpoint /workspace/outputs/xuannv_backbone_v8_clean/epoch_best_epoch223.pt \
        --output /workspace/outputs/xuannv_backbone_v8_clean/precomputed_embeddings.pt
"""
import sys, time, argparse, multiprocessing as mp
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import warnings
warnings.filterwarnings('ignore')

BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)


def load_model(cfg_path, ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.inference.engine import extract_embedding_map

    cfg = load_config(cfg_path)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset, extract_embedding_map


def worker(device, cfg_path, ckpt_path, patch_ids, return_dict):
    torch.npu.set_device(device)
    model, dataset, extract_fn = load_model(cfg_path, ckpt_path, device)
    results = {}
    for pid in patch_ids:
        try:
            idx = dataset.patches.index(pid)
            eb = extract_fn(model, dataset, idx, BEFORE_WINDOW[0], BEFORE_WINDOW[1], device, normalize=True)
            ea = extract_fn(model, dataset, idx, AFTER_WINDOW[0], AFTER_WINDOW[1], device, normalize=True)
            results[pid] = {
                "eb": torch.from_numpy(eb).float(),
                "ea": torch.from_numpy(ea).float(),
            }
        except Exception as e:
            print(f"  [{device}] {pid} ERROR: {e}")
    return_dict[device] = results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xuannv_v8_clean.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--devices", type=str, default="npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7")
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]

    print("=" * 70)
    print("  预提取所有 patch 的 embedding map")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {args.output}")
    print(f"  Devices:    {devices}")
    print("=" * 70)

    from src.config import load_config
    from src.data.dataset import HarbinPatchDataset
    cfg = load_config(args.config)
    cfg.data.preload = False
    ds = HarbinPatchDataset(cfg)
    all_patches = ds.patches[:]
    print(f"\n总 patch 数: {len(all_patches)}")

    print(f"\n多卡并行提取 ({len(devices)} 卡)...")
    start = time.time()

    manager = mp.Manager()
    return_dict = manager.dict()

    n = len(all_patches)
    chunk_size = (n + len(devices) - 1) // len(devices)
    chunks = [all_patches[i:i + chunk_size] for i in range(0, n, chunk_size)]

    processes = []
    for device, chunk in zip(devices, chunks):
        p = mp.Process(target=worker, args=(device, args.config, args.checkpoint, chunk, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并并保存
    all_results = {}
    for device in devices:
        if device in return_dict:
            all_results.update(return_dict[device])

    torch.save(all_results, args.output)
    elapsed = time.time() - start
    print(f"\n提取完成: {len(all_results)}/{len(all_patches)} patches")
    print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
