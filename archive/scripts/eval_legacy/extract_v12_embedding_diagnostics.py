#!/usr/bin/env python3
"""轻量级 V12 Embedding 诊断提取.

提取两类数据:
1. 所有 patch 所有月份的 global mean embedding (pre_norm + L2 norm) — 用于时间分析
2. 随机 N 个 patch 的完整 pre_norm_map — 用于 spatial 质量分析

输出:
  {output_dir}/global_means.npz
    - pre_norm: [N_total, 128] float32
    - embedding: [N_total, 128] float32
    - patch_ids: [N_total] str
    - year_months: [N_total, 2] int32

  {output_dir}/spatial_maps_sample.npz
    - pre_norm_maps: [N_spatial_samples, 128, H, W] float32
    - embeddings_maps: [N_spatial_samples, 128, H, W] float32
    - patch_ids: [N_spatial_samples] str
    - year_months: [N_spatial_samples, 2] int32
"""
import sys, os, argparse, time, random
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-spatial-patches", type=int, default=50,
                        help="随机选取多少 patch 保存完整 spatial map")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="最大提取样本数 (0=全部)")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    # ── 加载模型 ──
    print(f"加载配置: {args.config}")
    cfg = load_config(args.config)
    model = AEFModel(cfg).to(args.device)
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 创建 Dataset ──
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    n_total_all = len(dataset.monthly_samples)
    n_total = args.max_samples if args.max_samples > 0 else n_total_all
    print(f"总月度样本数: {n_total_all}，本次提取: {n_total}")
    print(f"Patches: {len(dataset.patches)}")

    # 随机采样索引
    indices = list(range(n_total_all))
    random.seed(42)
    random.shuffle(indices)
    use_indices = sorted(indices[:n_total])  # 排序后按顺序遍历，效率更高

    # 选取保存 spatial map 的 patch (在采样范围内)
    sampled_patches = set(dataset.monthly_samples[i][0] for i in use_indices)
    spatial_patches = set(random.sample(list(sampled_patches), min(args.n_spatial_patches, len(sampled_patches))))
    print(f"将保存完整 spatial map 的 patches: {len(spatial_patches)} 个")

    # ── 遍历提取 ──
    global_pre_norm = []
    global_embedding = []
    global_patch_ids = []
    global_year_months = []

    spatial_pre_norm = []
    spatial_embedding = []
    spatial_patch_ids = []
    spatial_year_months = []

    print("开始提取...")
    t0 = time.time()

    # 手动 batch 化以提高效率
    batch_inputs = []
    batch_meta = []

    def run_batch():
        nonlocal batch_inputs, batch_meta
        if not batch_inputs:
            return
        B = len(batch_inputs)
        # Stack tensors
        dev = args.device
        source_frames = torch.stack([b["source_frames"] for b in batch_inputs], dim=0).to(dev)
        source_ts = torch.stack([b["source_timestamps_ms"] for b in batch_inputs], dim=0).to(dev)
        source_mask = torch.stack([b["source_frame_mask"] for b in batch_inputs], dim=0).to(dev)
        input_mask = torch.stack([b["source_input_mask"] for b in batch_inputs], dim=0).to(dev)
        type_ids = torch.stack([b["source_type_ids"] for b in batch_inputs], dim=0).to(dev)
        valid_start = torch.stack([b["valid_start_ms"] for b in batch_inputs], dim=0).to(dev)
        valid_end = torch.stack([b["valid_end_ms"] for b in batch_inputs], dim=0).to(dev)
        target_time = torch.stack([b["target_relative_time"] for b in batch_inputs], dim=0).to(dev)
        target_meta = torch.stack([b["target_metadata"] for b in batch_inputs], dim=0).to(dev)

        with torch.no_grad():
            out = model(
                source_frames=source_frames,
                source_timestamps_ms=source_ts,
                source_frame_mask=source_mask,
                source_input_mask=input_mask,
                source_type_ids=type_ids,
                valid_start_ms=valid_start,
                valid_end_ms=valid_end,
                target_relative_time=target_time,
                target_metadata=target_meta,
            )

        emb = out.embedding.cpu().numpy().astype(np.float32)       # [B, D]
        pre = out.pre_norm_embedding.cpu().numpy().astype(np.float32)  # [B, D]
        emb_map = out.embedding_map.cpu()       # [B, D, H, W]
        pre_map = out.pre_norm_map.cpu() if out.pre_norm_map is not None else None  # [B, D, H, W]

        for i in range(B):
            pid, ym = batch_meta[i]
            global_pre_norm.append(pre[i])
            global_embedding.append(emb[i])
            global_patch_ids.append(pid)
            global_year_months.append(ym)

            if pid in spatial_patches:
                spatial_pre_norm.append(pre_map[i].numpy().astype(np.float32))
                spatial_embedding.append(emb_map[i].numpy().astype(np.float32))
                spatial_patch_ids.append(pid)
                spatial_year_months.append(ym)

        batch_inputs = []
        batch_meta = []

    for loop_idx, idx in enumerate(use_indices):
        batch = dataset[idx]
        pid = batch["patch_id"]
        ym = batch["year_month"]

        # 准备 tensor batch
        batch_item = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_item[k] = v
            # 非 tensor 忽略
        batch_inputs.append(batch_item)
        batch_meta.append((pid, ym))

        if len(batch_inputs) >= args.batch_size:
            run_batch()

        if (loop_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (loop_idx + 1) / elapsed
            eta = (n_total - loop_idx - 1) / rate
            print(f"  {loop_idx+1}/{n_total}  ({rate:.1f} samples/s, ETA {eta/60:.1f}min)")

    # 最后一批
    run_batch()

    elapsed = time.time() - t0
    print(f"\n提取完成，耗时 {elapsed:.1f}s")
    print(f"Global samples: {len(global_pre_norm)}")
    print(f"Spatial samples: {len(spatial_pre_norm)}")

    # ── 保存 ──
    print(f"保存到: {args.output_dir}")
    np.savez(
        os.path.join(args.output_dir, "global_means.npz"),
        pre_norm=np.stack(global_pre_norm, axis=0),
        embedding=np.stack(global_embedding, axis=0),
        patch_ids=np.array(global_patch_ids),
        year_months=np.array(global_year_months, dtype=np.int32),
    )

    if spatial_pre_norm:
        np.savez(
            os.path.join(args.output_dir, "spatial_maps_sample.npz"),
            pre_norm_maps=np.stack(spatial_pre_norm, axis=0),
            embedding_maps=np.stack(spatial_embedding, axis=0),
            patch_ids=np.array(spatial_patch_ids),
            year_months=np.array(spatial_year_months, dtype=np.int32),
        )

    print("\n全部完成！")
    print(f"  global_means: pre_norm shape = {np.stack(global_pre_norm, axis=0).shape}")
    if spatial_pre_norm:
        print(f"  spatial_maps: pre_norm_maps shape = {np.stack(spatial_pre_norm, axis=0).shape}")


if __name__ == "__main__":
    main()
