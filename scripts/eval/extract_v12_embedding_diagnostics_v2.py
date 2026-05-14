#!/usr/bin/env python3
"""轻量级 V12 Embedding 诊断提取 v2 — 逐个样本处理，更稳健.

输出:
  {output_dir}/global_means.npz
    - pre_norm: [N, D] float32
    - embedding: [N, D] float32
    - patch_ids: [N] str
    - year_months: [N, 2] int32

  {output_dir}/spatial_maps_sample.npz (可选)
    - pre_norm_maps: [M, D, H, W] float32
    - embedding_maps: [M, D, H, W] float32
    - patch_ids: [M] str
    - year_months: [M, 2] int32
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
    parser.add_argument("--n-spatial-patches", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="最大提取样本数 (0=全部)")
    parser.add_argument("--offset", type=int, default=0,
                        help="从第几个样本开始")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    print(f"加载配置: {args.config}")
    cfg = load_config(args.config)
    model = AEFModel(cfg).to(args.device)
    print(f"加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    n_total_all = len(dataset.monthly_samples)
    n_total = args.max_samples if args.max_samples > 0 else n_total_all
    print(f"总月度样本数: {n_total_all}，本次提取: {n_total}")

    # 随机采样索引
    random.seed(42)
    indices = list(range(n_total_all))
    random.shuffle(indices)
    use_indices = indices[args.offset:args.offset + n_total]

    sampled_patches = set(dataset.monthly_samples[i][0] for i in use_indices)
    spatial_patches = set(random.sample(list(sampled_patches), min(args.n_spatial_patches, len(sampled_patches))))
    print(f"将保存完整 spatial map 的 patches: {len(spatial_patches)} 个")

    # 存储
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
    n_success = 0
    n_fail = 0

    for loop_idx, idx in enumerate(use_indices):
        t1 = time.time()
        try:
            batch = dataset[idx]
            pid = batch["patch_id"]
            ym = batch["year_month"]

            # 准备输入
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_dev[k] = v.unsqueeze(0).to(args.device)

            with torch.no_grad():
                out = model(
                    source_frames=batch_dev["source_frames"],
                    source_timestamps_ms=batch_dev["source_timestamps_ms"],
                    source_frame_mask=batch_dev["source_frame_mask"],
                    source_input_mask=batch_dev["source_input_mask"],
                    source_type_ids=batch_dev["source_type_ids"],
                    valid_start_ms=batch_dev["valid_start_ms"],
                    valid_end_ms=batch_dev["valid_end_ms"],
                    target_relative_time=batch_dev["target_relative_time"],
                    target_metadata=batch_dev["target_metadata"],
                )

            emb = out.embedding[0].cpu().numpy().astype(np.float32)       # [D]
            pre = out.pre_norm_embedding[0].cpu().numpy().astype(np.float32)  # [D]

            global_pre_norm.append(pre)
            global_embedding.append(emb)
            global_patch_ids.append(pid)
            global_year_months.append(ym)

            if pid in spatial_patches:
                spatial_pre_norm.append(out.pre_norm_map[0].cpu().numpy().astype(np.float32))
                spatial_embedding.append(out.embedding_map[0].cpu().numpy().astype(np.float32))
                spatial_patch_ids.append(pid)
                spatial_year_months.append(ym)

            n_success += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"\n  [ERROR] idx={idx}: {e}")
            elif n_fail == 6:
                print("  ... 更多错误被抑制 ...")

        if (loop_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (loop_idx + 1) / elapsed
            eta = (n_total - loop_idx - 1) / rate
            last_time = time.time() - t1
            print(f"  {loop_idx+1}/{n_total}  success={n_success} fail={n_fail} "
                  f"({rate:.1f} samp/s, ETA {eta/60:.1f}min, last={last_time:.2f}s)")

    elapsed = time.time() - t0
    print(f"\n提取完成，耗时 {elapsed:.1f}s")
    print(f"成功: {n_success}, 失败: {n_fail}")

    # 保存
    print(f"保存到: {args.output_dir}")
    if global_pre_norm:
        np.savez(
            os.path.join(args.output_dir, "global_means.npz"),
            pre_norm=np.stack(global_pre_norm, axis=0),
            embedding=np.stack(global_embedding, axis=0),
            patch_ids=np.array(global_patch_ids),
            year_months=np.array(global_year_months, dtype=np.int32),
        )
        print(f"  global_means.npz: {len(global_pre_norm)} samples, dim={global_pre_norm[0].shape}")

    if spatial_pre_norm:
        np.savez(
            os.path.join(args.output_dir, "spatial_maps_sample.npz"),
            pre_norm_maps=np.stack(spatial_pre_norm, axis=0),
            embedding_maps=np.stack(spatial_embedding, axis=0),
            patch_ids=np.array(spatial_patch_ids),
            year_months=np.array(spatial_year_months, dtype=np.int32),
        )
        print(f"  spatial_maps_sample.npz: {len(spatial_pre_norm)} samples, map_shape={spatial_pre_norm[0].shape}")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
