#!/usr/bin/env python3
"""简单直接的 embedding 提取脚本 — 避免 DataLoader 和 preload 问题."""
import sys, os, time
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu


def main():
    config = "configs/xuannv_v2_expE_pure_recon.yaml"
    checkpoint = "/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/epoch_best_epoch52.pt"
    device = "npu:0"
    batch_size = 4
    output_dir = "/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/eval/embeddings_all_months"
    
    os.makedirs(output_dir, exist_ok=True)
    
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    
    print(f"加载配置: {config}")
    cfg = load_config(config)
    
    print(f"加载模型...")
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"创建数据集 (preload=False)...")
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    
    print(f"总样本数: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    
    # 收集每个 patch 的 embeddings
    patch_data = {}
    total = len(dataset)
    t0 = time.time()
    
    with torch.no_grad():
        batch_items = []
        for idx in range(total):
            item = dataset[idx]
            batch_items.append(item)
            
            if len(batch_items) == batch_size or idx == total - 1:
                # 手动 batch
                batch = {}
                for key in ["source_frames", "source_timestamps_ms", "source_frame_mask",
                            "source_input_mask", "source_type_ids", "valid_start_ms",
                            "valid_end_ms", "target_relative_time", "target_metadata"]:
                    batch[key] = torch.stack([it[key] for it in batch_items])
                
                output = model(
                    source_frames=batch["source_frames"].to(device),
                    source_timestamps_ms=batch["source_timestamps_ms"].to(device),
                    source_frame_mask=batch["source_frame_mask"].to(device),
                    source_input_mask=batch["source_input_mask"].to(device),
                    source_type_ids=batch["source_type_ids"].to(device),
                    valid_start_ms=batch["valid_start_ms"].to(device),
                    valid_end_ms=batch["valid_end_ms"].to(device),
                    target_relative_time=batch["target_relative_time"].to(device),
                    target_metadata=batch["target_metadata"].to(device),
                )
                
                emb_map = output.embedding_map.cpu()
                pre_map = output.pre_norm_map.cpu() if output.pre_norm_map is not None else None
                
                for i, item in enumerate(batch_items):
                    pid = item["patch_id"]
                    ym = item["year_month"]
                    if pid not in patch_data:
                        patch_data[pid] = []
                    patch_data[pid].append({
                        "year_month": ym,
                        "valid_start": float(item["valid_start_ms"]),
                        "valid_end": float(item["valid_end_ms"]),
                        "embedding_map": emb_map[i].numpy().astype(np.float32),
                        "pre_norm_map": pre_map[i].numpy().astype(np.float32) if pre_map is not None else None,
                    })
                
                batch_items = []
                
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    speed = (idx + 1) / elapsed
                    eta = (total - idx - 1) / speed if speed > 0 else 0
                    print(f"  [{idx+1}/{total}] {speed:.1f} 样本/秒, ETA: {eta/60:.1f}min")
    
    elapsed = time.time() - t0
    print(f"\n提取完成，耗时 {elapsed:.1f}s")
    print(f"共 {len(patch_data)} patches")
    
    # 保存
    print(f"保存到: {output_dir}")
    saved = 0
    for pid, entries in patch_data.items():
        entries.sort(key=lambda e: (e["year_month"][0], e["year_month"][1]))
        emb_stack = np.stack([e["embedding_map"] for e in entries], axis=0)
        pre_stack = np.stack([e["pre_norm_map"] for e in entries], axis=0) if entries[0]["pre_norm_map"] is not None else None
        year_months = np.array([e["year_month"] for e in entries], dtype=np.int32)
        valid_starts = np.array([e["valid_start"] for e in entries], dtype=np.float64)
        valid_ends = np.array([e["valid_end"] for e in entries], dtype=np.float64)
        
        np.savez(
            os.path.join(output_dir, f"{pid}.npz"),
            embedding_map=emb_stack,
            pre_norm_map=pre_stack,
            year_months=year_months,
            valid_starts=valid_starts,
            valid_ends=valid_ends,
        )
        saved += 1
        if saved % 50 == 0:
            print(f"  已保存 {saved}/{len(patch_data)}")
    
    print(f"\n全部完成，保存 {saved} 个 npz")


if __name__ == "__main__":
    main()
