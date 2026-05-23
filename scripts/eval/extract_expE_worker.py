#!/usr/bin/env python3
import sys, os, argparse, time
sys.path.insert(0, "/workspace/xuannv")
# 强制无缓冲输出
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch_npu
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--patches", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    target_patches = set(args.patches.split(","))
    
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    
    print(f"[Worker] 设备: {args.device}")
    print(f"[Worker] 处理 {len(target_patches)} 个 patches")
    
    cfg = load_config(args.config)
    model = AEFModel(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    
    patch_to_indices = defaultdict(list)
    for idx, (pid, year, month) in enumerate(dataset.monthly_samples):
        if pid in target_patches:
            patch_to_indices[pid].append(idx)
    
    target_indices = []
    for pid in sorted(target_patches):
        target_indices.extend(patch_to_indices[pid])
    
    print(f"[Worker] 目标样本数: {len(target_indices)}")
    
    patch_data = {}
    t0 = time.time()
    processed = 0
    
    with torch.no_grad():
        batch_items = []
        for idx in target_indices:
            item = dataset[idx]
            batch_items.append(item)
            
            if len(batch_items) == args.batch_size:
                batch = {}
                for key in ["source_frames", "source_timestamps_ms", "source_frame_mask",
                            "source_input_mask", "source_type_ids", "valid_start_ms",
                            "valid_end_ms", "target_relative_time", "target_metadata"]:
                    batch[key] = torch.stack([it[key] for it in batch_items])
                
                output = model(
                    source_frames=batch["source_frames"].to(args.device),
                    source_timestamps_ms=batch["source_timestamps_ms"].to(args.device),
                    source_frame_mask=batch["source_frame_mask"].to(args.device),
                    source_input_mask=batch["source_input_mask"].to(args.device),
                    source_type_ids=batch["source_type_ids"].to(args.device),
                    valid_start_ms=batch["valid_start_ms"].to(args.device),
                    valid_end_ms=batch["valid_end_ms"].to(args.device),
                    target_relative_time=batch["target_relative_time"].to(args.device),
                    target_metadata=batch["target_metadata"].to(args.device),
                )
                
                emb_map = output.embedding_map.cpu()
                pre_map = output.pre_norm_map.cpu() if output.pre_norm_map is not None else None
                
                for i, item in enumerate(batch_items):
                    pid = item["patch_id"]
                    if pid not in patch_data:
                        patch_data[pid] = []
                    patch_data[pid].append({
                        "year_month": item["year_month"],
                        "valid_start": float(item["valid_start_ms"]),
                        "valid_end": float(item["valid_end_ms"]),
                        "embedding_map": emb_map[i].numpy().astype(np.float32),
                        "pre_norm_map": pre_map[i].numpy().astype(np.float32) if pre_map is not None else None,
                    })
                
                processed += len(batch_items)
                batch_items = []
                
                if processed % 100 == 0:
                    elapsed = time.time() - t0
                    speed = processed / elapsed
                    eta = (len(target_indices) - processed) / speed if speed > 0 else 0
                    print(f"  [{processed}/{len(target_indices)}] {speed:.1f} 样本/秒, ETA {eta/60:.1f}min")
        
        if batch_items:
            batch = {}
            for key in ["source_frames", "source_timestamps_ms", "source_frame_mask",
                        "source_input_mask", "source_type_ids", "valid_start_ms",
                        "valid_end_ms", "target_relative_time", "target_metadata"]:
                batch[key] = torch.stack([it[key] for it in batch_items])
            
            output = model(
                source_frames=batch["source_frames"].to(args.device),
                source_timestamps_ms=batch["source_timestamps_ms"].to(args.device),
                source_frame_mask=batch["source_frame_mask"].to(args.device),
                source_input_mask=batch["source_input_mask"].to(args.device),
                source_type_ids=batch["source_type_ids"].to(args.device),
                valid_start_ms=batch["valid_start_ms"].to(args.device),
                valid_end_ms=batch["valid_end_ms"].to(args.device),
                target_relative_time=batch["target_relative_time"].to(args.device),
                target_metadata=batch["target_metadata"].to(args.device),
            )
            
            emb_map = output.embedding_map.cpu()
            pre_map = output.pre_norm_map.cpu() if output.pre_norm_map is not None else None
            
            for i, item in enumerate(batch_items):
                pid = item["patch_id"]
                if pid not in patch_data:
                    patch_data[pid] = []
                patch_data[pid].append({
                    "year_month": item["year_month"],
                    "valid_start": float(item["valid_start_ms"]),
                    "valid_end": float(item["valid_end_ms"]),
                    "embedding_map": emb_map[i].numpy().astype(np.float32),
                    "pre_norm_map": pre_map[i].numpy().astype(np.float32) if pre_map is not None else None,
                })
            processed += len(batch_items)
    
    elapsed = time.time() - t0
    print(f"[Worker] 提取完成，耗时 {elapsed:.1f}s，{processed} 样本")
    
    saved = 0
    for pid, entries in patch_data.items():
        entries.sort(key=lambda e: (e["year_month"][0], e["year_month"][1]))
        emb_stack = np.stack([e["embedding_map"] for e in entries], axis=0)
        pre_stack = np.stack([e["pre_norm_map"] for e in entries], axis=0) if entries[0]["pre_norm_map"] is not None else None
        year_months = np.array([e["year_month"] for e in entries], dtype=np.int32)
        valid_starts = np.array([e["valid_start"] for e in entries], dtype=np.float64)
        valid_ends = np.array([e["valid_end"] for e in entries], dtype=np.float64)
        
        np.savez(
            os.path.join(args.output_dir, f"{pid}.npz"),
            embedding_map=emb_stack,
            pre_norm_map=pre_stack,
            year_months=year_months,
            valid_starts=valid_starts,
            valid_ends=valid_ends,
        )
        saved += 1
    
    print(f"[Worker] 完成，保存 {saved} 个 npz")


if __name__ == "__main__":
    main()
