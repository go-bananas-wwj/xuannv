#!/usr/bin/envs xuannv/bin/python3
"""Batch embedding extraction - much faster version"""
from __future__ import annotations
import os, sys, json, time, calendar
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
from pathlib import Path
from tqdm import tqdm

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.utils.checkpoint import load_checkpoint

MONTHS = [(2025, 4), (2025, 6), (2025, 8), (2025, 9), (2025, 10)]

def month_to_window(year, month):
    start_sec = time.mktime((year, month, 1, 0, 0, 0, 0, 0, 0))
    last_day = calendar.monthrange(year, month)[1]
    end_sec = time.mktime((year, month, last_day, 23, 59, 59, 0, 0, 0))
    return int(start_sec * 1000), int(end_sec * 1000)

def extract_embeddings_for_experiment(checkpoint_path, config_path, output_dir, device_str="npu:0", batch_size=16):
    device = torch.device(device_str)
    cfg = load_config(config_path)
    cfg.data.preload = False
    cfg.data.num_workers = 4
    dataset = HarbinPatchDataset(cfg=cfg)
    print(f"Dataset: {len(dataset.patches)} patches, {len(dataset.monthly_samples)} monthly samples")

    model = AEFModel(cfg=cfg).to(device)
    state_dict = load_checkpoint(checkpoint_path, device=device_str)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_map = {(pid, year, month): idx for idx, (pid, year, month) in enumerate(dataset.monthly_samples)}

    with torch.no_grad():
        for year, month in MONTHS:
            month_str = f"{year}-{month:02d}"
            print(f"\n  Extracting {month_str}...")
            valid_start, valid_end = month_to_window(year, month)

            # Collect all valid indices for this month
            valid_indices = []
            valid_pids = []
            for pid in dataset.patches:
                key = (pid, year, month)
                if key in sample_map:
                    valid_indices.append(sample_map[key])
                    valid_pids.append(pid)
                else:
                    print(f"    Skip {pid}: no data for {month_str}")

            # Batch forward
            for batch_start in tqdm(range(0, len(valid_indices), batch_size), desc=f"  {month_str}"):
                batch_end = min(batch_start + batch_size, len(valid_indices))
                batch_indices = valid_indices[batch_start:batch_end]
                batch_pids = valid_pids[batch_start:batch_end]

                # Stack batch
                samples = [dataset[i] for i in batch_indices]
                batch_dev = {}
                for k in samples[0].keys():
                    vals = [s[k] for s in samples]
                    if isinstance(vals[0], torch.Tensor):
                        batch_dev[k] = torch.stack(vals).to(device)
                    else:
                        batch_dev[k] = vals

                emb_map = model(
                    source_frames=batch_dev["source_frames"],
                    source_timestamps_ms=batch_dev["source_timestamps_ms"],
                    source_frame_mask=batch_dev["source_frame_mask"],
                    source_input_mask=batch_dev["source_input_mask"],
                    source_type_ids=batch_dev["source_type_ids"],
                    valid_start_ms=batch_dev["valid_start_ms"],
                    valid_end_ms=batch_dev["valid_end_ms"],
                    target_relative_time=batch_dev["target_relative_time"],
                    target_metadata=batch_dev["target_metadata"],
                    skip_decoder=True,
                ).embedding_map

                for i, pid in enumerate(batch_pids):
                    np.save(output_dir / f"{pid}_{month_str}.npy", emb_map[i].cpu().numpy())

    metadata = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "patch_ids": dataset.patches,
        "months": [f"{y}-{m:02d}" for y, m in MONTHS],
        "embedding_dim": cfg.model.embedding_dim,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone! Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    extract_embeddings_for_experiment(args.checkpoint, args.config, args.output_dir, args.device, args.batch_size)
