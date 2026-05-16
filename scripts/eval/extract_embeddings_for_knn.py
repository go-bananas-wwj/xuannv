#!/usr/bin/env python3
"""Step 1: 用 NPU 提取所有 patch 的 embedding，保存为 .npy."""
import sys, os, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset
from src.utils.checkpoint import load_checkpoint

def extract_embeddings(model, dataset, device, cfg):
    embeddings = {}
    for i, pid in enumerate(dataset.patches):
        idx = dataset.patches.index(pid)
        item = dataset[idx]
        with torch.no_grad():
            out = model(
                source_frames=item["source_frames"].unsqueeze(0).to(device),
                source_timestamps_ms=item["source_timestamps_ms"].unsqueeze(0).to(device),
                source_frame_mask=item["source_frame_mask"].unsqueeze(0).to(device),
                source_input_mask=item["source_input_mask"].unsqueeze(0).to(device),
                source_type_ids=item["source_type_ids"].unsqueeze(0).to(device),
                valid_start_ms=item["valid_start_ms"].unsqueeze(0).to(device),
                valid_end_ms=item["valid_end_ms"].unsqueeze(0).to(device),
                target_relative_time=torch.zeros(1, cfg.data.num_target_sources, device=device),
                target_metadata=torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=device),
                skip_decoder=True,
            )
            emb = F.normalize(out.embedding_map, p=2, dim=1)
        embeddings[pid] = emb.squeeze(0).cpu().numpy()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dataset.patches)} done")
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    
    exp_dir = Path(f"/workspace/outputs/xuannv_round1/{args.experiment}")
    ckpt_path = list(exp_dir.glob("epoch_best_*.pt"))
    if not ckpt_path:
        print(f"No checkpoint in {exp_dir}"); return
    ckpt_path = ckpt_path[0]
    config_path = Path("/workspace/xuannv/configs") / f"{args.experiment}.yaml"
    print(f"Exp: {args.experiment}, ckpt: {ckpt_path}")
    
    cfg = load_config(str(config_path))
    cfg.experiment.name = args.experiment
    cfg.data.preload = False
    
    device = args.device
    if device.startswith("npu"):
        import torch_npu
        torch_npu.npu.set_device(int(device.split(":")[-1]))
    
    model = AEFModel(cfg)
    model.load_state_dict(load_checkpoint(str(ckpt_path)))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    dataset = HarbinPatchDataset(cfg)
    print(f"Dataset: {len(dataset.patches)} patches")
    
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataset, device, cfg)
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Save
    out_dir = exp_dir / "downstream_knn"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "patch_ids.json", 'w') as f:
        import json
        json.dump(list(embeddings.keys()), f)
    print(f"Saved to {out_dir}/embeddings.npy")

if __name__ == "__main__":
    main()
