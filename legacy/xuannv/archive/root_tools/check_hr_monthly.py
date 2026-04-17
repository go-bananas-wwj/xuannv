#!/usr/bin/env python3
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

CKPT = "/workspace/outputs/aef_qwen_v2_hr_only_small/epoch_99.pt"
CONFIG = "/workspace/xuannv/configs/qwen_v2_hr_only_small.yaml"

MONTHS = {
    "2025-04": (1744171200000.0, 1744171200000.0),
    "2025-06": (1750564800000.0, 1750564800000.0),
    "2025-08": (1754280000000.0, 1754280000000.0),
    "2025-09": (1756699200000.0, 1756699200000.0),
    "2025-10": (1759291200000.0, 1759291200000.0),
}

cfg = load_config(CONFIG)
model = AEFModel(cfg).to("cuda:0")
ckpt = torch.load(CKPT, map_location="cuda:0", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

ds = HarbinPatchDataset(cfg)
ds.training = False
ds._spatial_augmentation = False

def get_emb(idx, vs, ve):
    batch = ds[idx]
    batch["valid_start_ms"] = torch.tensor(vs, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(ve, dtype=torch.float64)
    dev = {k: (v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**{k:dev[k] for k in [
            "source_frames","source_timestamps_ms","source_frame_mask",
            "source_input_mask","source_type_ids","valid_start_ms",
            "valid_end_ms","target_relative_time","target_metadata"
        ]})
    return F.normalize(out.embedding_map, p=2, dim=1)[0].cpu().numpy()

def dist(a,b):
    D,H,W = a.shape
    fa=a.reshape(D,-1); fb=b.reshape(D,-1)
    fa=fa/np.linalg.norm(fa,axis=0,keepdims=True)
    fb=fb/np.linalg.norm(fb,axis=0,keepdims=True)
    return float(np.mean(((1.0-np.sum(fa*fb,axis=0))/2.0).reshape(H,W)))

pids = ds.patches[:10]
embs = {}
for i, pid in enumerate(pids):
    embs[pid] = {}
    for name, (vs, ve) in MONTHS.items():
        try: embs[pid][name] = get_emb(i, vs, ve)
        except: pass

print("HR-only Small (epoch_99) - Monthly Embedding Distances")
for label, m1, m2 in [("Apr-Jun","2025-04","2025-06"), ("Jun-Aug","2025-06","2025-08"), ("Aug-Sep","2025-08","2025-09"), ("Sep-Oct","2025-09","2025-10")]:
    vals = [dist(embs[pid][m1], embs[pid][m2]) for pid in pids if m1 in embs[pid] and m2 in embs[pid]]
    if vals: print(f"{label}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")
