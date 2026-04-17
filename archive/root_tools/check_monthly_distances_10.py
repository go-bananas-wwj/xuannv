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

CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/epoch_499.pt"

PERIODS = {
    "2025-04~2025-06": (1743465600000.0, 1751328000000.0),
    "2025-06~2025-08": (1748736000000.0, 1756684800000.0),
    "2025-08~2025-09": (1754006400000.0, 1759267200000.0),
    "2025-09~2025-10": (1756089600000.0, 1761945600000.0),
}

MONTHS = {
    "2025-04": (1743465600000.0, 1746057600000.0),
    "2025-06": (1748736000000.0, 1751328000000.0),
    "2025-08": (1754006400000.0, 1756684800000.0),
    "2025-09": (1756684800000.0, 1759267200000.0),
    "2025-10": (1759267200000.0, 1761945600000.0),
}

cfg = load_config(CONFIG_PATH)
model = AEFModel(cfg).to("cuda:0")
ckpt = torch.load(CKPT_PATH, map_location="cuda:0", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

ds = HarbinPatchDataset(cfg)
ds.training = False
ds._spatial_augmentation = False

def get_emb(idx, w):
    batch = ds[idx]
    batch["valid_start_ms"] = torch.tensor(w[0], dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(w[1], dtype=torch.float64)
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

pids = ds.patches[:15]

embs_month = {}
embs_period = {}
for i, pid in enumerate(pids):
    embs_month[pid] = {}
    for name, w in MONTHS.items():
        try: embs_month[pid][name] = get_emb(i, w)
        except: pass
    embs_period[pid] = {}
    for name, w in PERIODS.items():
        try: embs_period[pid][name] = get_emb(i, w)
        except: pass

results = []
for label, m1, m2 in [("Apr-Jun","2025-04","2025-06"), ("Jun-Aug","2025-06","2025-08"), ("Aug-Sep","2025-08","2025-09"), ("Sep-Oct","2025-09","2025-10")]:
    vals = [dist(embs_month[pid][m1], embs_month[pid][m2]) for pid in pids if m1 in embs_month[pid] and m2 in embs_month[pid]]
    if vals: results.append(f"{label}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

for label, p1, p2 in [("Apr-Jun vs Jun-Aug","2025-04~2025-06","2025-06~2025-08"), ("Jun-Aug vs Aug-Sep","2025-06~2025-08","2025-08~2025-09"), ("Aug-Sep vs Sep-Oct","2025-08~2025-09","2025-09~2025-10")]:
    vals = [dist(embs_period[pid][p1], embs_period[pid][p2]) for pid in pids if p1 in embs_period[pid] and p2 in embs_period[pid]]
    if vals: results.append(f"{label}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

vals = []
for i in range(len(pids)):
    try: vals.append(dist(get_emb(i, (1672531200000.0, 1703980800000.0)), get_emb(i, (1735689600000.0, 1767225600000.0))))
    except: pass
if vals: results.append(f"2023 vs 2025: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

with open("/workspace/xuannv/monthly_dist_results.txt", "w") as f:
    f.write("\n".join(results))
print("Done")
