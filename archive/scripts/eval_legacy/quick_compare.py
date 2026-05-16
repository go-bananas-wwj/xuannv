#!/usr/bin/env python3
"""快速比较两个 patch 的 embedding — 验证 spatial map 是否有区分度."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch_npu
except ImportError:
    pass

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

# ── 配置 ──
CKPT = "/workspace/outputs/xuannv_v12_clean/epoch_best_epoch18.pt"
CFG = "configs/xuannv_v12_clean.yaml"
DEVICE = "npu:0"

# 时间窗口（任意一个）
W_START, W_END = 1688169600000.0, 1703980800000.0

# ── 加载模型 ──
print("加载模型...")
cfg = load_config(CFG)
model = AEFModel(cfg).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

cfg.data.preload = False
dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

# ── 提取 embedding ──
@torch.no_grad()
def extract(pid):
    idx = dataset.patches.index(pid)
    item = dataset[idx]
    out = model(
        source_frames=item["source_frames"].unsqueeze(0).to(DEVICE),
        source_timestamps_ms=item["source_timestamps_ms"].unsqueeze(0).to(DEVICE),
        source_frame_mask=item["source_frame_mask"].unsqueeze(0).to(DEVICE),
        source_input_mask=item["source_input_mask"].unsqueeze(0).to(DEVICE),
        source_type_ids=item["source_type_ids"].unsqueeze(0).to(DEVICE),
        valid_start_ms=torch.tensor([W_START], dtype=torch.int64, device=DEVICE),
        valid_end_ms=torch.tensor([W_END], dtype=torch.int64, device=DEVICE),
        target_relative_time=torch.zeros(1, cfg.data.num_target_sources, device=DEVICE),
        target_metadata=torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=DEVICE),
        skip_decoder=True,
    )
    emb_map = out.embedding_map.squeeze(0).cpu().numpy()  # [D, H, W]
    emb_vec = out.embedding.squeeze(0).cpu().numpy()       # [D]
    return emb_map, emb_vec

for pid in ["patch_000287", "patch_000137"]:
    print(f"\n提取 {pid}...")
    emb_map, emb_vec = extract(pid)
    print(f"  spatial map shape: {emb_map.shape}")
    print(f"  global mean shape: {emb_vec.shape}")
    print(f"  global mean norm:  {np.linalg.norm(emb_vec):.4f}")
    print(f"  global mean 前5维: {emb_vec[:5]}")
    print(f"  spatial map 均值:  {emb_map.mean(axis=(1,2))[:5]}")
    print(f"  spatial map std:   {emb_map.std():.4f}")

# ── 比较 ──
emb287_map, emb287_vec = extract("patch_000287")
emb137_map, emb137_vec = extract("patch_000137")

print("\n" + "="*60)
print("  比较结果")
print("="*60)

# Global mean 比较
cos_vec = float(np.dot(emb287_vec, emb137_vec))
print(f"\nGlobal Mean cosine similarity: {cos_vec:.4f}")
print(f"Global Mean distance:            {1-cos_vec:.4f}")

# Spatial map 逐像素比较
D, H, W = emb287_map.shape
emb287_flat = emb287_map.reshape(D, -1)  # [D, H*W]
emb137_flat = emb137_map.reshape(D, -1)

cos_all = np.sum(emb287_flat * emb137_flat, axis=0)  # [H*W]
print(f"\nSpatial map cosine similarity:")
print(f"  mean:  {cos_all.mean():.4f}")
print(f"  min:   {cos_all.min():.4f}")
print(f"  max:   {cos_all.max():.4f}")
print(f"  std:   {cos_all.std():.4f}")

# 取各自中心区域比较
center = slice(H//4, 3*H//4)
emb287_center = emb287_map[:, center, center].reshape(D, -1)
emb137_center = emb137_map[:, center, center].reshape(D, -1)
cos_center = np.sum(emb287_center * emb137_center, axis=0)
print(f"\n中心区域 cosine similarity:")
print(f"  mean:  {cos_center.mean():.4f}")
print(f"  min:   {cos_center.min():.4f}")
print(f"  max:   {cos_center.max():.4f}")
