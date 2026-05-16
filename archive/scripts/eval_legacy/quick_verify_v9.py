#!/usr/bin/env python3
"""快速验证 V9 checkpoint — 单卡，少量 patch，估算 AUC.

用法:
    cd /workspace/xuannv
    python scripts/eval/quick_verify_v9.py \
        --checkpoint /workspace/outputs/xuannv_backbone_v9_temporal/epoch_best_epoch65.pt \
        --config configs/xuannv_v9_temporal.yaml \
        --device npu:0 \
        --max-patches 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset


BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)   # 2023-07 ~ 2023-12
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)     # 2024-07 ~ 2024-12


def extract_embedding_map(model, dataset, idx, valid_start, valid_end, device):
    """提取单个 patch 的 embedding map."""
    item = dataset[idx]
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
             for k, v in item.items()}

    with torch.no_grad():
        out = model(
            source_frames=batch["source_frames"],
            source_timestamps_ms=batch["source_timestamps_ms"],
            source_frame_mask=batch["source_frame_mask"],
            source_input_mask=batch["source_input_mask"],
            source_type_ids=batch["source_type_ids"],
            valid_start_ms=torch.tensor([valid_start], dtype=torch.float64, device=device),
            valid_end_ms=torch.tensor([valid_end], dtype=torch.float64, device=device),
            target_relative_time=batch["target_relative_time"],
            target_metadata=batch["target_metadata"],
        )
    emb = out.embedding_map[0]  # [D, H, W]
    return F.normalize(emb, p=2, dim=0).cpu().numpy()


def load_labels():
    """加载变化检测标注."""
    import json
    label_path = Path("/workspace/raw/harbin_scenes/change_labels.json")
    if label_path.exists():
        with open(label_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/xuannv_v9_temporal.yaml")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--max-patches", type=int, default=100)
    parser.add_argument("--use-cd-head", type=str, default=None, help="CD Head checkpoint (可选)")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)

    print(f"[QuickVerify] Loading checkpoint: {args.checkpoint}")
    cfg = load_config(args.config)
    cfg.data.preload = False  # 避免长时间缓存加载

    model = AEFModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # CD Head (可选)
    cd_head = None
    if args.use_cd_head:
        from src.models.heads import ChangeDetectionHeadV3
        cd_head = ChangeDetectionHeadV3(embedding_dim=cfg.model.embedding_dim).to(device)
        cd_ckpt = torch.load(args.use_cd_head, map_location=device, weights_only=False)
        cd_head.load_state_dict(cd_ckpt)
        cd_head.eval()

    dataset = HarbinPatchDataset(cfg)
    dataset.training = False

    # 加载标注
    labels = load_labels()
    
    # 选择有标注的 patch
    labeled_patches = []
    for pid in dataset.patches:
        if pid in labels:
            labeled_patches.append(pid)
    
    if len(labeled_patches) == 0:
        print("[QuickVerify] WARNING: No labeled patches found. Using first 50 patches.")
        labeled_patches = dataset.patches[:50]
    
    test_patches = labeled_patches[:args.max_patches]
    print(f"[QuickVerify] Testing {len(test_patches)} patches ({len(labeled_patches)} labeled total)")

    y_true = []
    y_score_cos = []   # cosine distance
    y_score_cd = []    # CD Head score (如果提供)

    for pid in test_patches:
        idx = dataset.patches.index(pid)
        
        eb = extract_embedding_map(model, dataset, idx, BEFORE_WINDOW[0], BEFORE_WINDOW[1], device)
        ea = extract_embedding_map(model, dataset, idx, AFTER_WINDOW[0], AFTER_WINDOW[1], device)

        # Global mean cosine distance
        eb_flat = eb.reshape(eb.shape[0], -1).mean(axis=1)
        ea_flat = ea.reshape(ea.shape[0], -1).mean(axis=1)
        cos_sim = np.dot(eb_flat, ea_flat) / (np.linalg.norm(eb_flat) * np.linalg.norm(ea_flat) + 1e-8)
        y_score_cos.append(1.0 - cos_sim)

        # CD Head (可选)
        if cd_head is not None:
            with torch.no_grad():
                eb_t = torch.from_numpy(eb).unsqueeze(0).to(device)
                ea_t = torch.from_numpy(ea).unsqueeze(0).to(device)
                cd_out = cd_head(eb_t, ea_t)
                y_score_cd.append(float(cd_out["change_prob"].cpu().item()))

        # Label
        y_true.append(1 if labels.get(pid, {}).get("has_change", False) else 0)

    y_true = np.array(y_true)
    y_score_cos = np.array(y_score_cos)

    # 计算 AUC
    if len(np.unique(y_true)) > 1:
        auc_cos = roc_auc_score(y_true, y_score_cos)
        print(f"\n[QuickVerify] Cosine Distance AUC: {auc_cos:.4f}")
        
        if cd_head is not None:
            y_score_cd = np.array(y_score_cd)
            auc_cd = roc_auc_score(y_true, y_score_cd)
            print(f"[QuickVerify] CD Head AUC: {auc_cd:.4f}")
        
        # 统计
        n_change = y_true.sum()
        n_nochange = len(y_true) - n_change
        mean_score_change = y_score_cos[y_true == 1].mean()
        mean_score_nochange = y_score_cos[y_true == 0].mean()
        print(f"\n[QuickVerify] Stats:")
        print(f"  Change patches: {n_change}, No-change: {n_nochange}")
        print(f"  Mean score (change): {mean_score_change:.4f}")
        print(f"  Mean score (nochange): {mean_score_nochange:.4f}")
        print(f"  Separation: {mean_score_change - mean_score_nochange:.4f}")
    else:
        print("[QuickVerify] WARNING: Only one class in labels, cannot compute AUC")

    print(f"\n[QuickVerify] Done. Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
