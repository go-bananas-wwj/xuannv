#!/usr/bin/env python3
"""线性探测评估 — 冻结Encoder，训练线性分类头.

参考: AEF & OlmoEarth 论文标准评估方法.
支持任务: WorldCover, JRC Water, Dynamic World

用法:
    python linear_probe_eval.py \
        --config configs/config_v17_fix_collapse.yaml \
        --checkpoint /workspace/xuannv/outputs/exp_v17_fix_collapse_0606/epoch_5.pt \
        --device npu:0 --epochs 50 --lr 0.01
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
import rasterio
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from src.inference.engine import load_backbone


def parse_args():
    p = argparse.ArgumentParser(description="线性探测评估")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", default="")
    return p.parse_args()


TASKS = [
    ("worldcover",    10, {10:0, 20:1, 30:2, 40:3, 50:4, 60:5, 80:6, 90:7}),
    ("jrc_water",      2, {}),
    ("dynamic_world", 12, {}),
]

DATA_ROOTS = [
    Path("/workspace/xuannv/data_raw/harbin/scenes"),
    Path("/workspace/xuannv/data_raw/haidian/scenes"),
]


def load_label_for_patch(patch_id: str, task_name: str):
    local_id = patch_id.split('_', 1)[1] if '_' in patch_id and not patch_id.startswith('patch_') else patch_id
    for root in DATA_ROOTS:
        candidates = [
            root / task_name / patch_id / "static.tif",
            root / task_name / local_id / "static.tif",
            root / task_name / patch_id / "2025Q2.tif",
            root / task_name / local_id / "2025Q2.tif",
        ]
        for p in candidates:
            if p.exists():
                with rasterio.open(p) as src:
                    return src.read(1), src.nodata
    return None, None


def extract_embeddings(model, dataset, cfg, device, max_samples: int = 5000):
    """提取 patch-month 级别的 spatial embedding (global pooled)."""
    model.eval()
    embeddings, labels_wc, labels_jw, labels_dw, pids = [], [], [], [], []
    
    idx = 0
    for i in range(len(dataset)):
        if idx >= max_samples:
            break
        item = dataset[i]
        pid = item.get("patch_id", f"patch_{i:06d}")
        year = item.get("year", 2025)
        month = item.get("month", 6)
        
        with torch.no_grad():
            out = model(
                source_frames=item["source_frames"].unsqueeze(0).to(device),
                source_timestamps_ms=item["source_timestamps_ms"].unsqueeze(0).to(device),
                source_frame_mask=item["source_frame_mask"].unsqueeze(0).to(device),
                source_input_mask=item["source_input_mask"].unsqueeze(0).to(device),
                source_type_ids=item["source_type_ids"].unsqueeze(0).to(device),
                valid_start_ms=torch.tensor([item["valid_start_ms"]], device=device),
                valid_end_ms=torch.tensor([item["valid_end_ms"]], device=device),
                target_relative_time=torch.zeros(1, cfg.data.num_target_sources, device=device),
                target_metadata=torch.zeros(1, cfg.data.num_target_sources, cfg.data.metadata_dim, device=device),
                skip_decoder=True,
            )
            emb = out.embedding.squeeze(0).cpu().numpy()  # [D]
        
        embeddings.append(emb)
        pids.append(pid)
        idx += 1
    
    return np.stack(embeddings), pids


def linear_probe(X_train, y_train, X_val, y_val, n_classes, epochs, lr, batch_size, device):
    """训练线性探测头并返回指标."""
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    
    D = X_train.shape[1]
    classifier = nn.Linear(D, n_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        classifier.eval()
        with torch.no_grad():
            logits = classifier(X_val_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X_val_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    bacc = balanced_accuracy_score(y_val, preds)
    cm = confusion_matrix(y_val, preds, labels=list(range(n_classes)))
    # mIoU
    ious = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou)
    miou = np.mean(ious)
    
    return {"accuracy": float(val_acc), "balanced_accuracy": float(bacc), "miou": float(miou)}


def main():
    args = parse_args()
    device = args.device
    
    print("=" * 60)
    print("  线性探测评估")
    print("=" * 60)
    
    print("\n加载模型...")
    model, dataset, cfg = load_backbone(args.config, args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print("提取 embedding...")
    embs, pids = extract_embeddings(model, dataset, cfg, device, max_samples=5000)
    print(f"  提取了 {len(embs)} 个 embedding, dim={embs.shape[1]}")
    
    results = {}
    for task_name, n_classes, mapping in TASKS:
        print(f"\n  任务: {task_name}")
        y_all = []
        valid_idx = []
        for i, pid in enumerate(pids):
            label, nodata = load_label_for_patch(pid, task_name)
            if label is None:
                continue
            # 全局平均标签（简化版线性探测，patch级别）
            # 实际应该用spatial embedding map做像素级linear probe
            # 这里先做patch级作为快速验证
            label_flat = label.flatten()
            if mapping:
                mapped = np.array([mapping.get(v, -1) for v in label_flat])
                label_flat = mapped
            else:
                # JRC Water: >0 算有水
                if task_name == "jrc_water":
                    label_flat = (label_flat > 0).astype(np.int64)
                else:
                    label_flat = label_flat.astype(np.int64)
            
            # 过滤nodata
            if nodata is not None:
                mask = label_flat != nodata
                label_flat = label_flat[mask]
            
            if len(label_flat) == 0:
                continue
            
            # patch级标签用众数
            from scipy import stats
            mode_val = stats.mode(label_flat, keepdims=True).mode[0]
            if mode_val < 0 or mode_val >= n_classes:
                continue
            y_all.append(mode_val)
            valid_idx.append(i)
        
        if len(valid_idx) < 50:
            print(f"    有效样本不足 ({len(valid_idx)}), 跳过")
            continue
        
        X = embs[valid_idx]
        y = np.array(y_all)
        
        # 分层划分train/val
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        metrics = linear_probe(X_train, y_train, X_val, y_val, n_classes,
                               args.epochs, args.lr, args.batch_size, device)
        results[task_name] = metrics
        print(f"    Acc={metrics['accuracy']:.4f} BAcc={metrics['balanced_accuracy']:.4f} mIoU={metrics['miou']:.4f}")
    
    out_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "linear_probe_results.json"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
