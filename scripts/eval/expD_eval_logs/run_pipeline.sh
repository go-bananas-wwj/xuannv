#!/bin/bash
set -e
cd /workspace/xuannv
EVAL_DIR=/workspace/xuannv/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation
CKPT=/workspace/xuannv/outputs/exp_v2_D_7target_7card_100ep_0521/epoch_best_epoch29.pt
CONFIG=configs/xuannv_v2_expD_7target_lowrecon.yaml

echo "=========================================="
echo "ExpD 验证 Pipeline 开始"
echo "Checkpoint: $CKPT"
echo "设备: npu:0 (物理 NPU 7)"
echo "开始时间: $(date)"
echo "=========================================="

# Step 1: 提取 Embedding
echo ""
echo "[Step 1/4] 提取 Embedding..."
echo "=========================================="
python scripts/eval/extract_embeddings_v2.py \
    --config $CONFIG \
    --checkpoint $CKPT \
    --output-dir $EVAL_DIR/embeddings \
    --device npu:0 \
    --batch-size 4 \
    --save-every 500 \
    > $EVAL_DIR/expD_step1_extract.log 2>&1

echo ""
echo "[Step 1/4] 完成! Embedding 文件: $EVAL_DIR/embeddings/patch_embeddings.npz"

# Step 2: Bare CD AUC
echo ""
echo "[Step 2/4] 变化检测 Bare AUC..."
echo "=========================================="
python scripts/eval/evaluate_cd_v2.py \
    --embedding-file $EVAL_DIR/embeddings/patch_embeddings.npz \
    --output-dir $EVAL_DIR/change_detection \
    > $EVAL_DIR/expD_step2_cd_auc.log 2>&1

echo ""
echo "[Step 2/4] 完成!"

# Step 3: MLP 下游任务
echo ""
echo "[Step 3/4] MLP 下游任务..."
echo "=========================================="
python scripts/eval/evaluate_mlp_v2.py \
    --embedding-file $EVAL_DIR/embeddings/patch_embeddings.npz \
    --output-dir $EVAL_DIR/mlp_downstream \
    --device npu:0 \
    --epochs 50 \
    > $EVAL_DIR/expD_step3_mlp.log 2>&1

echo ""
echo "[Step 3/4] 完成!"

# Step 4: KNN 下游任务（基于 embedding 文件）
echo ""
echo "[Step 4/4] KNN 下游任务..."
echo "=========================================="
python3 << 'PYEOF'
import sys, os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

EVAL_DIR = Path("/workspace/xuannv/outputs/exp_v2_D_7target_7card_100ep_0521/evaluation")
DATA_ROOT = Path("/workspace/xuannv/data_raw/harbin/scenes")

# 加载 embedding
print("[KNN] 加载 embedding...")
data = np.load(EVAL_DIR / "embeddings" / "patch_embeddings.npz")
global_mean = data["global_mean"]
patch_ids = data["patch_ids"]
months = data["months"]
print(f"[KNN] 形状: global_mean={global_mean.shape}, patches={len(patch_ids)}")

TASKS = [
    ("worldcover", "worldcover", "static.tif", 10),
    ("jrc_water", "jrc_water", "static.tif", 2),
    ("dynamic_world", "dynamic_world", "2025Q2.tif", 9),
]

LABEL_MAPPINGS = {
    "worldcover": {10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 6},
}

def load_label(patch_id, label_dir, fname):
    path = DATA_ROOT / label_dir / patch_id / fname
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        label = src.read(1)
        nodata = src.nodata
    return label, nodata

def resize_label(label, target_h, target_w):
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(label_t, size=(target_h, target_w), mode="nearest")
    return resized.squeeze().numpy().astype(label.dtype)

month_idx = 5
spatial_maps = data["spatial_maps"]
N, _, D, H, W = spatial_maps.shape

results = {}

for task_name, label_dir, label_file, num_classes in TASKS:
    print(f"\n[KNN] 任务: {task_name}")
    
    all_embs = []
    all_labels = []
    
    for i, pid in enumerate(patch_ids):
        pid_str = str(pid)
        label, nodata = load_label(pid_str, label_dir, label_file)
        if label is None:
            continue
        
        emb = spatial_maps[i, month_idx]
        
        if label.shape != (H, W):
            label = resize_label(label, H, W)
        
        if task_name in LABEL_MAPPINGS:
            mapping = LABEL_MAPPINGS[task_name]
            label = np.vectorize(lambda x: mapping.get(x, x))(label)
        
        emb_flat = emb.reshape(D, -1).T
        label_flat = label.flatten()
        
        if nodata is not None:
            mask = label_flat != nodata
        else:
            mask = label_flat >= 0
        
        if task_name == "dynamic_world":
            mask = mask & (label_flat > 0) & (label_flat <= num_classes)
        
        all_embs.append(emb_flat[mask])
        all_labels.append(label_flat[mask])
    
    if len(all_embs) == 0:
        print(f"  [KNN] 无有效数据，跳过")
        continue
    
    X = np.concatenate(all_embs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"  [KNN] 样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
    
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    n = len(X_norm)
    split = int(n * 0.8)
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y[:split], y[split:]
    
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    results[task_name] = {
        "accuracy": float(acc),
        "num_samples": int(n),
        "num_test": int(len(y_test)),
    }
    
    print(f"  [KNN] 准确率: {acc:.4f}")

import json
with open(EVAL_DIR / "knn_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n[KNN] 所有任务完成!")
print(json.dumps(results, indent=2))
PYEOF

echo ""
echo "[Step 4/4] 完成!"

echo ""
echo "=========================================="
echo "ExpD 验证 Pipeline 全部完成!"
echo "结束时间: $(date)"
echo "结果目录: $EVAL_DIR"
echo "=========================================="
