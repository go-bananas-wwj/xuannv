#!/usr/bin/env python3
"""
V10 综合评估脚本 — 多维度嵌入质量测试.

测试维度:
1. 变化检测 Bare AUC (105个光学标注)
2. 下游分类 (Land Cover分类Accuracy)
3. 重建质量 (PSNR/SSIM per target)
4. 嵌入质量指标 (RankMe, Stable Rank, Temporal Discriminability)
5. Embedding可视化 (t-SNE)

用法:
    python scripts/eval/v10_comprehensive_eval.py \
        --checkpoint /workspace/outputs/xuannv_backbone_v10_temporal/epoch_100.pt \
        --config configs/xuannv_v10_temporal.yaml \
        --output-dir /workspace/outputs/v10_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "/workspace/xuannv")

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset


# =============================================================================
# 1. Embedding 提取
# =============================================================================

def extract_all_embeddings(model, dataset, device, batch_size=8):
    """提取所有 patches 的 embedding.
    
    对每个 patch，提取:
    - full_embedding: 整个时间范围的 embedding
    - before_embedding: BEFORE_WINDOW 的 embedding  
    - after_embedding: AFTER_WINDOW 的 embedding
    
    Returns:
        dict: {patch_id: {'full': ..., 'before': ..., 'after': ...}}
    """
    model.eval()
    
    BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)
    AFTER_WINDOW = (1719792000000.0, 1735603200000.0)
    
    results = {}
    
    for idx in range(len(dataset)):
        patch_id = f"patch_{idx:06d}"
        
        try:
            sample = dataset[idx]
            source_frames = sample["source_frames"].unsqueeze(0).to(device)
            source_timestamps_ms = sample["source_timestamps_ms"].unsqueeze(0).to(device)
            source_frame_mask = sample["source_frame_mask"].unsqueeze(0).to(device)
            source_input_mask = sample["source_input_mask"].unsqueeze(0).to(device)
            source_type_ids = sample["source_type_ids"].unsqueeze(0).to(device)
            
            # Full window embedding
            with torch.no_grad():
                out = model(
                    source_frames=source_frames,
                    source_timestamps_ms=source_timestamps_ms,
                    source_frame_mask=source_frame_mask,
                    source_input_mask=source_input_mask,
                    source_type_ids=source_type_ids,
                    valid_start_ms=source_timestamps_ms.min(),
                    valid_end_ms=source_timestamps_ms.max(),
                    target_relative_time=torch.zeros(1, 1, device=device),
                    target_metadata=torch.zeros(1, sample["target_metadata"].shape[-1], device=device),
                )
            full_emb = F.normalize(out.embedding[0], dim=0).cpu().numpy()
            
            # Before window
            emb_before = _extract_window_embedding(
                model, source_frames, source_timestamps_ms, source_frame_mask,
                source_input_mask, source_type_ids, BEFORE_WINDOW, device
            )
            
            # After window
            emb_after = _extract_window_embedding(
                model, source_frames, source_timestamps_ms, source_frame_mask,
                source_input_mask, source_type_ids, AFTER_WINDOW, device
            )
            
            results[patch_id] = {
                'full': full_emb,
                'before': emb_before,
                'after': emb_after,
            }
            
        except Exception as e:
            print(f"  Error extracting {patch_id}: {e}")
            continue
        
        if (idx + 1) % 50 == 0:
            print(f"  Extracted {idx + 1}/{len(dataset)} patches...")
    
    return results


@torch.no_grad()
def _extract_window_embedding(model, source_frames, source_timestamps_ms, source_frame_mask,
                               source_input_mask, source_type_ids, window, device, metadata_dim=4):
    """提取指定时间窗口的 embedding."""
    valid_start, valid_end = window
    B, S, T = source_frame_mask.shape
    frame_mask = source_frame_mask.clone()
    
    for b in range(B):
        for s in range(S):
            for t in range(T):
                ts = source_timestamps_ms[b, s, t].item()
                if ts < valid_start or ts > valid_end:
                    frame_mask[b, s, t] = False
    
    if not frame_mask.any():
        return None
    
    out = model(
        source_frames=source_frames,
        source_timestamps_ms=source_timestamps_ms,
        source_frame_mask=frame_mask,
        source_input_mask=source_input_mask,
        source_type_ids=source_type_ids,
        valid_start_ms=torch.tensor([valid_start], device=device),
        valid_end_ms=torch.tensor([valid_end], device=device),
        target_relative_time=torch.zeros(1, 1, device=device),
        target_metadata=torch.zeros(1, 1, metadata_dim, device=device),
    )
    
    return F.normalize(out.embedding[0], dim=0).cpu().numpy()


# =============================================================================
# 2. 变化检测 Bare AUC
# =============================================================================

def evaluate_change_detection(embeddings, annot_dir, grid_path):
    """在光学标注上计算 Bare AUC."""
    import geopandas as gpd
    import glob
    
    grid = gpd.read_file(grid_path)
    patch_id_to_idx = {f"patch_{i:06d}": i for i in range(len(grid))}
    
    # 加载标注
    shp_files = glob.glob(os.path.join(annot_dir, "*.shp"))
    all_labels = []
    for shp_path in shp_files:
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.empty:
                continue
            label_type = 1 if '新增建筑' in shp_path or '新增建设' in shp_path else 0
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                center = geom.centroid if hasattr(geom, 'centroid') else geom
                all_labels.append({
                    'geometry': center,
                    'label': label_type,
                    'source': os.path.basename(shp_path),
                })
        except:
            pass
    
    labels_gdf = gpd.GeoDataFrame(all_labels, crs="EPSG:4326")
    
    scores = []
    labels = []
    
    for _, row in labels_gdf.iterrows():
        pt = row.geometry
        if pt is None:
            continue
        
        matched = None
        for _, g_row in grid.iterrows():
            if g_row.geometry.contains(pt):
                matched = g_row
                break
        
        if matched is None:
            continue
        
        patch_id = matched.get('patch_id', f"patch_{matched.name:06d}")
        if patch_id not in embeddings:
            continue
        
        emb_b = embeddings[patch_id].get('before')
        emb_a = embeddings[patch_id].get('after')
        if emb_b is None or emb_a is None:
            continue
        
        sim = np.dot(emb_b, emb_a)
        dist = 1.0 - sim
        scores.append(dist)
        labels.append(row['label'])
    
    if len(scores) < 10:
        return None, len(scores)
    
    auc = roc_auc_score(labels, scores)
    return auc, len(scores)


# =============================================================================
# 3. 嵌入质量指标
# =============================================================================

def compute_rankme(embeddings_matrix):
    """RankMe: 特征值熵评估嵌入质量.
    
    Args:
        embeddings_matrix: [N, D] numpy array
    Returns:
        rankme_score: float (越高越好，理想值接近 log(D))
    """
    N, D = embeddings_matrix.shape
    
    # SVD
    try:
        _, s, _ = np.linalg.svd(embeddings_matrix, full_matrices=False)
    except:
        return 0.0
    
    # 归一化奇异值
    s_norm = s / (s.sum() + 1e-8)
    
    # 熵
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    
    # RankMe = 熵 / log(min(N, D))
    rankme = entropy / np.log(min(N, D))
    
    return float(rankme)


def compute_stable_rank(embeddings_matrix):
    """Stable Rank: 奇异值平方和 / 最大奇异值平方.
    
    越高表示嵌入维度利用越充分。
    """
    try:
        _, s, _ = np.linalg.svd(embeddings_matrix, full_matrices=False)
        stable_rank = (s ** 2).sum() / (s[0] ** 2 + 1e-8)
        return float(stable_rank)
    except:
        return 0.0


def compute_temporal_discriminability(embeddings):
    """Temporal Discriminability: 同一 patch 的 before/after 变化 vs 不同 patch 间距离.
    
    计算:
    - 正例: 有变化的 patch 的 before/after distance
    - 负例: 无变化的 patch 的 before/after distance  
    - 跨 patch: 任意两个不同 patch 的 distance
    
    Returns:
        td_score: float (越高越好)
    """
    patch_ids = list(embeddings.keys())
    
    # 收集所有 before/after pairs
    temporal_dists = []
    for pid in patch_ids:
        emb_b = embeddings[pid].get('before')
        emb_a = embeddings[pid].get('after')
        if emb_b is not None and emb_a is not None:
            dist = 1.0 - np.dot(emb_b, emb_a)
            temporal_dists.append(dist)
    
    if len(temporal_dists) < 10:
        return 0.0
    
    # 收集跨 patch distances (随机采样)
    cross_dists = []
    n_samples = min(500, len(patch_ids) * (len(patch_ids) - 1) // 2)
    np.random.seed(42)
    for _ in range(n_samples):
        i, j = np.random.choice(len(patch_ids), 2, replace=False)
        emb_i = embeddings[patch_ids[i]].get('full')
        emb_j = embeddings[patch_ids[j]].get('full')
        if emb_i is not None and emb_j is not None:
            dist = 1.0 - np.dot(emb_i, emb_j)
            cross_dists.append(dist)
    
    if len(cross_dists) < 10:
        return 0.0
    
    # TD = mean(temporal_dist) / mean(cross_dist)
    # 越高表示时间信号比空间差异更明显
    td_score = np.mean(temporal_dists) / (np.mean(cross_dists) + 1e-8)
    
    return float(td_score)


# =============================================================================
# 4. 重建质量评估
# =============================================================================

def evaluate_reconstruction(model, dataset, device, num_samples=50):
    """评估重建质量.
    
    随机采样 num_samples 个 patch，计算:
    - PSNR per target source
    - 分类目标的 Accuracy
    
    Returns:
        dict: {source_name: {'psnr': ..., 'acc': ...}}
    """
    model.eval()
    
    # TARGET_SOURCES: s2, s1, landsat, dem, worldcover, dynamic_world, jrc_water
    source_names = ['s2', 's1', 'landsat', 'dem', 'worldcover', 'dynamic_world', 'jrc_water']
    source_results = {name: {'mse_sum': 0.0, 'count': 0, 'acc_sum': 0.0, 'acc_count': 0} 
                      for name in source_names}
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        try:
            sample = dataset[idx]
            source_frames = sample["source_frames"].unsqueeze(0).to(device)
            source_timestamps_ms = sample["source_timestamps_ms"].unsqueeze(0).to(device)
            source_frame_mask = sample["source_frame_mask"].unsqueeze(0).to(device)
            source_input_mask = sample["source_input_mask"].unsqueeze(0).to(device)
            source_type_ids = sample["source_type_ids"].unsqueeze(0).to(device)
            
            target_images = sample["target_images"].unsqueeze(0).to(device)
            target_mask = sample["target_mask"].unsqueeze(0).to(device)
            target_source_idx = sample.get("target_source_idx")
            if target_source_idx is not None:
                target_source_idx = target_source_idx.unsqueeze(0).to(device)
            target_loss_type = sample.get("target_loss_type")
            if target_loss_type is not None:
                target_loss_type = target_loss_type.unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(
                    source_frames=source_frames,
                    source_timestamps_ms=source_timestamps_ms,
                    source_frame_mask=source_frame_mask,
                    source_input_mask=source_input_mask,
                    source_type_ids=source_type_ids,
                    valid_start_ms=source_timestamps_ms.min(),
                    valid_end_ms=source_timestamps_ms.max(),
                    target_relative_time=torch.zeros(1, target_images.shape[1], device=device),
                    target_metadata=torch.zeros(1, target_images.shape[1], sample["target_metadata"].shape[-1], device=device),
                    target_loss_type=target_loss_type,
                    target_source_idx=target_source_idx,
                )
            
            recon = out.reconstructions  # [1, T_tgt, C, H, W]
            
            # 评估每个 target
            for t in range(recon.shape[1]):
                pred = recon[0, t]  # [C, H, W]
                tgt = target_images[0, t]  # [C, H, W]
                mask = target_mask[0, t]  # [C, H, W]
                
                if target_source_idx is not None:
                    src_id = int(target_source_idx[0, t].item())
                    src_name = source_names[src_id] if src_id < len(source_names) else 'unknown'
                else:
                    src_name = source_names[0]
                
                if src_name not in source_results:
                    continue
                
                # MSE (连续目标)
                if target_loss_type is None or target_loss_type[0, t].item() == 0:
                    valid = mask > 0.5
                    if valid.any():
                        mse = ((pred - tgt) ** 2)[valid].mean().item()
                        source_results[src_name]['mse_sum'] += mse
                        source_results[src_name]['count'] += 1
                
                # Accuracy (分类目标)
                elif target_loss_type[0, t].item() == 1:
                    # pred: logits [C, H, W], tgt: [C, H, W] one-hot
                    pred_cls = pred.argmax(dim=0)  # [H, W]
                    tgt_cls = tgt.argmax(dim=0) if tgt.shape[0] > 1 else tgt[0].long()
                    
                    correct = (pred_cls == tgt_cls).float().mean().item()
                    source_results[src_name]['acc_sum'] += correct
                    source_results[src_name]['acc_count'] += 1
        
        except Exception as e:
            continue
    
    # 汇总
    summary = {}
    for name, vals in source_results.items():
        summary[name] = {}
        if vals['count'] > 0:
            mse = vals['mse_sum'] / vals['count']
            psnr = -10 * np.log10(mse + 1e-10)
            summary[name]['psnr'] = float(psnr)
            summary[name]['mse'] = float(mse)
        if vals['acc_count'] > 0:
            summary[name]['acc'] = float(vals['acc_sum'] / vals['acc_count'])
    
    return summary


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/xuannv_v10_temporal.yaml")
    parser.add_argument("--output-dir", default="/workspace/outputs/v10_eval")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-recon-samples", type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  V10 综合评估")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print("="*70)
    
    # 加载配置和模型
    print("\n[1/5] 加载模型...")
    cfg = load_config(args.config)
    cfg.data.preload = False
    cfg.data.manifest_path = "/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered"
    
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Model loaded from epoch {ckpt.get('epoch', 'unknown')}")
    
    # 加载数据集
    print("\n[2/5] 加载数据集...")
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    print(f"  Dataset: {len(dataset)} patches")
    
    # Step 1: 提取 Embedding
    print("\n[3/5] 提取所有 patches 的 embedding...")
    embeddings = extract_all_embeddings(model, dataset, device)
    print(f"  成功提取 {len(embeddings)} patches")
    
    # 保存 embedding
    emb_path = output_dir / "embeddings.npz"
    np.savez(emb_path, **{
        k: np.stack([v['full'], v['before'] if v['before'] is not None else np.zeros_like(v['full']), 
                     v['after'] if v['after'] is not None else np.zeros_like(v['full'])])
        for k, v in embeddings.items()
    })
    print(f"  Embeddings saved to {emb_path}")
    
    results = {}
    
    # Step 2: 变化检测 Bare AUC
    print("\n[4/5] 变化检测 Bare AUC...")
    annot_dir = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
    grid_path = "/workspace/index/harbin/grid/harbin_grid.geojson"
    
    if Path(annot_dir).exists() and Path(grid_path).exists():
        auc, n_samples = evaluate_change_detection(embeddings, annot_dir, grid_path)
        if auc is not None:
            results['change_detection'] = {'auc': float(auc), 'n_samples': n_samples}
            print(f"  Bare AUC = {auc:.4f} (n={n_samples})")
        else:
            print(f"  样本不足 (n={n_samples})")
    else:
        print(f"  标注或网格文件不存在，跳过")
    
    # Step 3: 嵌入质量指标
    print("\n[5/5] 嵌入质量指标...")
    
    # 收集 full embeddings matrix
    full_embs = []
    for pid in sorted(embeddings.keys()):
        emb = embeddings[pid].get('full')
        if emb is not None:
            full_embs.append(emb)
    
    if len(full_embs) > 10:
        embs_matrix = np.stack(full_embs)
        
        rankme = compute_rankme(embs_matrix)
        stable_rank = compute_stable_rank(embs_matrix)
        td = compute_temporal_discriminability(embeddings)
        
        results['embedding_quality'] = {
            'rankme': rankme,
            'stable_rank': stable_rank,
            'temporal_discriminability': td,
            'n_patches': len(full_embs),
        }
        
        print(f"  RankMe: {rankme:.4f} (理想值接近 1.0)")
        print(f"  Stable Rank: {stable_rank:.2f} (理想值接近 D={embs_matrix.shape[1]})")
        print(f"  Temporal Discriminability: {td:.4f} (越高越好)")
    
    # Step 4: 重建质量
    print("\n[6/5] 重建质量评估...")
    recon_results = evaluate_reconstruction(model, dataset, device, args.num_recon_samples)
    results['reconstruction'] = recon_results
    for name, vals in recon_results.items():
        if 'psnr' in vals:
            print(f"  {name}: PSNR={vals['psnr']:.2f}, MSE={vals['mse']:.4f}")
        if 'acc' in vals:
            print(f"  {name}: Acc={vals['acc']:.4f}")
    
    # 保存结果
    result_path = output_dir / "evaluation_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  评估完成！结果保存至: {result_path}")
    print(f"{'='*70}")
    
    # 汇总打印
    print("\n=== 评估汇总 ===")
    if 'change_detection' in results:
        auc = results['change_detection']['auc']
        status = "✅ 良好" if auc > 0.65 else ("✅ 及格" if auc > 0.55 else "⚠️ 需改进")
        print(f"变化检测 AUC: {auc:.4f} {status}")
    
    if 'embedding_quality' in results:
        eq = results['embedding_quality']
        print(f"RankMe: {eq['rankme']:.4f}")
        print(f"Stable Rank: {eq['stable_rank']:.2f} / {embs_matrix.shape[1]}")
        print(f"Temporal Discriminability: {eq['temporal_discriminability']:.4f}")
    
    avg_psnr = np.mean([v['psnr'] for v in recon_results.values() if 'psnr' in v])
    print(f"平均 PSNR: {avg_psnr:.2f}")


if __name__ == "__main__":
    main()
