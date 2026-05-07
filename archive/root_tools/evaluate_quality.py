#!/usr/bin/env python3
"""AEF_qwen 模型质量验证实验.

评估:
1. Embedding 均匀性 - 是否在球面上均匀分布
2. Embedding 相似度矩阵 - 是否坍缩
3. Before/After 差异 - 不同时间窗口是否产生不同 embedding
4. 空间连续性 - 相邻位置是否相似
5. Patch 间区分度 - 不同地点是否可区分
"""
import sys
sys.path.insert(0, '/workspace/xuannv')

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/embedding_maps.npy"
IDS_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/patch_ids.json"

def load_data():
    emb = np.load(EMB_PATH)
    import json
    with open(IDS_PATH) as f:
        ids = json.load(f)
    return emb, ids

def experiment_1_uniformity(emb):
    """1. Embedding 均匀性测试."""
    print("\n" + "="*60)
    print("实验 1: Embedding 均匀性")
    print("="*60)
    
    # 随机采样 2000 个像素
    N, D, H, W = emb.shape
    flat = emb.reshape(N*D, -1)  # wrong shape - should be (N*H*W, D)
    flat = emb.reshape(N, D, -1).transpose(0, 2, 1).reshape(-1, D)  # [N*H*W, D]
    np.random.seed(42)
    idx = np.random.choice(len(flat), min(2000, len(flat)), replace=False)
    sample = flat[idx]
    
    # 已经是 L2 norm 的
    norms = np.linalg.norm(sample, axis=1)
    print(f"  样本量: {len(sample)}")
    print(f"  L2 norm 范围: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  L2 norm 均值: {norms.mean():.4f}")
    
    # 成对余弦相似度
    n = min(500, len(sample))
    sub = sample[:n]
    cos_sim = cosine_similarity(sub)
    
    # 去掉对角线
    mask = ~np.eye(n, dtype=bool)
    off_diag = cos_sim[mask]
    
    print(f"  成对余弦相似度:")
    print(f"    均值: {off_diag.mean():.4f}")
    print(f"    中位数: {np.median(off_diag):.4f}")
    print(f"    P5: {np.percentile(off_diag, 5):.4f}")
    print(f"    P95: {np.percentile(off_diag, 95):.4f}")
    print(f"    坍缩判定: 均值 > 0.5 → 坍缩; 均值 < 0.05 → 良好")
    
    if off_diag.mean() > 0.5:
        print(f"    ❌ 坍缩! 所有 embedding 指向相似方向")
    elif off_diag.mean() > 0.2:
        print(f"    ⚠️  部分坍缩，多样性不足")
    else:
        print(f"    ✅ 良好! Embedding 在球面上均匀分布")
    
    return off_diag.mean()

def experiment_2_before_after(model, dataset, device):
    """2. Before/After 时间敏感性测试."""
    print("\n" + "="*60)
    print("实验 2: Before/After 时间敏感性")
    print("="*60)
    
    from demo.embedding_engine import compute_before_after, cosine_distance_map
    
    # 选取 10 个有足够数据源的 patch
    test_patches = []
    for i in range(min(10, len(dataset.patches))):
        pm_source = dataset._load_input_frames(dataset.patches[i], "s2")
        if len(pm_source[0]) > 0:
            test_patches.append(i)
    
    if not test_patches:
        print("  ❌ 没有可用的测试 patch")
        return
    
    time_pairs = [
        ("2023 全窗口", "2024 全窗口", 
         1672531200000.0, 1703980800000.0,
         1704067200000.0, 1735603200000.0),
        ("2023-2024 中", "2025 中",
         1680307200000.0, 1719792000000.0,
         1735689600000.0, 1767225600000.0),
        ("2023Q1-Q2", "2025Q1-Q2",
         1672531200000.0, 1688169600000.0,
         1735689600000.0, 1751328000000.0),
    ]
    
    for name_b, name_a, bs, be, aes, aee in time_pairs:
        print(f"\n  时间窗口对: {name_b} vs {name_a}")
        cos_dists = []
        for i, pidx in enumerate(test_patches[:5]):
            try:
                emb_b, emb_a = compute_before_after(
                    model, dataset, pidx, (bs, be), (aes, aee), device)
                
                # Per-pixel cosine distance
                cd = cosine_distance_map(emb_b, emb_a)
                cos_dists.append(cd.mean())
            except Exception as e:
                cos_dists.append(-1)
        
        valid = [x for x in cos_dists if x >= 0]
        if valid:
            print(f"    平均 cosine distance: {np.mean(valid):.4f} (std={np.std(valid):.4f})")
            if np.mean(valid) < 0.01:
                print(f"    ❌ 无差异 - 模型对时间不敏感!")
            elif np.mean(valid) < 0.05:
                print(f"    ⚠️  差异较小 - 时间信号微弱")
            else:
                print(f"    ✅ 有显著差异 - 模型能区分不同时间")
        else:
            print(f"    ❌ 所有 patch 都失败")

def experiment_3_spatial_continuity(emb):
    """3. 空间连续性测试."""
    print("\n" + "="*60)
    print("实验 3: 空间连续性")
    print("="*60)
    
    # 计算单个 patch 内部相邻像素的余弦相似度
    patch_emb = emb[0]  # [D, H, W]
    D, H, W = patch_emb.shape
    
    # 水平相邻
    left = patch_emb[:, :, :-1].reshape(D, -1).T  # [H*(W-1), D]
    right = patch_emb[:, :, 1:].reshape(D, -1).T
    
    cos_sim_h = F.cosine_similarity(
        torch.from_numpy(left), torch.from_numpy(right), dim=1
    ).numpy()
    
    # 垂直相邻
    top = patch_emb[:, :-1, :].reshape(D, -1).T
    bottom = patch_emb[:, 1:, :].reshape(D, -1).T
    
    cos_sim_v = F.cosine_similarity(
        torch.from_numpy(top), torch.from_numpy(bottom), dim=1
    ).numpy()
    
    print(f"  水平相邻像素余弦相似度: mean={cos_sim_h.mean():.4f}, std={cos_sim_h.std():.4f}")
    print(f"  垂直相邻像素余弦相似度: mean={cos_sim_v.mean():.4f}, std={cos_sim_v.std():.4f}")
    print(f"  解读: 高相似度(>0.8)→空间连续性好; 低相似度→可能有噪声")
    
    if cos_sim_h.mean() > 0.7 and cos_sim_v.mean() > 0.7:
        print(f"  ✅ 空间连续性好 - embedding 平滑过渡")
    elif cos_sim_h.mean() > 0.4:
        print(f"  ⚠️  空间连续性一般")
    else:
        print(f"  ❌ 空间连续性差 - embedding 过于嘈杂")

def experiment_4_patch_discriminability(emb):
    """4. Patch 间区分度测试."""
    print("\n" + "="*60)
    print("实验 4: Patch 间区分度")
    print("="*60)
    
    N, D, H, W = emb.shape
    # 每个 patch 的全局平均 embedding
    global_embs = emb.mean(axis=(2, 3))  # [N, D]
    global_embs = global_embs / np.linalg.norm(global_embs, axis=1, keepdims=True)
    
    # 计算所有 patch 间的余弦相似度
    cos_sim = global_embs @ global_embs.T
    
    # 去掉对角线
    mask = ~np.eye(N, dtype=bool)
    off_diag = cos_sim[mask]
    
    print(f"  Patch 数量: {N}")
    print(f"  Patch 间余弦相似度:")
    print(f"    均值: {off_diag.mean():.4f}")
    print(f"    中位数: {np.median(off_diag):.4f}")
    print(f"    最大: {off_diag.max():.4f}")
    print(f"    最小: {off_diag.min():.4f}")
    
    # 最相似的 patch 对
    max_idx = np.argmax(off_diag)
    # 找对应索引
    flat_upper = cos_sim[np.triu_indices(N, k=1)]
    max_flat = np.argmax(flat_upper)
    r, c = np.triu_indices(N, k=1)
    print(f"    最相似 patch 对: [{r[max_flat]}] vs [{c[max_flat]}], sim={flat_upper[max_flat]:.4f}")
    
    if off_diag.mean() > 0.9:
        print(f"  ❌ 几乎所有 patch 都极度相似 - 模型坍缩!")
    elif off_diag.mean() > 0.5:
        print(f"  ⚠️  Patch 区分度不足")
    else:
        print(f"  ✅ Patch 区分度良好")
    
    # PCA 可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(global_embs)
    print(f"  PCA 解释方差比: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    
    return off_diag.mean()

def experiment_5_reconstruction_quality():
    """5. 训练日志分析."""
    print("\n" + "="*60)
    print("实验 5: 训练质量回顾")
    print("="*60)
    
    log_path = "/workspace/logs/qwen_v1_train_v3.log"
    try:
        with open(log_path) as f:
            lines = f.readlines()
        
        final_line = None
        for line in reversed(lines):
            if "Epoch 400/" in line and "Recon" in line:
                final_line = line
                break
        
        if final_line:
            import re
            recon = re.search(r'Recon:\s*([\d.]+)', final_line)
            runif = re.search(r'RawUnif:\s*([-.\d]+)', final_line)
            punif = re.search(r'PreUnif:\s*([-.\d]+)', final_line)
            
            print(f"  Epoch 400 最终指标:")
            print(f"    Reconstruction:  {recon.group(1)} (越低越好, 目标 <2.5)")
            print(f"    RawUnif:         {runif.group(1)} (越负越好, 目标 <-3.0)")
            print(f"    PreUnif:         {punif.group(1)} (越负越好, 目标 <-3.0)")
            
            r = float(recon.group(1))
            u = float(runif.group(1))
            p = float(punif.group(1))
            
            score = 0
            if r < 2.5: score += 1
            if u < -3.0: score += 1
            if p < -3.0: score += 1
            print(f"    综合评分: {score}/3 {'✅ 合格' if score >= 2 else '❌ 不足'}")
    except Exception as e:
        print(f"  无法读取训练日志: {e}")

def main():
    print("=" * 60)
    print(" AEF_qwen V1 模型质量验证")
    print("=" * 60)
    
    emb, ids = load_data()
    print(f"加载 embedding maps: {emb.shape}")
    print(f"Patch 数量: {len(ids)}")
    
    # 实验 1: 均匀性
    exp1_score = experiment_1_uniformity(emb)
    
    # 实验 3: 空间连续性
    experiment_3_spatial_continuity(emb)
    
    # 实验 4: Patch 区分度
    exp4_score = experiment_4_patch_discriminability(emb)
    
    # 实验 5: 训练质量
    experiment_5_reconstruction_quality()
    
    # 实验 2: Before/After (需要模型)
    print("\n" + "="*60)
    print("实验 2: Before/After 时间敏感性 (需要加载模型)")
    print("="*60)
    print("  正在加载模型...")
    try:
        device = torch.device("npu:5" if torch.npu.is_available() else "cpu")
        from demo.embedding_engine import load_model
        model, dataset, cfg = load_model(
            "qwen_v1 (3-input, 7-target)", str(device))
        experiment_2_before_after(model, dataset, device)
    except Exception as e:
        print(f"  ⚠️  模型加载失败: {e}")
        print(f"  跳过时间敏感性实验")
    
    print("\n" + "="*60)
    print(" 总结")
    print("="*60)
    print(f"  1. 嵌入均匀性:    {'✅' if exp1_score < 0.2 else '❌'} (均值={exp1_score:.4f})")
    print(f"  2. Patch区分度:   {'✅' if exp4_score < 0.5 else '❌'} (均值={exp4_score:.4f})")
    print(f"  3. 训练收敛:      查看上面训练质量指标")
    print("="*60)

if __name__ == "__main__":
    main()
