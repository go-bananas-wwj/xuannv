#!/usr/bin/env python
"""海淀数据集快速 Embedding 质量评估。

评估指标：
1. erank — embedding 有效秩（目标 > 8）
2. 时间区分能力 — 同 patch 不同月份 cosine sim 分布
3. 跨 patch 区分能力 — 不同 patch 之间的 cosine sim 分布
"""
from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def compute_erank(x: torch.Tensor) -> float:
    """计算有效秩 erank = exp(H(S/sum(S)))。"""
    with torch.no_grad():
        x_c = x - x.mean(0, keepdim=True)
        try:
            S = torch.linalg.svdvals(x_c)
        except Exception:
            S = torch.svd(x_c, compute_uv=False).S
        S = S.clamp(min=1e-8)
        p = S / S.sum()
        H = -(p * p.log()).sum()
        return float(H.exp())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_haidian_v1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-patches", type=int, default=40, help="评估的 patch 数量")
    args = parser.parse_args()

    print(f"[评估] config={args.config}")
    print(f"[评估] checkpoint={args.checkpoint}")
    print(f"[评估] device={args.device}")

    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    cfg = load_config(args.config)
    cfg.data.preload = False

    device = torch.device(args.device)

    print("[加载] 数据集...")
    dataset = HarbinPatchDataset(cfg=cfg)
    patch_ids = list({pid for pid, _, _ in dataset.monthly_samples})
    patch_ids.sort()
    print(f"[信息] 总 patch 数: {len(patch_ids)}")

    print("[加载] 模型...")
    model = AEFModel(cfg=cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ── 提取 embedding ────────────────────────────────────────────────────────
    sample_map = {
        (pid, y, m): idx
        for idx, (pid, y, m) in enumerate(dataset.monthly_samples)
    }

    import time as time_mod
    import calendar

    def month_to_window(year: int, month: int):
        start_s = int(time_mod.mktime((year, month, 1, 0, 0, 0, 0, 0, 0)))
        last_d = calendar.monthrange(year, month)[1]
        end_s = int(time_mod.mktime((year, month, last_d, 23, 59, 59, 0, 0, 0)))
        return start_s * 1000, end_s * 1000

    # 目标月份
    eval_months = [(2025, 4), (2025, 6), (2025, 8), (2025, 10)]

    # 选取前 num_patches 个 patch
    eval_patches = patch_ids[: args.num_patches]

    # patch_id → {month_str → embedding [D]}
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    extracted = 0

    print(f"[提取] {len(eval_patches)} patches × {len(eval_months)} months...")
    with torch.no_grad():
        for pid in eval_patches:
            month_embs = {}
            for year, month in eval_months:
                key = (pid, year, month)
                if key not in sample_map:
                    continue
                item = dataset[sample_map[key]]
                vs, ve = month_to_window(year, month)

                def _to(x):
                    return x.unsqueeze(0).to(device)

                try:
                    out = model(
                        source_frames        = _to(item["source_frames"]),
                        source_timestamps_ms = _to(item["source_timestamps_ms"]),
                        source_frame_mask    = _to(item["source_frame_mask"]),
                        source_input_mask    = _to(item["source_input_mask"]),
                        source_type_ids      = _to(item["source_type_ids"]),
                        valid_start_ms       = torch.tensor([vs], dtype=torch.int64, device=device),
                        valid_end_ms         = torch.tensor([ve], dtype=torch.int64, device=device),
                        target_relative_time = torch.zeros(1, cfg.data.num_target_sources, device=device),
                        target_metadata      = torch.zeros(1, cfg.data.num_target_sources,
                                                           cfg.data.metadata_dim, device=device),
                        skip_decoder=True,
                    )
                    emb = out.embedding  # [1, D]
                    emb_np = F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().numpy()
                    month_embs[f"{year}-{month:02d}"] = emb_np
                    extracted += 1
                except Exception as e:
                    print(f"  [警告] {pid} {year}-{month:02d}: {e}")
            if month_embs:
                embeddings[pid] = month_embs

    print(f"[完成] 提取 {extracted} 个 embedding")
    print()

    # ── 分析 1: erank ─────────────────────────────────────────────────────────
    all_embs = []
    for pid, months in embeddings.items():
        for m, emb in months.items():
            all_embs.append(emb)
    all_embs = np.stack(all_embs, axis=0)  # [N, D]

    erank_val = compute_erank(torch.from_numpy(all_embs).float())
    print(f"[erank] 所有 {len(all_embs)} 个 embedding 的 erank = {erank_val:.2f}")
    print(f"        (目标 > 8.0，当前 {'✅ 正常' if erank_val > 8 else '❌ 坍缩'})")
    print()

    # ── 分析 2: 同 patch 不同月份相似度 ────────────────────────────────────────
    intra_sims = []
    inter_sims = []

    patch_list = list(embeddings.keys())
    for i, pid in enumerate(patch_list):
        months = list(embeddings[pid].values())
        if len(months) < 2:
            continue
        # 同 patch 不同月份
        for j in range(len(months)):
            for k in range(j + 1, len(months)):
                sim = float(np.dot(months[j], months[k]))
                intra_sims.append(sim)

        # 不同 patch（取少量样本）
        if i < 5:
            for other_pid in patch_list[i + 1: i + 5]:
                if other_pid not in embeddings:
                    continue
                other_months = list(embeddings[other_pid].values())
                if not other_months:
                    continue
                sim = float(np.dot(months[0], other_months[0]))
                inter_sims.append(sim)

    if intra_sims:
        print(f"[时间一致性] 同 patch 不同月份 cosine sim:")
        print(f"  mean={np.mean(intra_sims):.4f}  std={np.std(intra_sims):.4f}"
              f"  min={np.min(intra_sims):.4f}  max={np.max(intra_sims):.4f}")
    if inter_sims:
        print(f"[跨 patch]   不同 patch cosine sim:")
        print(f"  mean={np.mean(inter_sims):.4f}  std={np.std(inter_sims):.4f}"
              f"  min={np.min(inter_sims):.4f}  max={np.max(inter_sims):.4f}")

    if intra_sims and inter_sims:
        gap = np.mean(intra_sims) - np.mean(inter_sims)
        print(f"\n[区分能力] intra - inter gap = {gap:.4f}")
        if gap < 0.1:
            print("  ❌ 区分能力差（intra ≈ inter，模型无法区分 patch）")
        elif gap < 0.3:
            print("  ⚠️  区分能力一般")
        else:
            print("  ✅ 区分能力良好")

    print()

    # ── 分析 3: 时间变化检测模拟 ──────────────────────────────────────────────
    # 用最早月份 vs 最晚月份的 cosine distance 作为变化分数
    # 所有 patch（假设都有变化/无变化是随机的）
    month_keys = [f"{y}-{m:02d}" for y, m in eval_months]
    first_key  = month_keys[0]
    last_key   = month_keys[-1]

    change_scores = []
    for pid, months in embeddings.items():
        if first_key in months and last_key in months:
            a = months[first_key]
            b = months[last_key]
            # cosine distance = 1 - cosine_sim
            dist = 1.0 - float(np.dot(a, b))
            change_scores.append((pid, dist))

    if change_scores:
        dists = [d for _, d in change_scores]
        print(f"[时序变化] {first_key} → {last_key} cosine distance:")
        print(f"  mean={np.mean(dists):.4f}  std={np.std(dists):.4f}")
        print(f"  min={np.min(dists):.4f}  max={np.max(dists):.4f}")
        # 如果大多数 dist ≈ 0，说明模型对时间不敏感
        if np.mean(dists) < 0.05:
            print("  ❌ 时间不敏感（embedding 不随时间变化）")
        elif np.mean(dists) < 0.15:
            print("  ⚠️  时间敏感性一般")
        else:
            print("  ✅ 时间敏感性良好")

    print("\n" + "=" * 60)
    print("评估总结:")
    print(f"  erank = {erank_val:.2f} ({'✅' if erank_val > 8 else '❌ 坍缩'})")
    if intra_sims and inter_sims:
        print(f"  intra/inter gap = {gap:.4f} ({'✅' if gap > 0.3 else '⚠️' if gap > 0.1 else '❌'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
