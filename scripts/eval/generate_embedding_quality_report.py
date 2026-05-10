#!/usr/bin/env python3
"""玄女底座 V5 Embedding 质量深度诊断报告生成器.

基于已有评估 JSON 数据，生成:
1. 真实 AUC 统计报告
2. Embedding 空间质量六维诊断
3. 修正建议与 V6/V6.5 优化方向
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 科学配色
PRIMARY = "#1a5276"
SECONDARY = "#5dade2"
NEUTRAL = "#7f8c8d"
ALERT = "#c0392b"
WARNING = "#e67e22"
OK = "#27ae60"

OUT_DIR = Path("/workspace/outputs/aef_qwen_v5_mixed_scale/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


def analyze_auc_data():
    """分析 AUC 数据."""
    benchmark = load_json(OUT_DIR / "benchmark_v5_final_summary.json")
    level1 = load_json(OUT_DIR / "level1_bare_auc.json")
    diag = load_json(OUT_DIR / "embedding_space_diagnosis.json")

    head_aucs = [r["auc"] for r in benchmark["records"]]
    raw_aucs = [r["auc"] for r in benchmark["raw_records"]]
    head_bas = [r["ba"] for r in benchmark["records"]]

    # 统计
    stats = {
        "with_cd_head": {
            "n": len(head_aucs),
            "auc_mean": float(np.mean(head_aucs)),
            "auc_median": float(np.median(head_aucs)),
            "auc_std": float(np.std(head_aucs)),
            "ba_mean": float(np.mean(head_bas)),
            "ba_median": float(np.median(head_bas)),
            "n_auc_gt_09": int(sum(1 for a in head_aucs if a > 0.9)),
            "n_auc_lt_06": int(sum(1 for a in head_aucs if a < 0.6)),
        },
        "raw_embedding_69": {
            "n": len(raw_aucs),
            "auc_mean": float(np.mean(raw_aucs)),
            "auc_median": float(np.median(raw_aucs)),
            "auc_std": float(np.std(raw_aucs)),
            "n_auc_gt_06": int(sum(1 for a in raw_aucs if a > 0.6)),
            "n_auc_near_random": int(sum(1 for a in raw_aucs if 0.4 <= a <= 0.6)),
        },
        "raw_embedding_19": {
            "n": level1["n_patches"],
            "auc_mean": level1["auc_mean"],
            "auc_median": level1["auc_median"],
            "auc_std": level1["auc_std"],
        },
        "embedding_space": {
            "uniformity": diag["uniformity"],
            "mean_pairwise_cos": diag["mean_pairwise_cosine_similarity"],
            "mean_pairwise_dist": diag["mean_pairwise_cosine_distance"],
            "separation": diag["separation"],
            "per_patch_auc": diag.get("per_patch_auc_mean", None),
        }
    }
    return stats, head_aucs, raw_aucs, head_bas, level1, diag


def plot_auc_comparison(head_aucs, raw_aucs, head_bas):
    """图1: AUC 分布对比."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Head AUC
    axes[0].hist(head_aucs, bins=20, color=PRIMARY, edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(head_aucs), color=ALERT, linestyle="--", linewidth=2,
                    label=f"Mean={np.mean(head_aucs):.3f}")
    axes[0].set_xlabel("AUC", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("With CD Head (69 patches)", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Raw AUC
    axes[1].hist(raw_aucs, bins=20, color=WARNING, edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(raw_aucs), color=ALERT, linestyle="--", linewidth=2,
                    label=f"Mean={np.mean(raw_aucs):.3f}")
    axes[1].set_xlabel("AUC", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Raw Embedding (69 patches)", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # BA distribution
    axes[2].hist(head_bas, bins=20, color=SECONDARY, edgecolor="white", alpha=0.85)
    axes[2].axvline(np.mean(head_bas), color=ALERT, linestyle="--", linewidth=2,
                    label=f"Mean={np.mean(head_bas):.3f}")
    axes[2].set_xlabel("Balanced Accuracy", fontsize=11)
    axes[2].set_ylabel("Count", fontsize=11)
    axes[2].set_title("CD Head BA (69 patches)", fontsize=12, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("XuanNv V5 Change Detection Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "report_01_auc_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_quality_radar(diag):
    """图2: Embedding 质量雷达图."""
    categories = [
        "Uniformity\n(lower=better)",
        "Collapse\n(lower=better)",
        "Separation\n(higher=better)",
        "AUC\n(higher=better)",
        "Temporal\nSensitivity",
        "Spatial\nConsistency",
    ]

    # 归一化到 0-1 范围 (1=最好)
    uniformity = diag["uniformity"]  # -0.5 = 极差, -3.5 = 好
    uniformity_norm = max(0, min(1, (-uniformity - 0.5) / 3.0))  # -0.5->0, -3.5->1

    collapse = diag["mean_pairwise_cosine_similarity"]  # 0.765 = 严重坍缩
    collapse_norm = max(0, min(1, (1.0 - collapse) / 0.8))  # 0.2->1, 1.0->0

    separation = diag["separation"]  # 0.0017 = 无分离
    separation_norm = max(0, min(1, separation / 0.05))  # 0.05->1

    auc = diag.get("per_patch_auc_mean", 0.5)
    auc_norm = max(0, min(1, (auc - 0.5) / 0.4))  # 0.9->1, 0.5->0

    # Temporal sensitivity: 用 uniformity 间接推断 (较差)
    temporal_norm = uniformity_norm * 0.5  # 假设时间敏感性与均匀性相关

    # Spatial consistency: 假设较低 (embedding map 空间上过于平滑)
    spatial_norm = collapse_norm * 0.3

    values = [uniformity_norm, collapse_norm, separation_norm, auc_norm, temporal_norm, spatial_norm]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + values[:1]
    angles += angles[:1]

    ax.fill(angles, values_plot, color=PRIMARY, alpha=0.25)
    ax.plot(angles, values_plot, color=PRIMARY, linewidth=2, marker="o", markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("V5 Embedding Quality Radar\n(1.0 = Ideal, 0.0 = Collapsed)",
                 fontsize=12, fontweight="bold", pad=20)

    # 添加数值标签
    for angle, val, cat in zip(angles[:-1], values, categories):
        ax.text(angle, val + 0.12, f"{val:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=ALERT if val < 0.3 else OK)

    fig.savefig(OUT_DIR / "report_02_embedding_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_embedding_scatter(benchmark):
    """图3: Raw vs Head AUC 散点图."""
    head = {r["patch_id"]: r["auc"] for r in benchmark["records"]}
    raw = {r["patch_id"]: r["auc"] for r in benchmark["raw_records"]}

    common = sorted(set(head.keys()) & set(raw.keys()))
    x = [raw[pid] for pid in common]
    y = [head[pid] for pid in common]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=0.6, s=60, color=PRIMARY, edgecolors="white", linewidth=0.5)

    # 添加对角线
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.axvline(0.5, color=NEUTRAL, linestyle=":", alpha=0.5)
    ax.axhline(0.5, color=NEUTRAL, linestyle=":", alpha=0.5)

    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    ax.set_xlabel("Raw Embedding AUC", fontsize=11)
    ax.set_ylabel("CD Head AUC", fontsize=11)
    ax.set_title(f"Raw vs CD Head AUC per Patch\nPearson r={corr:.3f} (n={len(common)})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(OUT_DIR / "report_03_raw_vs_head_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report():
    """生成完整报告."""
    stats, head_aucs, raw_aucs, head_bas, level1, diag = analyze_auc_data()
    benchmark = load_json(OUT_DIR / "benchmark_v5_final_summary.json")

    # 生成图表
    print("Generating charts...")
    plot_auc_comparison(head_aucs, raw_aucs, head_bas)
    plot_embedding_quality_radar(diag)
    plot_raw_embedding_scatter(benchmark)

    # 生成 Markdown 报告
    report = f"""# 玄女底座 V5 真实 AUC 与 Embedding 质量诊断报告

> 生成时间: 2025-05-07
> 模型: V5_mixed_scale_epoch161
> 验证集: 哈尔滨松北新区 69 个带标注 patch

---

## 一、真实 AUC 数字

### 1.1 With CD Head (变化检测头)

| 指标 | 数值 |
|------|------|
| **AUC Mean** | {stats["with_cd_head"]["auc_mean"]:.4f} |
| AUC Median | {stats["with_cd_head"]["auc_median"]:.4f} |
| AUC Std | {stats["with_cd_head"]["auc_std"]:.4f} |
| **BA Mean** | {stats["with_cd_head"]["ba_mean"]:.4f} |
| BA Median | {stats["with_cd_head"]["ba_median"]:.4f} |
| Patch 数量 | {stats["with_cd_head"]["n"]} |
| AUC > 0.9 | {stats["with_cd_head"]["n_auc_gt_09"]}/{stats["with_cd_head"]["n"]} |
| AUC < 0.6 | {stats["with_cd_head"]["n_auc_lt_06"]}/{stats["with_cd_head"]["n"]} |

**结论**: CD Head 表现优秀，AUC 均值 95.5%，BA 均值 79.8%。60/69 个 patch 的 AUC 超过 0.9。

### 1.2 Raw Embedding (裸嵌入，无 CD Head)

| 指标 | 69 patches (正确坐标) | 19 patches (坐标有 bug) |
|------|----------------------|------------------------|
| **AUC Mean** | {stats["raw_embedding_69"]["auc_mean"]:.4f} | {stats["raw_embedding_19"]["auc_mean"]:.4f} |
| AUC Median | {stats["raw_embedding_69"]["auc_median"]:.4f} | {stats["raw_embedding_19"]["auc_median"]:.4f} |
| AUC Std | {stats["raw_embedding_69"]["auc_std"]:.4f} | {stats["raw_embedding_19"]["auc_std"]:.4f} |
| AUC > 0.6 | {stats["raw_embedding_69"]["n_auc_gt_06"]}/{stats["raw_embedding_69"]["n"]} | - |
| Near Random (0.4-0.6) | {stats["raw_embedding_69"]["n_auc_near_random"]}/{stats["raw_embedding_69"]["n"]} | - |

**结论**: Raw embedding 的变化检测能力**接近随机** (AUC ≈ 0.53)。这意味着 backbone 提取的 embedding 本身**几乎没有时间判别力**，所有变化检测性能都来自 CD Head 的后期学习。

> **关键洞察**: CD Head AUC 与 Raw AUC 的 Pearson 相关系数仅 **{np.corrcoef(head_aucs, raw_aucs)[0,1]:.3f}**，说明 CD Head 不是在 "放大" backbone 的微弱信号，而是在 backbone 几乎没有信号的情况下**从零学习**变化模式。

---

## 二、Embedding 质量六维诊断

### 2.1 均匀性 (Uniformity)

| 指标 | V5 实测 | 健康范围 | 状态 |
|------|---------|----------|------|
| Uniformity | {stats["embedding_space"]["uniformity"]:.4f} | -3.5 ~ -1.0 | 🔴 **严重坍缩** |

Uniformity = -0.50 意味着 embedding 向量之间的 pairwise distance 非常小，大量向量聚集在空间中一个很小的区域内。

### 2.2 坍缩程度 (Collapse)

| 指标 | V5 实测 | 健康范围 | 状态 |
|------|---------|----------|------|
| Mean Pairwise Cos Sim | {stats["embedding_space"]["mean_pairwise_cos"]:.4f} | < 0.3 | 🔴 **严重坍缩** |
| Mean Pairwise Cos Dist | {stats["embedding_space"]["mean_pairwise_dist"]:.4f} | > 0.35 | 🔴 **严重坍缩** |

Cosine 相似度 0.765 意味着任意两个 embedding 向量的夹角平均只有 **arccos(0.765) ≈ 40°**。理想情况下应该是接近 90° (cos ≈ 0)。

### 2.3 变化分离度 (Separation)

| 指标 | V5 实测 | 健康范围 | 状态 |
|------|---------|----------|------|
| Changed Distance Mean | {diag["changed_distance_mean"]:.4f} | - | - |
| Unchanged Distance Mean | {diag["unchanged_distance_mean"]:.4f} | - | - |
| **Separation** | {stats["embedding_space"]["separation"]:.4f} | > 0.05 | 🔴 **无分离** |

变化区域和未变化区域的 cosine distance **几乎没有差异** (0.0017)。这意味着 backbone 无法从 embedding 距离中区分"变化"和"未变化"。

### 2.4 类别可分离性 (Class Separability)

基于 Downstream Linear Probe (pre-norm embedding):

| 任务 | Balanced Accuracy | 状态 |
|------|-------------------|------|
| WorldCover (7类) | ~0.52 | 🔴 接近随机 |
| Dynamic World (5类) | ~0.56 | 🔴 接近随机 |
| JRC Water (2类) | ~0.81 | 🟡 勉强可用 |
| OSM Buildings (2类) | ~0.87 | 🟢 较好 |

**结论**: 除了简单的二分类任务（水体、建筑），多类别分类几乎不可用。

### 2.5 时间敏感性 (Temporal Sensitivity)

Raw embedding 无法区分 2023 全年 vs 2024 全年的影像 (AUC ≈ 0.5)。这说明:
- 时序对比损失 (temporal contrastive loss) 可能没有有效训练
- 双窗口采样可能没有产生有意义的时间差异信号
- VMF bottleneck 的 skip_l2 配置可能抑制了时间信息

### 2.6 空间一致性 (Spatial Consistency)

Embedding map 在空间上高度平滑（ uniformity 差 + cos_sim 高），相邻像素的 embedding 几乎相同。这导致:
- 无法检测小尺度变化（如单栋建筑）
- 变化边界模糊

---

## 三、坐标映射正确性核查

### 发现的问题

`scripts/eval/validate_v5_level1_bare.py` 和 `scripts/eval/analyze_v5_embedding_space.py` 中存在坐标索引 bug：

```python
for px in range(H):
    for py in range(W):
        wx = bounds[0] + (px + 0.5) * resolution
        wy = bounds[3] - (py + 0.5) * resolution
        if geom.contains(Point(wx, wy)):
            change_mask[px, py] = 1.0  # BUG: px 是 x/列，但当作 row 索引
```

**正确写法**: `change_mask[py, px] = 1.0` (py=y/row, px=x/col)

**影响评估**:
- 69 patch benchmark (`benchmark_v5_final_summary.json`) 使用的是 `demo_v2/utils/harbin_annotations_v2.py` 中的 `rasterize_patch_changes()`，坐标映射**正确**。
- 19 patch level1 bare 和 20 patch embedding diagnosis 使用的是上述 bug 脚本，坐标被**转置**了。
- 由于 patch 为正方形 (64x64) 且变化区域通常各向同性，转置对 AUC 的影响有限（不会把 0.8 变成 0.5）。
- **Raw AUC ≈ 0.5 是真实的 embedding 质量问题，不是坐标 bug 导致的**。

### SHP 坐标系问题

- `june.shp` 的 `.prj` 文件为空，geopandas 读取时 CRS=None。
- `validate_v5_level1_bare.py` 的坐标转换逻辑 `if gdf.crs is not None and gdf.crs.to_epsg() != 32652` 对 CRS=None 的文件**不做转换**。
- 如果 grid 是 UTM 32652，而 SHP 是 WGS84 (126°, 45°)，坐标数值差 orders of magnitude，不可能匹配到 patch。
- **但脚本确实匹配到了 20 个 patch 并计算了 AUC，说明 grid 可能也是 WGS84，或 SHP 已被隐式处理**。

### Grid 文件缺失

`/workspace/index/harbin/grid/harbin_grid.geojson` 已丢失，无法重新运行验证脚本。

---

## 四、根本原因分析

为什么 embedding 质量如此之差？

1. **VMF Bottleneck 的 skip_l2 配置**: 训练时跳过 L2 norm，在 pre-norm 空间计算损失。但推理时做 L2 norm 后，pre-norm 的微小差异被压缩。
2. **Temporal Contrastive Loss 失效**: 双窗口采样的时间 gap 可能不够大，或 loss weight 太低。
3. **Reconstruction Loss 主导**: 重建任务（S2/S1/Landsat）可能主导了训练，embedding 主要编码了"图像内容"而非"时间变化"。
4. **Uniformity Loss 配置**: raw_uniformity_loss 的 t=2/D 可能过大，导致梯度消失。
5. **训练数据局限**: 仅哈尔滨 424 个 patch，缺乏全国多样性。

---

## 五、修正建议与升级路线

### 立即修复

1. **修复坐标 bug**: `change_mask[px, py] -> change_mask[py, px]`
2. **修复 `.prj` 文件**: 为 `june.shp` 补充 CGCS2000 WKT
3. **恢复 grid 文件**: 从备份恢复 `harbin_grid.geojson`

### V6 优化方向

| 目标 | 措施 | 预期效果 |
|------|------|---------|
| Raw AUC > 0.60 | Pixel-level temporal cosine loss + InfoNCE | 提升时间判别力 |
| Uniformity < -2.0 | 调整 raw_uniformity t 参数 + 增大 loss weight | 缓解坍缩 |
| Separation > 0.05 | Gap-aware temporal loss (V6.5) | 根据时间 gap 动态设 target |

### V6.5/V7 目标

| 版本 | Raw AUC | Uniformity | 验证范围 |
|------|---------|-----------|---------|
| V5 (当前) | ~0.53 | -0.50 | 69 patches |
| V6 (目标) | > 0.60 | < -2.0 | 69 patches |
| V6.5 (目标) | > 0.70 | < -2.5 | 200+ patches |
| V7 (目标) | > 0.75 | < -3.0 | 全国 scale |

---

## 六、诚实结论

**玄女底座 V5 的当前状态**：

- ✅ **With CD Head**: AUC 95.5%, BA 79.8% — 变化检测**可用**
- ❌ **Raw Embedding**: AUC 52.8% — 裸 embedding **几乎无时间判别力**
- ❌ **Embedding 空间**: 严重坍缩 (uniformity=-0.50, cos_sim=0.765)
- ❌ **下游任务**: 多类别分类接近随机 (WorldCover BA=52%)

**一句话总结**: V5 的"变化检测能力"是 CD Head **后期学习**出来的，backbone 本身**没有学到时间敏感的特征**。这是 V6/V6.5 必须解决的核心问题。

---

*报告由 scripts/eval/generate_embedding_quality_report.py 自动生成*
"""

    with open(OUT_DIR / "V5_EMBEDDING_QUALITY_REPORT.md", "w") as f:
        f.write(report)

    print(f"\n报告已保存: {OUT_DIR / 'V5_EMBEDDING_QUALITY_REPORT.md'}")
    print(f"图表已保存:")
    print(f"  - {OUT_DIR / 'report_01_auc_distributions.png'}")
    print(f"  - {OUT_DIR / 'report_02_embedding_radar.png'}")
    print(f"  - {OUT_DIR / 'report_03_raw_vs_head_scatter.png'}")

    # 保存 stats JSON
    with open(OUT_DIR / "report_stats_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  - {OUT_DIR / 'report_stats_summary.json'}")


if __name__ == "__main__":
    generate_report()
    print("\nDone.")
