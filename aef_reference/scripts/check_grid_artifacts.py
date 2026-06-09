#!/usr/bin/env python3
"""
检查修改后的 AEF 模型架构是否存在周期性棋盘伪影。
提取 STPBlock 后的 space_features、encoder 输出 feats_teacher、最终 embedding mu_t，
计算 2D 自相关，检查周期性峰值。
"""
from __future__ import annotations

import os

os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, "/workspace/xuannv/aef_reference")

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.data.haidian_dataset import HaidianAEFDataset, collate_fn


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def pca_rgb(features_hwc: np.ndarray, n_components: int = 3) -> np.ndarray:
    """对 (H, W, C) 特征做 PCA 降到 n_components 通道，归一化到 [0,1] RGB。"""
    H, W, C = features_hwc.shape
    X = features_hwc.reshape(-1, C).astype(np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD: Xc = U S Vt, Vt 的行是主成分方向
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components].T  # (C, n_comp)
    projected = Xc @ comps  # (HW, n_comp)
    rgb = projected.reshape(H, W, n_components)
    for i in range(n_components):
        ch = rgb[..., i]
        vmin, vmax = ch.min(), ch.max()
        if vmax - vmin > 1e-8:
            rgb[..., i] = (ch - vmin) / (vmax - vmin)
        else:
            rgb[..., i] = 0.0
    return rgb.astype(np.float32)


def compute_autocorr_2d(feature_map: np.ndarray) -> np.ndarray:
    """计算 2D 自相关图 (中心化)，输出范围 [0, 1]（中心=1.0）。"""
    f = np.fft.fft2(feature_map)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)
    mx = autocorr.max()
    if mx > 1e-12:
        autocorr = autocorr / mx
    return autocorr


def analyze_periodic_peaks(autocorr: np.ndarray, name: str) -> dict[str, Any]:
    """分析自相关图中的周期性峰值。"""
    H, W = autocorr.shape
    cy, cx = H // 2, W // 2

    # 构建中心屏蔽掩码（3x3 或按比例）
    radius = max(1, min(H, W) // 16)
    yy, xx = np.ogrid[:H, :W]
    center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    off_center = autocorr.copy()
    off_center[center_mask] = 0.0

    max_val = float(off_center.max())
    max_pos = np.unravel_index(np.argmax(off_center), autocorr.shape)

    # 计算径向平均功率谱密度（从自相关反推）
    freqs_y = np.fft.fftshift(np.fft.fftfreq(H))
    freqs_x = np.fft.fftshift(np.fft.fftfreq(W))
    dists = np.sqrt((yy - cy) ** 2 / (H / 2) ** 2 + (xx - cx) ** 2 / (W / 2) ** 2)

    # 简单的“网格分数”：非中心区域最大值
    grid_score = max_val

    return {
        "name": name,
        "shape": (H, W),
        "grid_score": grid_score,
        "max_peak_pos": (int(max_pos[0]), int(max_pos[1])),
        "max_peak_value": max_val,
        "center_radius": radius,
    }


def visualize_feature_and_autocorr(
    feature_map: np.ndarray,
    name: str,
    out_path: Path,
    cmap: str = "viridis",
) -> dict[str, Any]:
    """绘制特征图 + 自相关热图 + 中心线剖面，返回峰值分析结果。"""
    autocorr = compute_autocorr_2d(feature_map)
    analysis = analyze_periodic_peaks(autocorr, name)

    H, W = feature_map.shape
    cy, cx = H // 2, W // 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{name}\nGridScore={analysis['grid_score']:.3f} @ {analysis['max_peak_pos']}", fontsize=11)

    # 原始特征图
    ax = axes[0, 0]
    im = ax.imshow(feature_map, cmap=cmap, aspect="auto")
    ax.set_title("Feature map (mean over channels / PCA)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 自相关图
    ax = axes[0, 1]
    im = ax.imshow(autocorr, cmap="hot", aspect="auto")
    ax.set_title("Auto-correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 水平中心线
    ax = axes[1, 0]
    ax.plot(autocorr[cy, :], label="Horizontal")
    ax.axvline(cx, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Center horizontal profile")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 垂直中心线
    ax = axes[1, 1]
    ax.plot(autocorr[:, cx], label="Vertical", color="orange")
    ax.axhline(cy, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Center vertical profile")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return analysis


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    device = get_device()
    print(f"[INFO] Using device: {device}")

    out_dir = Path("outputs/viz_preview/grid_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 数据集 --------------------
    print("[INFO] Loading dataset...")
    dataset = HaidianAEFDataset(
        data_root="/workspace/xuannv/data_raw/haidian/scenes",
        planet_root="/workspace/xuannv/data_raw/beijing/planetscene",
        stats_dir="/workspace/xuannv/statistics/haidian",
        cache_dir="/workspace/xuannv/aef_reference/src/aef/.cache",
        image_size=128,
        source_names=["s1", "s2", "tianyi_sar", "landsat", "planet"],
        required_sources=["s2"],
        split="train",
        train_ratio=0.9,
        seed=42,
        max_frames=16,
        start_date="20251201",
        end_date="20260430",
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty!")
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")

    # 取第一个样本
    sample = dataset[0]
    batch = collate_fn([sample])
    patch_id = batch["patch_ids"][0]
    print(f"[INFO] Checking patch_id={patch_id}")

    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"].to(device)

    # -------------------- 模型 --------------------
    input_sources = {
        "s1": 2,
        "s2": 6,
        "tianyi_sar": 1,
        "landsat": 6,
        "planet": 4,
    }
    decode_sources = {
        "s1": 2,
        "s2": 6,
        "tianyi_sar": 1,
        "landsat": 6,
        "planet": 4,
        "dem": 1,
        "worldcover": 11,
        "dynamic_world": 9,
        "jrc_water": 1,
    }
    model = AlphaEarthFoundations(
        model_size="small",
        input_sources=input_sources,
        decode_sources=decode_sources,
        per_source_latent=32,
        enable_text_align=False,
    ).to(device)
    model.eval()
    print("[INFO] Model initialized (random weights)")

    # -------------------- 注册 Hooks 捕获 block space_features --------------------
    block_space_features: list[torch.Tensor] = []

    def make_hook(idx: int):
        def hook(module, inp, out):
            space_out = out[0]  # Tuple[space, time, precision]
            block_space_features.append(space_out.detach())
        return hook

    hooks = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(model.encoder.blocks)]

    # -------------------- Forward (只跑 teacher 路径) --------------------
    x_stacked = model._stack_inputs(source_data)
    first_src = next(iter(input_sources.keys()))
    ts = timestamps[first_src]

    with torch.no_grad():
        feats_teacher = model.encoder(x_stacked, ts)
        mu_t = model.summarizer(feats_teacher, ts, valid_periods)

    for h in hooks:
        h.remove()

    print(f"[INFO] feats_teacher shape: {feats_teacher.shape}")
    print(f"[INFO] mu_t shape: {mu_t.shape}")
    print(f"[INFO] Captured {len(block_space_features)} block space features")

    # -------------------- 分析与可视化 --------------------
    analyses: list[dict[str, Any]] = []

    # 1. STPBlock 后的 space_features (1/8L = 16x16)
    num_blocks = len(block_space_features)
    for i, sf in enumerate(block_space_features):
        # sf: (B, T, H_s, W_s, C)
        arr = to_numpy(sf[0, 0])  # (H_s, W_s, C)
        # 取通道均值
        mean_map = arr.mean(axis=-1)
        name = f"block_{i:02d}_space_features"
        analysis = visualize_feature_and_autocorr(
            mean_map, name, out_dir / f"{name}.png", cmap="viridis"
        )
        analyses.append(analysis)
        print(f"  [{name}] shape={arr.shape}, grid_score={analysis['grid_score']:.3f}")

    # 2. Encoder 最终输出 feats_teacher (full res = 128x128)
    ft = to_numpy(feats_teacher[0, 0])  # (H, W, d_p)
    mean_map = ft.mean(axis=-1)
    name = "encoder_output_feats_teacher"
    analysis = visualize_feature_and_autocorr(
        mean_map, name, out_dir / f"{name}.png", cmap="viridis"
    )
    analyses.append(analysis)
    print(f"  [{name}] shape={ft.shape}, grid_score={analysis['grid_score']:.3f}")

    # 3. 最终 embedding mu_t (64D -> PCA RGB)
    mu = to_numpy(mu_t[0])  # (H, W, 64)
    rgb = pca_rgb(mu, n_components=3)  # (H, W, 3)
    # 保存 RGB 图
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title("mu_t PCA RGB")
    ax.axis("off")
    fig.savefig(out_dir / "mu_t_pca_rgb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 对 PCA 后的每个通道分别做自相关
    for c_idx, ch_name in enumerate(["R", "G", "B"]):
        ch = rgb[..., c_idx]
        name = f"mu_t_pca_{ch_name}"
        analysis = visualize_feature_and_autocorr(
            ch, name, out_dir / f"{name}.png", cmap="hot"
        )
        analyses.append(analysis)
        print(f"  [{name}] grid_score={analysis['grid_score']:.3f}")

    # -------------------- 汇总报告 --------------------
    print("\n" + "=" * 60)
    print("GRID ARTIFACT CHECK REPORT")
    print("=" * 60)

    grid_scores = [a["grid_score"] for a in analyses]
    max_score = max(grid_scores)
    max_item = analyses[grid_scores.index(max_score)]

    print(f"Total checked maps: {len(analyses)}")
    print(f"Max grid score: {max_score:.4f} ({max_item['name']})")
    print(f"Mean grid score: {np.mean(grid_scores):.4f}")
    print(f"Median grid score: {np.median(grid_scores):.4f}")

    # 判定逻辑
    # 随机权重的特征图通常不会有强周期性结构，grid score 应该很低 (<0.1)。
    # 如果 >0.3 则认为存在显著伪影；0.1~0.3 为轻微；<0.1 为无明显伪影。
    threshold_strong = 0.30
    threshold_weak = 0.10

    strong = [a for a in analyses if a["grid_score"] >= threshold_strong]
    weak = [a for a in analyses if threshold_weak <= a["grid_score"] < threshold_strong]

    print(f"\nStrong artifacts (score >= {threshold_strong}): {len(strong)}")
    for a in strong:
        print(f"  - {a['name']}: {a['grid_score']:.4f} @ {a['max_peak_pos']}")

    print(f"\nWeak artifacts ({threshold_weak} <= score < {threshold_strong}): {len(weak)}")
    for a in weak:
        print(f"  - {a['name']}: {a['grid_score']:.4f} @ {a['max_peak_pos']}")

    if len(strong) == 0 and len(weak) == 0:
        conclusion = "结论：在随机权重的前向验证中，未发现显著的周期性棋盘伪影。修改后的架构（1/8L Space、双卷积 LearnedSpatialResampling、2D 位置编码、Encoder Fusion Conv）在当前单样本测试下未表现出网格状周期结构。"
    elif len(strong) == 0:
        conclusion = "结论：仅检测到轻微的周期性波动，未出现强烈棋盘伪影。"
    else:
        conclusion = "结论：检测到显著的周期性峰值，棋盘伪影可能仍然存在，建议进一步检查对应层。"

    print(f"\n{conclusion}")

    # 保存文本报告
    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("GRID ARTIFACT CHECK REPORT\n")
        f.write("=" * 60 + "\n\n")
        for a in analyses:
            f.write(f"{a['name']}: shape={a['shape']}, grid_score={a['grid_score']:.4f}, peak={a['max_peak_pos']}\n")
        f.write(f"\n{conclusion}\n")
    print(f"[INFO] Report saved to {report_path}")


if __name__ == "__main__":
    main()
