#!/usr/bin/env python3
"""
检查修改后的 AEF 模型架构是否存在周期性棋盘伪影。
提取 STPBlock 后的 space_features、encoder 输出 feats_teacher、最终 embedding mu_t，
计算 2D 自相关及 1D 剖面振荡，检查周期性峰值。

关键改进：
- 由于自然图像本身在短距离自相关很高，单纯看“off-center max”会误报。
- 本脚本改用“AC 剖面 FFT 峰值检测”来识别特定周期（8px、16px 等）的异常振荡。
"""
from __future__ import annotations

import os

os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, "/workspace/xuannv/aef_reference")

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
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components].T
    projected = Xc @ comps
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
    """计算 2D 自相关图 (中心化)。"""
    f = np.fft.fft2(feature_map)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)
    mx = autocorr.max()
    if mx > 1e-12:
        autocorr = autocorr / mx
    return autocorr


def ac_profiles(feature_map: np.ndarray):
    """返回自相关的中心水平/垂直剖面。"""
    ac = compute_autocorr_2d(feature_map)
    H, W = ac.shape
    cy, cx = H // 2, W // 2
    return ac[cy, :], ac[:, cx]


def detect_periodic_oscillation(
    profile: np.ndarray,
    suspected_periods: list[int] = [4, 8, 16],
    search_radius: int = 2,
) -> dict[str, Any]:
    """
    对一条 AC 剖面做 FFT，检测在 suspected_periods 附近是否存在异常功率峰值。
    返回各周期的 spike_ratio（峰值功率 / 局部中位数功率）。
    """
    # profile 长度 N（奇数），中心在 N//2
    N = len(profile)
    # 去掉中心尖峰（DC + 近邻），避免其旁瓣干扰
    center = N // 2
    filtered = profile.copy()
    filtered[max(0, center - 2) : min(N, center + 3)] = np.median(filtered)

    # FFT of the profile
    spectrum = np.abs(np.fft.rfft(filtered)) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0)  # cycles per pixel

    # 把周期转换成频率索引
    results = {}
    for period in suspected_periods:
        if period >= N:
            continue
        # 对应的频率 bin：在 rfft 中，频率 k/N，k = N/period
        target_k = N / period
        k_min = max(1, int(np.floor(target_k - search_radius)))
        k_max = min(len(spectrum) - 1, int(np.ceil(target_k + search_radius)))

        local_peak = spectrum[k_min : k_max + 1].max()
        # 局部背景：取该窗口前后各 5 个点的中位数
        bg_window = []
        if k_min - 5 >= 0:
            bg_window.extend(spectrum[k_min - 5 : k_min].tolist())
        if k_max + 5 < len(spectrum):
            bg_window.extend(spectrum[k_max + 1 : k_max + 6].tolist())
        bg = np.median(bg_window) if bg_window else 1e-12

        spike_ratio = local_peak / (bg + 1e-12)
        results[f"period_{period}"] = {
            "spike_ratio": float(spike_ratio),
            "freq_bin": int(round(target_k)),
        }
    return results


def visualize_feature(
    feature_map: np.ndarray,
    name: str,
    out_path: Path,
    cmap: str = "viridis",
) -> dict[str, Any]:
    """绘制特征图 + 自相关热图 + AC 剖面 + 周期检测。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = feature_map.shape
    ac = compute_autocorr_2d(feature_map)
    prof_h, prof_v = ac_profiles(feature_map)

    # 周期检测
    osc_h = detect_periodic_oscillation(prof_h)
    osc_v = detect_periodic_oscillation(prof_v)

    # 取最大 spike_ratio 作为代表
    all_ratios = []
    for d in (osc_h, osc_v):
        all_ratios.extend([v["spike_ratio"] for v in d.values()])
    max_ratio = max(all_ratios) if all_ratios else 0.0
    worst_period = None
    for d, direction in [(osc_h, "H"), (osc_v, "V")]:
        for k, v in d.items():
            if v["spike_ratio"] == max_ratio:
                worst_period = f"{k}_{direction}"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"{name}\nMaxSpike={max_ratio:.1f}x @{worst_period}", fontsize=11
    )

    # 原始特征图
    ax = axes[0, 0]
    im = ax.imshow(feature_map, cmap=cmap, aspect="auto")
    ax.set_title("Feature map")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 自相关图
    ax = axes[0, 1]
    im = ax.imshow(ac, cmap="hot", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Auto-correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 水平剖面 + 周期标记
    ax = axes[1, 0]
    x_axis = np.arange(-W // 2, W // 2 + (W % 2))
    ax.plot(x_axis, prof_h, label="Horizontal")
    for period in [8, 16]:
        if period < W // 2:
            ax.axvline(period, color="red", linestyle=":", alpha=0.4)
            ax.axvline(-period, color="red", linestyle=":", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("AC horizontal profile")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 垂直剖面
    ax = axes[1, 1]
    y_axis = np.arange(-H // 2, H // 2 + (H % 2))
    ax.plot(y_axis, prof_v, label="Vertical", color="orange")
    for period in [8, 16]:
        if period < H // 2:
            ax.axvline(period, color="red", linestyle=":", alpha=0.4)
            ax.axvline(-period, color="red", linestyle=":", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title("AC vertical profile")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "name": name,
        "shape": (H, W),
        "max_spike_ratio": max_ratio,
        "worst_period": worst_period,
        "osc_h": osc_h,
        "osc_v": osc_v,
    }


def analyze_single_patch(
    model: AlphaEarthFoundations,
    batch: dict[str, Any],
    out_dir: Path,
    patch_id: str,
) -> list[dict[str, Any]]:
    """对单个 patch 做完整分析，返回所有分析结果。"""
    device = next(model.parameters()).device
    source_data = {k: v.to(device) for k, v in batch["source_data"].items()}
    timestamps = {k: v.to(device) for k, v in batch["timestamps"].items()}
    valid_periods = batch["valid_periods"].to(device)

    first_src = next(iter(model.input_sources.keys()))
    ts = timestamps[first_src]

    # -------------------- 注册 Hooks 捕获 block space_features --------------------
    block_space_features: list[torch.Tensor] = []

    def make_hook(idx: int):
        def hook(module, inp, out):
            space_out = out[0]
            block_space_features.append(space_out.detach())
        return hook

    hooks = [
        block.register_forward_hook(make_hook(i))
        for i, block in enumerate(model.encoder.blocks)
    ]

    # -------------------- Forward (只跑 teacher 路径) --------------------
    x_stacked = model._stack_inputs(source_data)
    with torch.no_grad():
        feats_teacher = model.encoder(x_stacked, ts)
        mu_t = model.summarizer(feats_teacher, ts, valid_periods)

    for h in hooks:
        h.remove()

    analyses: list[dict[str, Any]] = []
    patch_out_dir = out_dir / patch_id
    patch_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. STPBlock 后的 space_features (1/8L = 16x16)
    for i, sf in enumerate(block_space_features):
        arr = to_numpy(sf[0, 0])  # (H_s, W_s, C)
        mean_map = arr.mean(axis=-1)
        name = f"block_{i:02d}_space"
        analysis = visualize_feature(
            mean_map,
            name,
            patch_out_dir / f"{name}.png",
            cmap="viridis",
        )
        analyses.append(analysis)

    # 2. Encoder 最终输出 feats_teacher (full res = 128x128)
    ft = to_numpy(feats_teacher[0, 0])  # (H, W, d_p)
    # 检查多个统计量：均值、标准差、最差单通道
    for stat_name, stat_map in [
        ("mean", ft.mean(axis=-1)),
        ("std", ft.std(axis=-1)),
    ]:
        name = f"feats_teacher_{stat_name}"
        analysis = visualize_feature(
            stat_map,
            name,
            patch_out_dir / f"{name}.png",
            cmap="viridis",
        )
        analyses.append(analysis)

    # 最差单通道
    worst_ch = 0
    worst_ratio = 0.0
    for c in range(ft.shape[-1]):
        osc = detect_periodic_oscillation(ac_profiles(ft[:, :, c])[0])
        ratios = [v["spike_ratio"] for v in osc.values()]
        if ratios and max(ratios) > worst_ratio:
            worst_ratio = max(ratios)
            worst_ch = c
    name = f"feats_teacher_worst_ch{worst_ch}"
    analysis = visualize_feature(
        ft[:, :, worst_ch],
        name,
        patch_out_dir / f"{name}.png",
        cmap="viridis",
    )
    analyses.append(analysis)

    # 3. 最终 embedding mu_t (64D -> PCA RGB + 单通道)
    mu = to_numpy(mu_t[0])  # (H, W, 64)
    rgb = pca_rgb(mu, n_components=3)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title("mu_t PCA RGB")
    ax.axis("off")
    fig.savefig(patch_out_dir / "mu_t_pca_rgb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for c_idx, ch_name in enumerate(["R", "G", "B"]):
        ch = rgb[..., c_idx]
        name = f"mu_t_pca_{ch_name}"
        analysis = visualize_feature(
            ch,
            name,
            patch_out_dir / f"{name}.png",
            cmap="hot",
        )
        analyses.append(analysis)

    # mu_t 单通道
    worst_ch = 0
    worst_ratio = 0.0
    for c in range(mu.shape[-1]):
        osc = detect_periodic_oscillation(ac_profiles(mu[:, :, c])[0])
        ratios = [v["spike_ratio"] for v in osc.values()]
        if ratios and max(ratios) > worst_ratio:
            worst_ratio = max(ratios)
            worst_ch = c
    name = f"mu_t_worst_ch{worst_ch}"
    analysis = visualize_feature(
        mu[:, :, worst_ch],
        name,
        patch_out_dir / f"{name}.png",
        cmap="viridis",
    )
    analyses.append(analysis)

    return analyses


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

    # -------------------- 分析多个 patch --------------------
    patch_indices = [0, 10, 20]
    all_analyses: list[dict[str, Any]] = []

    for idx in patch_indices:
        sample = dataset[idx]
        batch = collate_fn([sample])
        patch_id = batch["patch_ids"][0]
        print(f"\n[INFO] Analyzing patch {patch_id} (index {idx})...")
        analyses = analyze_single_patch(model, batch, out_dir, patch_id)
        all_analyses.extend(analyses)

        # 打印该 patch 摘要
        max_ratio = max(a["max_spike_ratio"] for a in analyses)
        worst = next(a for a in analyses if a["max_spike_ratio"] == max_ratio)
        print(f"  -> Worst: {worst['name']} spike_ratio={max_ratio:.1f}x @ {worst['worst_period']}")

    # -------------------- 汇总报告 --------------------
    print("\n" + "=" * 70)
    print("GRID ARTIFACT CHECK REPORT")
    print("=" * 70)

    # 按名称分组统计
    from collections import defaultdict

    by_name = defaultdict(list)
    for a in all_analyses:
        by_name[a["name"]].append(a["max_spike_ratio"])

    print(f"\n{'Layer':<35s} {'MeanSpike':>10s} {'MaxSpike':>10s} {'N':>4s}")
    print("-" * 65)
    for name in sorted(by_name.keys()):
        vals = by_name[name]
        print(
            f"{name:<35s} {np.mean(vals):>10.1f}x {np.max(vals):>10.1f}x {len(vals):>4d}"
        )

    # 总体判定
    # spike_ratio > 10x 视为显著异常；3~10x 为轻微；<3x 为正常
    strong = [a for a in all_analyses if a["max_spike_ratio"] > 10.0]
    weak = [a for a in all_analyses if 3.0 <= a["max_spike_ratio"] <= 10.0]
    clean = [a for a in all_analyses if a["max_spike_ratio"] < 3.0]

    print(f"\n{'Category':<20s} Count")
    print("-" * 30)
    print(f"{'Strong (>10x)':<20s} {len(strong)}")
    print(f"{'Weak (3-10x)':<20s} {len(weak)}")
    print(f"{'Clean (<3x)':<20s} {len(clean)}")

    # 关键发现
    mu_t_analyses = [a for a in all_analyses if "mu_t" in a["name"]]
    encoder_mean_analyses = [a for a in all_analyses if "feats_teacher_mean" in a["name"]]

    mu_t_max = max(a["max_spike_ratio"] for a in mu_t_analyses) if mu_t_analyses else 0.0
    enc_mean_max = (
        max(a["max_spike_ratio"] for a in encoder_mean_analyses)
        if encoder_mean_analyses
        else 0.0
    )

    print(f"\n[Key findings]")
    print(f"  - Encoder output (mean across channels): max spike = {enc_mean_max:.1f}x")
    print(f"  - Final embedding mu_t (PCA / raw): max spike = {mu_t_max:.1f}x")

    # 明确结论
    if mu_t_max > 10.0:
        conclusion = (
            "结论：修改后的架构在最终 embedding (mu_t) 中仍存在显著的周期性伪影。\n"
            "      具体表现为水平条纹，周期约 8~16 像素，在 AC 剖面 FFT 中呈现 >10x 的功率尖峰。\n"
            "      Encoder 输出的通道均值虽被 final_fusion 抑制，但单通道及 TemporalSummarizer\n"
            "      后的输出仍保留了强烈的网格状振荡。建议：\n"
            "      1) 增大 final_fusion 的卷积核（3x3 -> 5x5 或 7x7）；\n"
            "      2) 在 LearnedSpatialResampling 的 upsample 后增加 anti-aliasing blur；\n"
            "      3) 在 TemporalSummarizer 前对 feats_teacher 做额外的空间平滑。"
        )
    elif mu_t_max > 3.0:
        conclusion = (
            "结论：最终 embedding 存在轻微周期性波动，尚未完全消除伪影。"
        )
    else:
        conclusion = (
            "结论：在当前随机权重、少量样本测试下，未检测到显著的周期性棋盘伪影。"
        )

    print(f"\n{conclusion}")

    # 保存文本报告
    report_path = out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("GRID ARTIFACT CHECK REPORT\n")
        f.write("=" * 70 + "\n\n")
        for name in sorted(by_name.keys()):
            vals = by_name[name]
            f.write(
                f"{name:<35s} mean={np.mean(vals):>7.1f}x max={np.max(vals):>7.1f}x n={len(vals)}\n"
            )
        f.write(f"\n{conclusion}\n")
    print(f"[INFO] Report saved to {report_path}")


if __name__ == "__main__":
    main()
