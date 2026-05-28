"""统计量计算流水线。

封装原 scripts/preprocessing/compute_statistics.py，改为由区域配置驱动。
支持对多区域分别计算，输出到各区域独立的 statistics_dir。

输出格式（与训练代码完全兼容）:
    /workspace/statistics/{region_name}/{source}_stats.json
    {
        "n_channels": 6,
        "channel_0": {"mean": 0.123, "std": 0.045},
        ...
    }
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from preprocessing.utils.logging import get_logger

logger = get_logger(__name__)

# 每个区域最多采样的 patch 数量（统计量对样本数不敏感）
_MAX_PATCHES_DEFAULT = 50


def _compute_source_stats_worker(args: tuple) -> tuple[str, dict | None]:
    """多进程 worker：计算单个数据源的统计量。"""
    source_name, data_root_str, s2_subdir, max_patches = args
    data_root = Path(data_root_str)

    # S2 的子目录可能是 "s2"（原始）或 "s2"（云筛选后），由调用方决定
    if source_name == "s2":
        src_dir = data_root / s2_subdir
    else:
        src_dir = data_root / source_name

    if not src_dir.exists():
        return source_name, None

    patches = sorted([p.name for p in src_dir.iterdir() if p.is_dir()])
    if not patches:
        return source_name, None

    if len(patches) > max_patches:
        np.random.seed(42)
        patches = np.random.choice(patches, max_patches, replace=False).tolist()

    from src.data.transforms import read_tif  # 复用训练代码的 TIFF 读取

    all_samples: list[np.ndarray] = []
    for patch_id in patches:
        patch_dir = src_dir / patch_id
        for tif_path in sorted(patch_dir.glob("*.tif")):
            try:
                data = read_tif(str(tif_path), image_size=-1)
                if data is None:
                    continue
                if all_samples and data.shape[0] != all_samples[0].shape[0]:
                    continue

                # 光学源：log(x+1)/10 变换（与训练时一致）
                if source_name in {"s2", "landsat"}:
                    if data.max() < 2.0:
                        data = data * 10000.0
                    data = np.log(np.clip(data, 0, None) + 1) / 10.0

                # SAR 源：若还是线性 DN 则转 dB
                if source_name in {"s1", "s1_hr", "sar"}:
                    if data.max() > 100:
                        data = 10.0 * np.log10(np.clip(data / 10000.0, 1e-10, None))
                    elif data.max() <= 2.0:
                        data = 10.0 * np.log10(np.clip(data, 1e-10, None))
                    # 已是 dB（含负值）则不转换

                all_samples.append(data)
            except Exception:
                pass

    if not all_samples:
        return source_name, None

    stacked = np.stack(all_samples, axis=0)        # (N, C, H, W) 或 (N, C) 等
    if stacked.ndim == 3:
        stacked = stacked[:, :, np.newaxis, np.newaxis]  # 统一 4D

    n_channels = stacked.shape[1]
    stats: dict = {"n_channels": n_channels}
    for c in range(n_channels):
        ch = stacked[:, c].ravel()
        ch = ch[np.isfinite(ch)]
        stats[f"channel_{c}"] = {
            "mean": float(np.mean(ch)),
            "std": float(np.std(ch)),
        }
    return source_name, stats


def run_statistics(region_cfg: dict, workers: int = 4) -> dict[str, dict]:
    """
    计算区域所有数据源的通道统计量并写入 statistics_dir。

    统计基于 cloud_filtered_dir（S2 优先）或 output_dir（其他源）。

    Returns:
        {source_name: stats_dict}
    """
    region_name = region_cfg["region_name"]
    output_dir = Path(region_cfg["output_dir"])
    cloud_filtered_dir = Path(region_cfg.get("cloud_filtered_dir", region_cfg["output_dir"]))
    stats_dir = Path(region_cfg["statistics_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    # 确定各源数据根目录
    # S2 用云筛选后的目录，s2 子目录名为 "s2"
    # 其他源用 output_dir
    data_roots: dict[str, tuple[str, str]] = {}  # source → (data_root, s2_subdir)

    for src_name, src_cfg in region_cfg["sources"].items():
        if not src_cfg.get("enabled", False):
            continue
        if src_name == "s2":
            root = cloud_filtered_dir
            s2_subdir = "s2"
        elif src_name == "sar":
            # SAR 数据在 cloud_filtered_dir 的 sar/ 子目录（LocalSARImporter 写入 output_dir/sar）
            root = output_dir
            s2_subdir = "s2"
        else:
            root = output_dir
            s2_subdir = "s2"
        data_roots[src_name] = (str(root), s2_subdir)

    sources_to_compute = list(data_roots.keys())
    logger.info(
        f"[Statistics/{region_name}] 计算 {len(sources_to_compute)} 个源的统计量: "
        f"{sources_to_compute}"
    )

    worker_args = [
        (src_name, data_root, s2_subdir, _MAX_PATCHES_DEFAULT)
        for src_name, (data_root, s2_subdir) in data_roots.items()
    ]

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(worker_args))) as executor:
        for src_name, stats in executor.map(_compute_source_stats_worker, worker_args):
            if stats is None:
                logger.warning(f"  [{src_name}] 无有效样本，跳过")
                continue
            out_fp = stats_dir / f"{src_name}_stats.json"
            out_fp.write_text(json.dumps(stats, indent=2))
            ch0 = stats.get("channel_0", {})
            logger.info(
                f"  [{src_name}] ch0 mean={ch0.get('mean', '?'):.4f} "
                f"std={ch0.get('std', '?'):.4f} → {out_fp}"
            )
            results[src_name] = stats

    logger.info(f"[Statistics/{region_name}] 完成，已保存 {len(results)} 个统计文件到 {stats_dir}")
    return results
