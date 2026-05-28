"""S2 云筛选处理器。

封装原 scripts/preprocessing/filter_cloudy_frames.py 的逻辑，
改为从区域配置读取参数，支持任意区域复用。
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from preprocessing.utils.logging import get_logger

logger = get_logger(__name__)


def compute_cloud_score(data: np.ndarray) -> float:
    """
    计算云量评分，值越低越 clear。

    使用亮度 + NDVI 的组合指标:
        cloud_score = brightness/10000 - ndvi

    支持 0-1 范围（GEE 导出 scale=1）和 0-10000 范围两种输入。
    """
    if data.shape[0] < 4:
        return 0.0  # 波段不足时不过滤

    # 解包波段（假设顺序 B2 B3 B4 [B8A/NIR] ...）
    blue = data[0].astype(np.float32)
    green = data[1].astype(np.float32)
    red = data[2].astype(np.float32)
    nir = data[3].astype(np.float32)

    # 自适应范围检测：GEE 导出 scale=1 时值域 0-1
    if data.max() < 2.0:
        blue, green, red, nir = blue * 10000, green * 10000, red * 10000, nir * 10000

    brightness = (blue + green + red).mean() / 3.0
    ndvi = float(np.nanmean((nir - red) / (nir + red + 1e-6)))
    return brightness / 10000.0 - ndvi


def _process_patch_worker(args: tuple) -> tuple[str, int, int, str]:
    """多进程 worker：处理单个 patch 的云筛选。"""
    patch_id, src_dir, out_dir, max_per_month, cloud_threshold = args
    patch_src = Path(src_dir) / patch_id
    patch_out = Path(out_dir) / patch_id
    patch_out.mkdir(parents=True, exist_ok=True)

    if not patch_src.exists():
        return patch_id, 0, 0, "missing"

    tif_files = sorted(patch_src.glob("*.tif"))
    if not tif_files:
        return patch_id, 0, 0, "empty"

    frames: list[tuple[Path, float]] = []
    for f in tif_files:
        try:
            with rasterio.open(f) as src:
                data = src.read()
                score = compute_cloud_score(data)
                frames.append((f, score))
        except Exception:
            continue

    if not frames:
        return patch_id, 0, 0, "read_error"

    # 按月分组（文件名前 6 位 YYYYMM）
    monthly: dict[str, list] = defaultdict(list)
    for f, score in frames:
        month = f.stem[:6]
        monthly[month].append((f, score))

    selected: list[tuple[Path, float]] = []
    all_cloudy_count = 0
    for month, month_frames in sorted(monthly.items()):
        month_frames.sort(key=lambda x: x[1])
        if all(score > cloud_threshold for _, score in month_frames):
            all_cloudy_count += 1
            selected.append(month_frames[0])  # fallback: 最 clear 的一帧
        else:
            selected.extend(month_frames[:max_per_month])

    n_copied = 0
    for f, _ in selected:
        dst = patch_out / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            n_copied += 1

    return patch_id, n_copied, all_cloudy_count, "ok"


class S2CloudFilter:
    """
    S2 云筛选处理器。

    从 source_cfg["cloud_filter"] 读取参数：
        max_per_month   : 每月最多保留帧数（default 2）
        cloud_threshold : cloud_score 阈值（default 0.3）
    """

    def __init__(self, region_cfg: dict) -> None:
        self.region_cfg = region_cfg
        src_cfg = region_cfg["sources"]["s2"]
        cf = src_cfg.get("cloud_filter", {})
        self.max_per_month: int = cf.get("max_per_month", 2)
        self.cloud_threshold: float = cf.get("cloud_threshold", 0.3)
        self.output_dir = Path(region_cfg["output_dir"])
        self.cloud_filtered_dir = Path(region_cfg["cloud_filtered_dir"])

    def run(self, workers: int = 16) -> dict:
        """
        对 output_dir/s2/ 执行云筛选，输出到 cloud_filtered_dir/s2/。

        Returns:
            统计字典 {"n_patches", "before_total", "after_total", "fallbacks"}
        """
        src_dir = self.output_dir / "s2"
        out_dir = self.cloud_filtered_dir / "s2"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            logger.warning(f"[S2CloudFilter] S2 源目录不存在: {src_dir}")
            return {}

        patches = sorted([d.name for d in src_dir.iterdir()
                          if d.is_dir() and d.name.startswith("patch_")])
        logger.info(
            f"[S2CloudFilter] {len(patches)} patches, "
            f"max_per_month={self.max_per_month}, threshold={self.cloud_threshold}"
        )

        worker_args = [
            (p, str(src_dir), str(out_dir), self.max_per_month, self.cloud_threshold)
            for p in patches
        ]

        total_copied = 0
        total_fallbacks = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for pid, n_copied, n_fallbacks, status in executor.map(
                _process_patch_worker, worker_args
            ):
                total_copied += n_copied
                total_fallbacks += n_fallbacks
                if status not in ("ok", "skipped"):
                    logger.debug(f"  {pid}: {status}")

        before_total = sum(
            len(list((src_dir / p).glob("*.tif"))) for p in patches
        )
        after_total = sum(
            len(list((out_dir / p).glob("*.tif"))) for p in patches
        )

        stats = {
            "n_patches": len(patches),
            "before_total": before_total,
            "after_total": after_total,
            "fallbacks": total_fallbacks,
            "avg_per_patch": after_total / max(len(patches), 1),
        }

        # 写统计文件
        stats_file = out_dir / "cloud_filter_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        logger.info(
            f"[S2CloudFilter] 完成: {before_total} → {after_total} 帧, "
            f"fallback={total_fallbacks}, stats={stats_file}"
        )
        return stats
