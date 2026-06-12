"""数据集 — 严格按论文设计: 3类输入 (S2+S1+Landsat), 7类重建目标.

核心设计 (对齐 AlphaEarth 论文):
- 输入 (Input): 仅 S2, S1, Landsat — 带时间戳的图像帧
- 目标 (Target): 输入3类 + DEM + WorldCover + Dynamic World + JRC Water = 7类
- 静态源 (DEM/WorldCover/JRC) 不作为输入，仅作为重建目标
- 训练时随机裁剪 valid_period (temporal window augmentation)
- 兼容单景 (YYYYMMDD) 和季度 (YYYYQN) 文件名
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.data.transforms import (
    INPUT_SOURCES,
    TARGET_SOURCES,
    SOURCE_TYPE_MAP,
    CATEGORICAL_SOURCES,
    label_to_timestamp_ms,
    read_tif,
    read_tif_aligned,
    normalize_data,
    WC_CLASS_MAP,
)


# ---------------------------------------------------------------------------
# 多进程预加载 worker
# ---------------------------------------------------------------------------

def _preload_patch_worker(args: tuple) -> tuple[str, dict, int]:
    """多进程 worker：加载单个 patch 的所有数据.
    
    将实例方法的核心逻辑复制到模块级别，避免 pickle 实例方法的问题.
    """
    (
        patch_id,
        data_root_str,
        input_sources,
        target_sources,
        image_size,
        input_dim,
        reconstruction_channels,
        num_classes,
        stats,
        merge_hr_into_lr,
        filter_2025_monthly,
        max_frames,
    ) = args

    data_root = Path(data_root_str)
    cache_entry: dict = {}
    n_cached = 0

    def _resolve(source_name: str, pid: str) -> Path | None:
        # 新结构: patch_id / source_name
        source_dir = data_root / pid / source_name
        if source_dir.exists():
            return source_dir
        # 旧结构: source_name / patch_id
        source_dir = data_root / source_name / pid
        if source_dir.exists():
            return source_dir
        for sub_dir in data_root.iterdir():
            if not sub_dir.is_dir():
                continue
            candidate = sub_dir / source_name / pid
            if candidate.exists():
                return candidate
        return None

    def _pad_channels(data: np.ndarray, target_dim: int) -> np.ndarray:
        current = data.shape[0]
        if current >= target_dim:
            return data[:target_dim]
        padded = np.zeros((target_dim, data.shape[1], data.shape[2]), dtype=data.dtype)
        padded[:current] = data
        return padded

    def _load_input_frames(pid: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        source_dir = _resolve(source_name, pid)
        tif_files = sorted(source_dir.glob("*.tif")) if source_dir is not None else []

        hr_name = None
        if merge_hr_into_lr and source_name == "s2":
            hr_name = "s2_hr"
        elif merge_hr_into_lr and source_name == "s1":
            hr_name = "s1_hr"

        hr_files: dict[str, Path] = {}
        if hr_name:
            hr_dir = _resolve(hr_name, pid)
            if hr_dir is not None:
                for p in sorted(hr_dir.glob("*.tif")):
                    hr_files[p.stem] = p

        if not tif_files and not hr_files:
            return (np.zeros((0, input_dim, image_size, image_size), dtype=np.float32),
                    np.zeros(0, dtype=np.float64))

        if filter_2025_monthly:
            def _is_valid(path: Path) -> bool:
                stem = path.stem
                if "Q" in stem.upper():
                    return False
                if len(stem) == 8 and stem.isdigit() and stem.startswith("2025"):
                    return 4 <= int(stem[4:6]) <= 10
                return False
            tif_files = [f for f in tif_files if _is_valid(f)]

        frames_list: list[np.ndarray] = []
        timestamps: list[float] = []

        # 限制预加载帧数，避免缓存过大
        if len(tif_files) > max_frames:
            tif_files = tif_files[::max(1, len(tif_files) // max_frames)][:max_frames]

        for tif_path in tif_files:
            stem = tif_path.stem
            data = read_tif(tif_path, image_size)
            if data is None:
                continue
            data = normalize_data(data, source_name, stats, num_classes)

            if hr_name and stem in hr_files:
                hr_data = read_tif(hr_files[stem], image_size)
                if hr_data is not None:
                    hr_data = normalize_data(hr_data, hr_name, stats, num_classes)
                    data = np.concatenate([data, hr_data], axis=0)

            data = _pad_channels(data, input_dim)
            frames_list.append(data)
            timestamps.append(float(label_to_timestamp_ms(stem)))

        if not frames_list:
            return (np.zeros((0, input_dim, image_size, image_size), dtype=np.float32),
                    np.zeros(0, dtype=np.float64))

        return np.stack(frames_list), np.array(timestamps, dtype=np.float64)

    # 加载输入源
    for src_name in input_sources:
        result = _load_input_frames(patch_id, src_name)
        if len(result[0]) > 0:
            cache_entry[src_name] = result
            n_cached += 1

    # 加载目标源
    for tgt_name, loss_type, sensor_src in target_sources:
        if tgt_name in cache_entry:
            continue
        if tgt_name in ("dem", "worldcover", "jrc_water", "dynamic_world"):
            src_dir = _resolve(tgt_name, patch_id)
            if src_dir is not None:
                tif_files = sorted(src_dir.glob("*.tif"))
                if tif_files:
                    data = read_tif(tif_files[0], 0)
                    if data is not None:
                        if tgt_name == "jrc_water":
                            data = data.astype(np.float32)
                            data[data == -128] = np.nan
                        data = normalize_data(data, tgt_name, stats, num_classes)
                        data = _pad_channels(data, reconstruction_channels)
                        cache_entry[tgt_name] = (data[np.newaxis, ...], np.array([0.0]))
                        n_cached += 1

    return patch_id, cache_entry, n_cached


class HarbinPatchDataset(Dataset):
    """哈尔滨 Patch 数据集 — 3类输入, 7类目标."""

    def __init__(self, cfg) -> None:
        d = cfg.data
        self.data_root = Path(d.manifest_path)
        self.image_size = d.image_size
        self.max_frames = d.max_frames
        self.input_dim = d.input_dim
        self.metadata_dim = d.metadata_dim
        self.num_input_sources = d.num_input_sources      # 3
        self.num_target_sources = d.num_target_sources     # 7
        self.input_sources = getattr(d, "input_sources", None) or INPUT_SOURCES[:self.num_input_sources]
        self.target_sources_cfg = getattr(d, "target_sources", None)
        if self.target_sources_cfg is not None:
            self.target_sources = [(t["name"], t["loss_type"], t["sensor_src"]) for t in self.target_sources_cfg]
        else:
            self.target_sources = TARGET_SOURCES[:self.num_target_sources]
        self.num_classes = d.num_classes
        self.reconstruction_channels = getattr(cfg.model, "reconstruction_channels", self.input_dim)
        self.training = True
        self.cloud_filter_threshold = d.cloud_filter_threshold
        self.variance_weighted = d.variance_weighted
        self.filter_2025_monthly = getattr(d, "filter_2025_monthly", False)
        self.source_channels = getattr(d, "source_channels", {})
        self.merge_hr_into_lr = getattr(d, "merge_hr_into_lr", False)
        self.cfg = cfg

        # ★ 多分辨率输入配置
        self.use_multires = getattr(d, "use_multires", False)
        self.patch_size_m = getattr(d, "patch_size_m", 1280.0)
        self.source_gsd = getattr(d, "source_gsd", {})
        self.source_image_sizes = getattr(d, "source_image_sizes", None) or {}
        self.common_spatial_size = getattr(cfg.model, "common_spatial_size", (64, 64))
        # 缓存每个 patch 的统一地理 bounds 与 CRS（多分辨率模式下使用）
        self._patch_bounds: dict[str, tuple] = {}

        self.temporal_window_augmentation = d.temporal_window_augmentation
        self.temporal_window_prob = d.temporal_window_prob
        self.temporal_window_min_frames = d.temporal_window_min_frames
        self.temporal_window_max_frames = d.temporal_window_max_frames

        # 双窗口采样模式: "random_split" | "adjacent_month" | "non_overlap" | "mixed_scale"
        self.window_mode = getattr(d, "window_mode", "random_split")
        # non_overlap 参数
        self._non_overlapping_windows = getattr(d, "non_overlap", True) if self.window_mode == "non_overlap" else False
        self._min_window_frames = getattr(d, "non_overlap_min_frames", 4)
        self._max_window_frames = getattr(d, "non_overlap_max_frames", 12)
        self._min_window_gap_ms = getattr(d, "non_overlap_min_gap_ms", 6 * 30 * 24 * 3600 * 1000)
        # mixed_scale 参数
        self._mixed_scale_long_prob = getattr(d, "mixed_scale_long_prob", 0.5)
        self._mixed_scale_short_prob = getattr(d, "mixed_scale_short_prob", 0.5)
        self._mixed_scale_short_max_gap_ms = getattr(d, "mixed_scale_short_max_gap_ms", 3 * 30 * 24 * 3600 * 1000)
        self._mixed_scale_long_min_gap_ms = getattr(d, "mixed_scale_long_min_gap_ms", 6 * 30 * 24 * 3600 * 1000)
        # 跨时相掩码重建配置
        self.ct_mask_ratio = getattr(d, "ct_mask_ratio", 0.3)   # 掩码比例
        self.ct_mask_patch_size = getattr(d, "ct_mask_patch_size", 8)  # 掩码 patch 尺寸
        # MAE-style recon_mask：visible_ratio = 1 - mask_ratio，默认 75% 掩码 → 25% 可见
        self.recon_mask_visible_ratio = 1.0 - getattr(d, "recon_mask_ratio", 0.75)
        self.recon_mask_patch_size = getattr(d, "recon_mask_patch_size", 16)  # block 掩码大小
        
        # ★ Round 2: 跨时相重建配置
        self.cross_temporal = getattr(d, "cross_temporal", False)
        self.cross_temporal_prob = getattr(d, "cross_temporal_prob", 0.0)
        self.cross_temporal_min_gap = getattr(d, "cross_temporal_min_gap_months", 2)

        self.patches = self._discover_patches()
        # V13-fix: 支持随机采样部分 patch 用于快速验证
        max_patches = getattr(d, 'max_patches', None)
        if max_patches and isinstance(max_patches, int) and max_patches > 0:
            if len(self.patches) > max_patches:
                rng = random.Random(getattr(cfg.experiment, 'seed', 42) + 12345)
                self.patches = rng.sample(self.patches, max_patches)
                print(f"[Dataset] Randomly sampled {max_patches} patches for fast validation")
        # 实例判别: 构建 patch_id → 整数索引的映射
        self._patch_to_idx: dict[str, int] = {p: i for i, p in enumerate(self.patches)}
        stats_dir = Path(d.stats_dir) if d.stats_dir else None
        self.stats = self._load_stats(stats_dir)

        self._sample_weights: np.ndarray | None = None
        if self.variance_weighted:
            self._compute_sample_weights()

        # ★ 内存预加载: 避免每个 epoch 重复从磁盘读取 GeoTIFF
        self._cache: dict[str, dict[str, tuple]] = {}
        if getattr(d, "preload", True):
            self._preload_all()

        # ★ OlmoEarth teacher tokens 预加载到内存（避免每个 __getitem__ 重复读 ~1.3GB npz）
        #   在主进程加载，DataLoader fork worker 时通过 COW 共享，不会按 worker 数翻倍。
        self._teacher_tokens: dict[int, np.ndarray] = {}   # month -> (N, 32, 32, 768) fp16
        self._teacher_global: dict[int, np.ndarray] = {}   # month -> (N, 768) fp32
        self._teacher_tok_pid2row: dict[int, dict | None] = {}  # month -> {patch_id: row}
        self._teacher_glb_pid2row: dict[int, dict | None] = {}  # month -> {patch_id: row}
        self._teacher_months: list[int] = []
        self._preload_teacher_tokens()
        self._preload_aef_embeddings()
        
        # ★ V13: 构建月度样本索引
        self.monthly_samples = self._build_monthly_samples()
        print(f"[Dataset] 月度样本数: {len(self.monthly_samples)} (来自 {len(self.patches)} patches)")

    def _build_monthly_samples(self) -> list[tuple[str, int, int]]:
        """构建月度样本索引: [(patch_id, year, month), ...].
        
        每个 patch 每个月只要有任意输入源有数据，就生成一个样本。
        V13-fix: 严格 2025-only，过滤掉 2023/2024 数据。
        Round 2: 同时保存每个 patch 的月份列表用于跨时相采样。
        """
        from datetime import datetime
        from collections import defaultdict
        
        samples = []
        self._patch_months: dict[str, list[tuple[int, int]]] = {}  # Round 2
        for patch_id in self.patches:
            # 收集该 patch 所有输入源的时间戳
            month_groups = defaultdict(list)
            
            for src_name in self.input_sources:
                src_dir = self._resolve_source_dir(src_name, patch_id)
                if src_dir is None or not src_dir.exists():
                    continue
                tif_files = sorted(src_dir.glob("*.tif"))
                for tf in tif_files:
                    ts = label_to_timestamp_ms(tf.stem)
                    if ts > 0:
                        dt = datetime.fromtimestamp(ts / 1000)
                        # ★ 支持 2025-2026 年月度数据
                        if dt.year in (2025, 2026):
                            month_groups[(dt.year, dt.month)].append(ts)
            
            months = sorted(month_groups.keys())
            self._patch_months[patch_id] = months
            # 每个月一个样本（只要有数据）
            for (year, month) in months:
                samples.append((patch_id, year, month))
        
        return samples
    
    def _discover_patches(self) -> list[str]:
        d = self.cfg.data
        # ★ 精确指定 patch 列表（最高优先级）
        patch_list = getattr(d, "patch_list", None)
        if patch_list:
            return sorted(patch_list)

        if self.data_root.suffix == ".json":
            with self.data_root.open("r") as f:
                manifest = json.load(f)
            return [r["sample_id"] for r in manifest]

        patch_ids: set[str] = set()
        if self.data_root.is_dir():
            for patch_dir in self.data_root.rglob("patch_*"):
                if patch_dir.is_dir():
                    patch_ids.add(patch_dir.name)
        patches = sorted(patch_ids)
        if not patches:
            raise FileNotFoundError(f"No patches found in {self.data_root}")
        return patches

    def _get_source_shape(self, source_name: str) -> tuple[int, int]:
        """多分辨率模式下：返回该源在统一地理范围内的像素尺寸。"""
        if source_name in self.source_image_sizes:
            shape = self.source_image_sizes[source_name]
            if isinstance(shape, int):
                return (shape, shape)
            return tuple(shape)
        gsd = self.source_gsd.get(source_name, 10.0)
        size = int(round(self.patch_size_m / gsd))
        return (size, size)

    def _compute_patch_bounds(self, patch_id: str) -> tuple[tuple[float, float, float, float], rasterio.crs.CRS]:
        """多分辨率模式下：以第一个可用输入源的 bounds 作为统一地理范围。"""
        if patch_id in self._patch_bounds:
            return self._patch_bounds[patch_id]
        for src_name in self.input_sources:
            src_dir = self._resolve_source_dir(src_name, patch_id)
            if src_dir is not None:
                tif_files = sorted(src_dir.glob("*.tif"))
                if tif_files:
                    with rasterio.open(tif_files[0]) as ds:
                        bounds = ds.bounds
                        crs = ds.crs
                        self._patch_bounds[patch_id] = (bounds, crs)
                        return bounds, crs
        raise FileNotFoundError(f"No input source found for patch {patch_id}")

    def _load_stats(self, stats_dir: Path | None) -> dict:
        stats: dict[str, dict[str, dict[str, float]]] = {}
        if stats_dir is None or not stats_dir.exists():
            return stats
        for stats_file in stats_dir.glob("*_stats.json"):
            source_name = stats_file.stem.replace("_stats", "")
            with stats_file.open("r") as f:
                stats[source_name] = json.load(f)
        return stats

    def _compute_sample_weights(self) -> None:
        """基于 S2 空间方差计算采样权重."""
        variances = []
        for pid in self.patches:
            # 新结构: patch_id / s2
            patch_s2 = self.data_root / pid / "s2"
            if not patch_s2.exists():
                # 旧结构: s2 / patch_id
                patch_s2 = self.data_root / "s2" / pid
                if not patch_s2.exists():
                    variances.append(0.0)
                    continue
            frame_vars = []
            for tif in sorted(patch_s2.glob("*.tif")):
                data = read_tif(tif, self.image_size)
                if data is not None:
                    frame_vars.append(float(np.mean([data[c].var() for c in range(data.shape[0])])))
            variances.append(float(np.mean(frame_vars)) if frame_vars else 0.0)
        var_arr = np.array(variances, dtype=np.float64)
        self._sample_weights = np.sqrt(var_arr + 1e-4)

    def _normalize(self, data: np.ndarray, source_name: str) -> np.ndarray:
        """归一化封装，调用 transforms.normalize_data."""
        return normalize_data(data, source_name, self.stats, self.num_classes)

    def _pad_channels(self, data: np.ndarray, target_dim: int) -> np.ndarray:
        current = data.shape[0]
        if current >= target_dim:
            return data[:target_dim]
        padded = np.zeros((target_dim, data.shape[1], data.shape[2]), dtype=data.dtype)
        padded[:current] = data
        return padded

    def _subsample_frames(self, frames: list, ts: list) -> tuple[list, list]:
        """超过 max_frames 时: 训练随机采样, 验证等间距."""
        n = len(frames)
        if n <= self.max_frames:
            return frames, ts
        if self.training:
            indices = sorted(random.sample(range(n), self.max_frames))
        else:
            step = max(1, n / self.max_frames)
            indices = [min(int(i * step), n - 1) for i in range(self.max_frames)]
        return [frames[i] for i in indices], [ts[i] for i in indices]

    def _preload_all(self) -> None:
        """预加载所有数据到内存 — 支持持久化缓存 (DDP 安全: 仅 rank 0 预加载)."""
        import time, hashlib

        # ★ 多分辨率模式：暂不预加载（缓存结构需重新设计），按需求加载
        if self.use_multires:
            print("[Dataset] 多分辨率模式：禁用内存预加载，按需求从磁盘读取")
            return

        # ★ 每个实验独立缓存：将实验名+关键采样参数纳入哈希，彻底避免并发竞争
        exp_name = getattr(self.cfg.experiment, 'name', 'unknown')
        cache_inputs = [
            exp_name,
            str(self.data_root),
            ",".join(self.patches),
            str(self.filter_2025_monthly),
            str(self.max_frames),
            str(self.image_size),
            ",".join(self.input_sources),
            ",".join([f"{n}:{lt}:{ss}" for n, lt, ss in self.target_sources]),
            str(self.merge_hr_into_lr),
        ]
        cache_key = hashlib.md5(
            "|".join(cache_inputs).encode()
        ).hexdigest()[:16]
        # 共享缓存目录（每个实验独立文件，无竞争）
        shared_cache_dir = Path("/workspace/xuannv/outputs/.cache_shared")
        shared_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = shared_cache_dir / f"dataset_cache_{cache_key}.pt"

        # 尝试加载已有缓存
        if cache_file.exists():
            start = time.time()
            ckpt = torch.load(cache_file, weights_only=False)
            self._cache = ckpt["cache"]
            print(f"[Dataset] Loaded cache from {cache_file} ({cache_file.stat().st_size/1e9:.1f}GB) in {time.time()-start:.1f}s")
            return

        # 检测是否在 DDP 环境中
        try:
            import torch.distributed as dist
            is_ddp = dist.is_initialized() and dist.get_world_size() > 1
            rank = dist.get_rank() if is_ddp else 0
        except Exception:
            is_ddp = False
            rank = 0

        if is_ddp and rank > 0:
            # rank > 0: 等待 rank 0 写完缓存文件，然后加载
            print(f"[Dataset] Rank {rank} waiting for cache file...")
            wait_start = time.time()
            while not cache_file.exists():
                time.sleep(2)
                if time.time() - wait_start > 600:
                    raise RuntimeError("Timeout waiting for dataset cache")
            # ★ 错开读取大缓存文件，避免所有 rank 同时 I/O 竞争
            stagger = rank * 3  # 每 rank 间隔 3 秒
            if stagger > 0:
                print(f"[Dataset] Rank {rank} staggering cache load by {stagger}s...")
                time.sleep(stagger)
            ckpt = torch.load(cache_file, weights_only=False)
            self._cache = ckpt["cache"]
            print(f"[Dataset] Rank {rank} loaded cache in {time.time()-wait_start:.1f}s")
            return

        # rank 0 (或非 DDP): 执行预加载并保存 (原子写入，防止 rank 1 读到不完整文件)
        start = time.time()
        n_cached = 0

        # ★ 多进程并行预加载 (DDP rank 0 也启用，其他 rank 只等待缓存，无冲突)
        use_parallel = len(self.patches) > 20
        if use_parallel:
            import os
            from concurrent.futures import ProcessPoolExecutor
            n_workers = min(16, os.cpu_count() or 1)
            worker_args = []
            for patch_id in self.patches:
                worker_args.append((
                    patch_id,
                    str(self.data_root),
                    self.input_sources,
                    self.target_sources,
                    self.image_size,
                    self.input_dim,
                    self.reconstruction_channels,
                    self.num_classes,
                    self.stats,
                    self.merge_hr_into_lr,
                    self.filter_2025_monthly,
                    self.max_frames,
                ))
            print(f"[Dataset] Parallel preloading with {n_workers} workers...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for patch_id, cache_entry, patch_n_cached in executor.map(_preload_patch_worker, worker_args):
                    self._cache[patch_id] = cache_entry
                    n_cached += patch_n_cached
        else:
            # 串行回退 (DDP 环境或少量 patch)
            for patch_id in self.patches:
                self._cache[patch_id] = {}
                for src_name in self.input_sources:
                    result = self._load_input_frames_impl(patch_id, src_name)
                    if len(result[0]) > 0:
                        self._cache[patch_id][src_name] = result
                        n_cached += 1
                for tgt_name, loss_type, sensor_src in self.target_sources:
                    if tgt_name in self._cache[patch_id]:
                        continue
                    if tgt_name in ("dem", "worldcover", "jrc_water", "dynamic_world"):
                        src_dir = self._resolve_source_dir(tgt_name, patch_id)
                        if src_dir is not None:
                            tif_files = sorted(src_dir.glob("*.tif"))
                            if tif_files:
                                data = read_tif(tif_files[0], 0)
                                if data is not None:
                                    if tgt_name == "jrc_water":
                                        data = data.astype(np.float32)
                                        data[data == -128] = np.nan
                                    data = self._normalize(data, tgt_name)
                                    data = self._pad_channels(data, self.reconstruction_channels)
                                    self._cache[patch_id][tgt_name] = (data[np.newaxis, ...], np.array([0.0]))
                                    n_cached += 1
        elapsed = time.time() - start
        print(f"[Dataset] Pre-loaded {len(self.patches)} patches, {n_cached} sources in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        save_start = time.time()
        tmp_file = cache_file.with_suffix(".tmp")
        torch.save({"cache": self._cache}, tmp_file)
        tmp_file.rename(cache_file)  # 原子重命名
        print(f"[Dataset] Saved cache to {cache_file} ({cache_file.stat().st_size/1e9:.1f}GB) in {time.time()-save_start:.1f}s")

    def _resolve_source_dir(self, source_name: str, patch_id: str) -> Path | None:
        """解析数据源目录，支持 data_root 的直接子目录嵌套结构.

        优先查找新结构 data_root / patch_id / source_name，
        再回退旧结构 data_root / source_name / patch_id，
        若都不存在则在 data_root 的直接子目录中搜索.
        """
        # 新结构: patch_id / source_name
        source_dir = self.data_root / patch_id / source_name
        if source_dir.exists():
            return source_dir
        # 旧结构: source_name / patch_id
        source_dir = self.data_root / source_name / patch_id
        if source_dir.exists():
            return source_dir
        # 在 data_root 的直接子目录中搜索（兼容 harbin/scenes/s2/patch_*）
        for sub_dir in self.data_root.iterdir():
            if not sub_dir.is_dir():
                continue
            candidate = sub_dir / source_name / patch_id
            if candidate.exists():
                return candidate
        return None

    def _load_input_frames(self, patch_id: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        """带缓存的输入帧加载."""
        if patch_id in self._cache and source_name in self._cache[patch_id]:
            return self._cache[patch_id][source_name]
        result = self._load_input_frames_impl(patch_id, source_name)
        # ★ 评估时多区域数据集可能未预加载，缓存按需读取避免重复磁盘 I/O
        if patch_id not in self._cache:
            self._cache[patch_id] = {}
        self._cache[patch_id][source_name] = result
        return result

    def _select_bands(self, data: np.ndarray, source_name: str) -> np.ndarray:
        """跨区域波段对齐 — 根据数据源选择统一的波段子集.
        
        Haidian S2: 6 bands = [B02, B03, B04, B05, B06, B07]
        Harbin  S2: 12 bands (PC标准), 取 [B02, B03, B04, B05, B06, B07] = idx [1,2,3,4,5,6]
        
        Haidian Landsat: 6 bands = [red(B4), green(B3), blue(B2), nir08(B5), swir16(B6), lwir11(B10)]
        Harbin  Landsat: 11 bands (PC标准), 取对应 idx [3,2,1,4,5,9]
        """
        n = data.shape[0]
        if source_name == "s2":
            if n == 12:
                return data[[1, 2, 3, 4, 5, 6]]  # B02-B07
            elif n > 6:
                return data[:6]
        elif source_name == "landsat":
            if n == 11:
                # PC顺序: B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11
                # 目标:    red(B4) green(B3) blue(B2) nir08(B5) swir16(B6) lwir11(B10)
                return data[[3, 2, 1, 4, 5, 9]]
            elif n > 6:
                return data[:6]
        return data

    def _load_input_frames_impl(self, patch_id: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        """从磁盘加载一个输入源的所有帧."""
        source_dir = self._resolve_source_dir(source_name, patch_id)
        tif_files = sorted(source_dir.glob("*.tif")) if source_dir is not None else []

        hr_name = None
        if self.merge_hr_into_lr and source_name == "s2":
            hr_name = "s2_hr"
        elif self.merge_hr_into_lr and source_name == "s1":
            hr_name = "s1_hr"

        hr_files = {}
        if hr_name:
            hr_dir = self._resolve_source_dir(hr_name, patch_id)
            if hr_dir is not None:
                for p in sorted(hr_dir.glob("*.tif")):
                    hr_files[p.stem] = p

        if not tif_files and not hr_files:
            if self.use_multires:
                h, w = self._get_source_shape(source_name)
                return (np.zeros((0, self.input_dim, h, w), dtype=np.float32),
                        np.zeros(0, dtype=np.float64))
            return (np.zeros((0, self.input_dim, self.image_size, self.image_size), dtype=np.float32),
                    np.zeros(0, dtype=np.float64))

        if self.filter_2025_monthly:
            def _is_valid_monthly_2025(path: Path) -> bool:
                stem = path.stem
                if "Q" in stem.upper():
                    return False
                if len(stem) == 8 and stem.isdigit():
                    year = int(stem[:4])
                    month = int(stem[4:6])
                    return year in (2025, 2026) and 1 <= month <= 12
                return False
            tif_files = [f for f in tif_files if _is_valid_monthly_2025(f)]

        frames_list: list[np.ndarray] = []
        timestamps: list[float] = []

        # ★ 多分辨率模式：按真实 GSD 重投影
        if self.use_multires:
            dst_bounds, dst_crs = self._compute_patch_bounds(patch_id)
            dst_shape = self._get_source_shape(source_name)
            resampling = "nearest" if source_name in CATEGORICAL_SOURCES else "bilinear"

        for tif_path in tif_files:
            stem = tif_path.stem
            if self.use_multires:
                data = read_tif_aligned(tif_path, dst_bounds, dst_shape, dst_crs, resampling=resampling)
            else:
                data = read_tif(tif_path, self.image_size)
            if data is None:
                continue
            # ★ 跨区域波段对齐
            data = self._select_bands(data, source_name)
            data = self._normalize(data, source_name)

            if hr_name and stem in hr_files:
                if self.use_multires:
                    hr_data = read_tif_aligned(hr_files[stem], dst_bounds, dst_shape, dst_crs, resampling=resampling)
                else:
                    hr_data = read_tif(hr_files[stem], self.image_size)
                if hr_data is not None:
                    hr_data = self._normalize(hr_data, hr_name)
                    data = np.concatenate([data, hr_data], axis=0)

            data = self._pad_channels(data, self.input_dim)
            frames_list.append(data)
            timestamps.append(float(label_to_timestamp_ms(stem)))

        if not frames_list:
            if self.use_multires:
                h, w = self._get_source_shape(source_name)
                return (np.zeros((0, self.input_dim, h, w), dtype=np.float32),
                        np.zeros(0, dtype=np.float64))
            return (np.zeros((0, self.input_dim, self.image_size, self.image_size), dtype=np.float32),
                    np.zeros(0, dtype=np.float64))

        return np.stack(frames_list), np.array(timestamps, dtype=np.float64)

    def _load_target_frame(self, patch_id: str, source_name: str):
        """加载单个目标帧."""
        source_dir = self._resolve_source_dir(source_name, patch_id)
        if source_dir is None:
            return None

        tif_files = sorted(source_dir.glob("*.tif"))
        if not tif_files:
            return None

        if source_name in ("dem", "worldcover", "jrc_water"):
            data = read_tif(tif_files[0], self.image_size)
            if data is not None:
                return self._normalize(data, source_name)
            return None

        return tif_files

    def _load_target_multires(
        self,
        patch_id: str,
        tgt_name: str,
        year: int,
        month_a: int,
        month_b: int,
        is_categorical: bool,
    ) -> np.ndarray | None:
        """多分辨率模式下加载单个目标源，保持其真实分辨率."""
        dst_bounds, dst_crs = self._compute_patch_bounds(patch_id)
        dst_shape = self._get_source_shape(tgt_name)
        resampling = "nearest" if is_categorical else "bilinear"

        if tgt_name == "dem":
            src_dir = self._resolve_source_dir("dem", patch_id)
            if src_dir is not None:
                tif_files = sorted(src_dir.glob("*.tif"))
                if tif_files:
                    data = read_tif_aligned(tif_files[0], dst_bounds, dst_shape, dst_crs, resampling=resampling)
                    if data is not None:
                        data = self._normalize(data, "dem")
                        return self._pad_channels(data, self.reconstruction_channels)
            return None

        if tgt_name in ("s2", "s1", "landsat", "tianyi_sar", "planet"):
            if month_b == month_a:
                # 同月：从已加载输入帧中随机选一帧作为目标
                if tgt_name in self.input_sources:
                    s_idx = self.input_sources.index(tgt_name)
                    frames, ts = self._load_monthly_frames(patch_id, tgt_name, year, month_a)
                    if len(frames) > 0:
                        random_idx = random.randint(0, len(frames) - 1)
                        data = frames[random_idx]
                        return self._pad_channels(data, self.reconstruction_channels)
            else:
                tgt_frames, tgt_ts = self._load_monthly_frames(patch_id, tgt_name, year, month_b)
                if len(tgt_frames) > 0:
                    random_idx = random.randint(0, len(tgt_frames) - 1)
                    data = tgt_frames[random_idx]
                    return self._pad_channels(data, self.reconstruction_channels)
            return None

        return None

    def _load_target_legacy(
        self,
        patch_id: str,
        tgt_name: str,
        year: int,
        month_a: int,
        month_b: int,
        source_frames: np.ndarray,
        source_mask: np.ndarray,
        is_categorical: bool,
    ) -> np.ndarray | None:
        """ legacy 模式加载单个目标源，与输入同分辨率."""
        if tgt_name == "dem":
            if tgt_name in self._cache.get(patch_id, {}):
                return self._cache[patch_id][tgt_name][0][0]
            src_dir = self.data_root / patch_id / tgt_name
            if not src_dir.exists():
                src_dir = self.data_root / tgt_name / patch_id
            if src_dir.exists():
                tif_files = sorted(src_dir.glob("*.tif"))
                if tif_files:
                    data = read_tif(tif_files[0], 0)
                    if data is not None:
                        return self._normalize(data, tgt_name)
            return None

        if tgt_name in ("s2", "s1", "landsat"):
            if month_b == month_a:
                if tgt_name in self.input_sources:
                    s_idx = self.input_sources.index(tgt_name)
                    valid_indices = [i for i in range(self.max_frames) if source_mask[s_idx, i]]
                    if valid_indices:
                        random_idx = random.choice(valid_indices)
                        data = source_frames[s_idx, random_idx].copy()
                        if len(valid_indices) > 1:
                            source_mask[random_idx] = False
                            source_frames[s_idx, random_idx] = 0.0
                        return data
            else:
                tgt_frames, tgt_ts = self._load_monthly_frames(patch_id, tgt_name, year, month_b)
                if len(tgt_frames) > 0:
                    random_idx = random.randint(0, len(tgt_frames) - 1)
                    return tgt_frames[random_idx]
            return None

        return None

    def _preload_teacher_tokens(self) -> None:
        """一次性把所有月份的 OlmoEarth teacher tokens 加载进内存（fp16）.

        此前 `_load_teacher_tokens` 每次 __getitem__ 都 `np.load(...)["tokens"]`，
        会把整份 ~1.3GB 数组读进内存再取一行，造成灾难性的 I/O / 内存抖动。
        改为主进程预加载，worker fork 时通过 COW 共享。

        ★ 关键：按 npz 自带的 patch_ids 建立 patch_id→行号 映射，
        而不是假设 self.patches 的枚举顺序与 npz 行序一致
        （否则 max_patches 随机采样 / 过滤会导致 teacher 张冠李戴）。
        
        ★ 键格式: year*100 + month，避免 2025-01 与 2026-01 冲突。
        """
        tokens_root = getattr(self.cfg.data, "olmoearth_tokens_root", None)
        if tokens_root is None:
            return
        from pathlib import Path
        root = Path(tokens_root)
        if not root.exists():
            return
        # 遍历月份目录：直接子目录是 2025 年月份，2026/ 子目录下是 2026 年月份
        month_dirs = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "2026":
                for sub in sorted(d.iterdir()):
                    if sub.is_dir() and sub.name.isdigit():
                        month_dirs.append((2026, int(sub.name), sub))
            elif d.name.isdigit() and len(d.name) <= 2:
                month_dirs.append((2025, int(d.name), d))
        for year, m, d in month_dirs:
            key = year * 100 + m
            tok_path = d / "spatial_tokens.npz"
            if tok_path.exists():
                try:
                    zd = np.load(str(tok_path))
                    self._teacher_tokens[key] = zd["tokens"].astype(np.float16)
                    self._teacher_tok_pid2row[key] = {
                        str(p): i for i, p in enumerate(zd["patch_ids"])
                    } if "patch_ids" in zd.files else None
                except Exception:
                    continue
            emb_path = d / "emb_all.npz"
            if emb_path.exists():
                try:
                    ed = np.load(str(emb_path))
                    if "embeddings" in ed.files:
                        self._teacher_global[key] = ed["embeddings"].astype(np.float32)
                        self._teacher_glb_pid2row[key] = {
                            str(p): i for i, p in enumerate(ed["patch_ids"])
                        } if "patch_ids" in ed.files else None
                except Exception:
                    pass
        self._teacher_months = sorted(self._teacher_tokens.keys())
        if self._teacher_months:
            n = next(iter(self._teacher_tokens.values())).shape[0]
            gb = sum(a.nbytes for a in self._teacher_tokens.values()) / 1e9
            print(f"[Dataset] OlmoEarth teacher tokens 预加载: {len(self._teacher_months)} 个月, "
                  f"{n} patches, {gb:.2f}GB (fp16, 内存常驻)")

    def _preload_aef_embeddings(self) -> None:
        """预加载 AEF (AlphaEarth Foundations) 64D 嵌入.
        
        AEF 嵌入按 patch 存储为 .npy 文件:
            aef_embed_dir/{patch_id}.npy  -> (64, H, W) float32
        """
        aef_dir = getattr(self.cfg.data, "aef_embed_dir", None)
        if aef_dir is None:
            return
        from pathlib import Path
        root = Path(aef_dir)
        if not root.exists():
            return
        
        self._aef_embeds: dict[str, np.ndarray] = {}
        loaded = 0
        for patch_id in self.patches:
            pid = patch_id["patch_id"] if isinstance(patch_id, dict) else str(patch_id)
            fpath = root / f"{pid}.npy"
            if fpath.exists():
                try:
                    self._aef_embeds[pid] = np.load(fpath).astype(np.float32)
                    loaded += 1
                except Exception:
                    pass
        if loaded > 0:
            print(f"[Dataset] AEF 嵌入预加载: {loaded}/{len(self.patches)} patches")
    
    def _load_aef_embedding(self, patch_id: str) -> dict:
        """加载 AEF 64D 嵌入（双教师蒸馏用）.
        
        返回:
            {"aef_spatial_emb": (64, H, W) tensor, "aef_global_emb": (64,) tensor}
        """
        emb = self._aef_embeds.get(patch_id) if hasattr(self, "_aef_embeds") else None
        if emb is None:
            return {}
        # emb: (64, H, W)
        result = {}
        # 空间嵌入
        result["aef_spatial_emb"] = torch.from_numpy(np.ascontiguousarray(emb))
        # 全局嵌入 (空间平均)
        result["aef_global_emb"] = torch.from_numpy(np.ascontiguousarray(emb.mean(axis=(1, 2))))
        return result

    def _load_teacher_tokens(self, patch_id: str, year: int, month: int) -> dict:
        """从内存缓存取 OlmoEarth teacher tokens（蒸馏用）.

        若未配置 / 无缓存，返回占位 tensor（训练照常进行）.
        若请求月份不存在，自动选最近可用月份（避免 batch collate 时 key 不一致）.
        按 patch_id 字符串匹配行号（顺序无关，安全）。
        """
        if not self._teacher_months:
            return {
                "teacher_spatial_tokens": torch.zeros((32, 32, 768), dtype=torch.float16),
                "teacher_global_emb": torch.zeros((768,), dtype=torch.float32),
            }
        # 键格式: year*100 + month
        req_key = year * 100 + month
        # 优先精确匹配
        if req_key in self._teacher_tokens:
            am = req_key
        else:
            # 找同一年中最近的月份
            same_year_keys = [k for k in self._teacher_months if k // 100 == year]
            if same_year_keys:
                am = min(same_year_keys, key=lambda a: abs(a - req_key))
            else:
                am = min(self._teacher_months, key=lambda a: abs(a - req_key))
        result: dict = {}
        toks = self._teacher_tokens.get(am)
        if toks is not None:
            pid2row = self._teacher_tok_pid2row.get(am)
            row = pid2row.get(patch_id) if pid2row is not None else self._patch_to_idx.get(patch_id)
            if row is not None and row < toks.shape[0]:
                result["teacher_spatial_tokens"] = torch.from_numpy(
                    np.ascontiguousarray(toks[row])
                )  # (32, 32, 768) fp16
        gemb = self._teacher_global.get(am)
        if gemb is not None:
            gpid2row = self._teacher_glb_pid2row.get(am)
            grow = gpid2row.get(patch_id) if gpid2row is not None else self._patch_to_idx.get(patch_id)
            if grow is not None and grow < gemb.shape[0]:
                result["teacher_global_emb"] = torch.from_numpy(
                    np.ascontiguousarray(gemb[grow])
                )  # (768,) fp32
        # ★ FIX: 确保所有样本有相同的键，避免 collate 失败
        if "teacher_spatial_tokens" not in result:
            result["teacher_spatial_tokens"] = torch.zeros((32, 32, 768), dtype=torch.float16)
        if "teacher_global_emb" not in result:
            result["teacher_global_emb"] = torch.zeros((768,), dtype=torch.float32)
        return result
    def _get_worldcover_label(self, patch_id: str) -> int:
        """从原始 WorldCover TIFF 提取 patch-level 众数类别标签."""
        try:
            wc_dir = self._resolve_source_dir("worldcover", patch_id)
            if wc_dir is not None:
                tif_files = list(wc_dir.glob("*.tif"))
                if tif_files:
                    data = read_tif(tif_files[0], 0)
                    if data is not None:
                        raw = data[0] if data.ndim == 3 else data
                        mapped = np.full_like(raw, -1, dtype=np.int64)
                        for val, idx in WC_CLASS_MAP.items():
                            mapped[raw == val] = idx
                        valid = mapped >= 0
                        if valid.any():
                            unique, counts = np.unique(mapped[valid], return_counts=True)
                            return int(unique[np.argmax(counts)])
        except Exception:
            pass
        return 0

    def _sample_long_gap_windows(self, ts_sorted: list[int]) -> tuple[float, float, float, float]:
        """长间隔窗口采样 (gap ≥ long_min_gap_ms)."""
        min_frames = getattr(self, '_min_window_frames', 4)
        max_frames = getattr(self, '_max_window_frames', 12)
        min_gap_ms = getattr(self, '_mixed_scale_long_min_gap_ms', 6 * 30 * 24 * 3600 * 1000)

        if len(ts_sorted) >= min_frames * 2:
            for _ in range(20):
                split_point = random.randint(min_frames, len(ts_sorted) - min_frames)
                early_frames = ts_sorted[:split_point]
                late_frames = ts_sorted[split_point:]

                w1_size = random.randint(min_frames, min(max_frames, len(early_frames)))
                w2_size = random.randint(min_frames, min(max_frames, len(late_frames)))

                w1_start_idx = random.randint(0, len(early_frames) - w1_size)
                w2_start_idx = random.randint(0, len(late_frames) - w2_size)

                cand_w1_start = float(early_frames[w1_start_idx])
                cand_w1_end = float(early_frames[w1_start_idx + w1_size - 1])
                cand_w2_start = float(late_frames[w2_start_idx])
                cand_w2_end = float(late_frames[w2_start_idx + w2_size - 1])

                center1 = (cand_w1_start + cand_w1_end) / 2.0
                center2 = (cand_w2_start + cand_w2_end) / 2.0
                if abs(center2 - center1) >= min_gap_ms:
                    return cand_w1_start, cand_w1_end, cand_w2_start, cand_w2_end
        # fallback
        if len(ts_sorted) >= min_frames * 2:
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[min_frames - 1])
            w2_start = float(ts_sorted[-min_frames])
            w2_end = float(ts_sorted[-1])
        elif len(ts_sorted) >= 2:
            mid = len(ts_sorted) // 2
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[mid - 1])
            w2_start = float(ts_sorted[mid])
            w2_end = float(ts_sorted[-1])
        else:
            t = float(ts_sorted[0]) if ts_sorted else 1672531200000.0
            return t, t, t, t
        return w1_start, w1_end, w2_start, w2_end

    def _sample_short_gap_windows(self, ts_sorted: list[int]) -> tuple[float, float, float, float]:
        """短间隔窗口采样: 选相邻/近邻的 1-3 个月合并为 w1/w2, gap ≤ 3个月."""
        from collections import defaultdict
        from datetime import datetime

        min_frames = getattr(self, '_min_window_frames', 4)
        short_max_gap_ms = getattr(self, '_mixed_scale_short_max_gap_ms', 3 * 30 * 24 * 3600 * 1000)

        # 按月份分组
        month_groups: dict[str, list[int]] = defaultdict(list)
        for ts in ts_sorted:
            dt = datetime.fromtimestamp(ts / 1000.0)
            key = f"{dt.year:04d}-{dt.month:02d}"
            month_groups[key].append(ts)

        months = sorted(month_groups.keys())

        # 策略: w1/w2 各包含 1-3 个月, 可以紧挨着或隔 0-1 个月
        if len(months) >= 2:
            for _ in range(50):
                w1_months = random.randint(1, 3)
                w2_months = random.randint(1, 3)
                gap_months = random.randint(0, 1)  # 0=紧挨, 1=隔1个月

                total_months_needed = w1_months + gap_months + w2_months
                if total_months_needed > len(months):
                    continue

                start_idx = random.randint(0, len(months) - total_months_needed)
                w2_start_idx = start_idx + w1_months + gap_months

                # 收集帧
                m1_frames = []
                for k in range(start_idx, start_idx + w1_months):
                    m1_frames.extend(month_groups[months[k]])

                m2_frames = []
                for k in range(w2_start_idx, w2_start_idx + w2_months):
                    m2_frames.extend(month_groups[months[k]])

                if len(m1_frames) >= min_frames and len(m2_frames) >= min_frames:
                    w1_start = float(m1_frames[0])
                    w1_end = float(m1_frames[-1])
                    w2_start = float(m2_frames[0])
                    w2_end = float(m2_frames[-1])

                    # 检查 center gap 是否在短间隔范围内
                    center1 = (w1_start + w1_end) / 2.0
                    center2 = (w2_start + w2_end) / 2.0
                    gap_ms = abs(center2 - center1)

                    if gap_ms <= short_max_gap_ms:
                        return w1_start, w1_end, w2_start, w2_end

        # fallback: 中点分割
        mid = len(ts_sorted) // 2
        return float(ts_sorted[0]), float(ts_sorted[mid - 1]), float(ts_sorted[mid]), float(ts_sorted[-1])

    def _sample_dual_windows(self, ts_sorted: list[int]) -> tuple[float, float, float, float]:
        """根据 window_mode 采样两个时间窗口.

        支持四种模式:
        - "adjacent_month": 选择相邻两个月的帧作为 w1/w2（与下游月度变化检测对齐）
        - "non_overlap": 非重叠长间隔窗口（原 v3 模式）
        - "random_split": 随机中点分割（默认）
        - "mixed_scale": 混合尺度 — 随机选择长间隔(≥6月)或短间隔(1-3月)
        """
        if len(ts_sorted) < 2:
            t = float(ts_sorted[0]) if ts_sorted else 1672531200000.0
            return t, t, t, t

        w1_start = w1_end = w2_start = w2_end = float(ts_sorted[0])

        if self.window_mode == "mixed_scale" and self.training:
            # 混合尺度: 以概率选择长间隔或短间隔
            r = random.random()
            long_prob = getattr(self, '_mixed_scale_long_prob', 0.5)
            if r < long_prob:
                # 长间隔模式: 使用 non_overlap 逻辑 (gap ≥ long_min_gap_ms)
                return self._sample_long_gap_windows(ts_sorted)
            else:
                # 短间隔模式: gap 在 [1月, short_max_gap_ms] 之间
                return self._sample_short_gap_windows(ts_sorted)

        if self.window_mode == "adjacent_month" and self.training:
            # 按月份分组 (YYYY-MM)
            from collections import defaultdict
            from datetime import datetime
            month_groups: dict[str, list[int]] = defaultdict(list)
            for ts in ts_sorted:
                dt = datetime.fromtimestamp(ts / 1000.0)
                key = f"{dt.year:04d}-{dt.month:02d}"
                month_groups[key].append(ts)

            months = sorted(month_groups.keys())
            if len(months) >= 2:
                # 随机选择一对相邻月份
                i = random.randint(0, len(months) - 2)
                m1_frames = month_groups[months[i]]
                m2_frames = month_groups[months[i + 1]]
                # 每边至少 min_frames 才用，否则 fallback
                min_f = getattr(self, '_min_window_frames', 4)
                if len(m1_frames) >= min_f and len(m2_frames) >= min_f:
                    w1_start = float(m1_frames[0])
                    w1_end = float(m1_frames[-1])
                    w2_start = float(m2_frames[0])
                    w2_end = float(m2_frames[-1])
                    return w1_start, w1_end, w2_start, w2_end
            # fallback: 使用默认分割
            mid = len(ts_sorted) // 2
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[mid - 1])
            w2_start = float(ts_sorted[mid])
            w2_end = float(ts_sorted[-1])
            return w1_start, w1_end, w2_start, w2_end

        # 非重叠模式 (原 v3)
        non_overlap = getattr(self, '_non_overlapping_windows', False)
        min_frames = getattr(self, '_min_window_frames', 4)
        max_frames = getattr(self, '_max_window_frames', 12)
        min_gap_ms = getattr(self, '_min_window_gap_ms', 6 * 30 * 24 * 3600 * 1000)

        if self.training and non_overlap and len(ts_sorted) >= min_frames * 2:
            for _ in range(20):
                split_point = random.randint(min_frames, len(ts_sorted) - min_frames)
                early_frames = ts_sorted[:split_point]
                late_frames = ts_sorted[split_point:]

                w1_size = random.randint(min_frames, min(max_frames, len(early_frames)))
                w2_size = random.randint(min_frames, min(max_frames, len(late_frames)))

                w1_start_idx = random.randint(0, len(early_frames) - w1_size)
                w2_start_idx = random.randint(0, len(late_frames) - w2_size)

                cand_w1_start = float(early_frames[w1_start_idx])
                cand_w1_end = float(early_frames[w1_start_idx + w1_size - 1])
                cand_w2_start = float(late_frames[w2_start_idx])
                cand_w2_end = float(late_frames[w2_start_idx + w2_size - 1])

                center1 = (cand_w1_start + cand_w1_end) / 2.0
                center2 = (cand_w2_start + cand_w2_end) / 2.0
                if abs(center2 - center1) >= min_gap_ms:
                    return cand_w1_start, cand_w1_end, cand_w2_start, cand_w2_end
            # fallback
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[min_frames - 1])
            w2_start = float(ts_sorted[-min_frames])
            w2_end = float(ts_sorted[-1])
            return w1_start, w1_end, w2_start, w2_end

        # random_split: 随机中点分割（默认）
        if len(ts_sorted) >= 4:
            # 随机选择分割点，确保两边至少有2个帧
            split_min = 2
            split_max = len(ts_sorted) - 2
            if split_min <= split_max:
                split_point = random.randint(split_min, split_max)
            else:
                split_point = len(ts_sorted) // 2
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[split_point - 1])
            w2_start = float(ts_sorted[split_point])
            w2_end = float(ts_sorted[-1])
        elif len(ts_sorted) >= 2:
            mid = len(ts_sorted) // 2
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[mid - 1])
            w2_start = float(ts_sorted[mid])
            w2_end = float(ts_sorted[-1])
        else:
            t = float(ts_sorted[0]) if ts_sorted else 1672531200000.0
            w1_start = w1_end = w2_start = w2_end = t

        return w1_start, w1_end, w2_start, w2_end

    def _resize_to_target(self, data: np.ndarray, target_res: int, is_categorical: bool = False, has_nan: bool = False) -> np.ndarray:
        """将数据 resize 到 target_res.

        Args:
            data: (C, H, W) numpy array.
            target_res: 目标分辨率.
            is_categorical: 是否为分类数据（最近邻）.
            has_nan: 是否包含 NaN（如 JRC Water 的 no-data），使用 nearest 避免扩散.
        Returns:
            (C, target_res, target_res) numpy array.
        """
        C, H, W = data.shape
        if H == target_res and W == target_res:
            return data
        import torch.nn.functional as F
        import torch
        t = torch.from_numpy(data).unsqueeze(0)  # (1, C, H, W)
        if is_categorical or has_nan:
            mode = "nearest"
            align = None
        elif H < target_res:
            mode = "bilinear"
            align = False
        else:
            mode = "area"
            align = None
        t = F.interpolate(t, size=(target_res, target_res), mode=mode, align_corners=align)
        return t.squeeze(0).numpy()

    def _generate_spatial_mask(self) -> np.ndarray | None:
        """生成随机空间掩码 [H, W]，用于跨时相掩码重建.

        以 patch_size × patch_size 的块为单位随机掩码。
        返回 1.0 表示可见，0.0 表示掩码。
        """
        if self.ct_mask_ratio <= 0 or self.ct_mask_patch_size <= 0:
            return None
        H = W = self.image_size
        p = self.ct_mask_patch_size
        grid_h = H // p
        grid_w = W // p
        total = grid_h * grid_w
        n_mask = max(1, int(total * self.ct_mask_ratio))
        mask = np.ones((H, W), dtype=np.float32)
        indices = random.sample(range(total), n_mask)
        for idx in indices:
            gh = idx // grid_w
            gw = idx % grid_w
            mask[gh * p:(gh + 1) * p, gw * p:(gw + 1) * p] = 0.0
        return mask

    def _generate_recon_mask(self) -> np.ndarray:
        """MAE-style block 重建掩码 [target_res, target_res].

        将图像分成 patch_size×patch_size 的 block，随机保留 visible_ratio 的 block 可见
        (mask=1.0)，其余 block 掩码 (mask=0.0)。

        Block 掩码比随机像素掩码更有效：decoder 无法利用邻近像素作弊，
        必须完全依赖 embedding → 更强的结构性约束。
        默认：75% 掩码、25% 可见（recon_mask_ratio=0.75）。
        """
        target_res = self.image_size
        ps = self.recon_mask_patch_size  # block 尺寸，默认 16
        # 将 image_size 对齐到 ps 的倍数
        n_patches_h = target_res // ps
        n_patches_w = target_res // ps
        total_patches = n_patches_h * n_patches_w

        n_visible = max(1, int(total_patches * self.recon_mask_visible_ratio))
        visible_idx = random.sample(range(total_patches), n_visible)
        visible_set = set(visible_idx)

        mask = np.zeros((target_res, target_res), dtype=np.float32)
        for pid in visible_set:
            ph = pid // n_patches_w
            pw = pid % n_patches_w
            h0, w0 = ph * ps, pw * ps
            mask[h0:h0 + ps, w0:w0 + ps] = 1.0

        # 若 image_size 不是 ps 整除，边缘区域默认可见
        if target_res % ps != 0:
            mask[n_patches_h * ps:, :] = 1.0
            mask[:, n_patches_w * ps:] = 1.0

        return mask

    def __len__(self) -> int:
        return len(self.monthly_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._get_item(idx)

    def _sample_target_month(self, patch_id: str, year: int, month_a: int) -> int | None:
        """Round 2: 为跨时相重建采样目标月份 month_B.
        
        从该 patch 的可用月份中随机选择，满足 |month_B - month_A| >= min_gap.
        若无满足条件的月份，返回 None.
        """
        months = self._patch_months.get(patch_id, [])
        if not months:
            return None
        available_months = [m for y, m in months if y == year]
        if not available_months:
            return None
        candidates = [m for m in available_months if abs(m - month_a) >= self.cross_temporal_min_gap]
        if not candidates:
            return None
        return random.choice(candidates)

    def _load_monthly_frames(self, patch_id: str, source_name: str, year: int, month: int) -> tuple[np.ndarray, np.ndarray]:
        """只加载指定自然月内的帧."""
        from datetime import datetime
        import calendar
        
        start_ms = datetime(year, month, 1).timestamp() * 1000
        last_day = calendar.monthrange(year, month)[1]
        end_ms = datetime(year, month, last_day, 23, 59, 59).timestamp() * 1000
        
        frames, ts = self._load_input_frames(patch_id, source_name)
        if len(frames) == 0:
            return frames, ts
        
        # 过滤当月
        mask = (ts >= start_ms) & (ts <= end_ms)
        return frames[mask], ts[mask]
    
    def _get_item(self, idx: int) -> dict[str, Any]:
        """V13: 月度采样 — 每个样本 = (patch, month), 输入/目标都是当月1帧.
        Round 2: 支持跨时相重建 (输入 month_A, 目标 month_B).
        多分辨率模式: source_frames / target_images 改为 List[Tensor].
        """
        patch_id, year, month_a = self.monthly_samples[idx]

        # Round 2: 决定是否使用跨时相采样
        month_b = month_a
        if self.training and self.cross_temporal and self.cross_temporal_prob > 0:
            if random.random() < self.cross_temporal_prob:
                month_b = self._sample_target_month(patch_id, year, month_a)
                if month_b is None:
                    month_b = month_a  # fallback

        month = month_a  # 输入月份
        S_inp = self.num_input_sources
        T = self.max_frames
        C = self.input_dim
        H = W = self.image_size
        S_tgt = self.num_target_sources
        M = self.metadata_dim

        # 1. 加载输入
        source_ts = np.zeros((S_inp, T), dtype=np.float64)
        source_mask = np.zeros((S_inp, T), dtype=bool)
        source_input_mask = np.ones(S_inp, dtype=bool)
        source_type_ids = np.zeros(S_inp, dtype=np.int64)
        all_monthly_ts: list[float] = []

        if self.use_multires:
            source_frames_list: list[np.ndarray] = []
            for s_idx, src_name in enumerate(self.input_sources):
                frames, ts = self._load_monthly_frames(patch_id, src_name, year, month)
                n_avail = len(frames)
                h, w = self._get_source_shape(src_name)
                if n_avail == 0:
                    source_input_mask[s_idx] = False
                    # 填充到 max_frames 个零帧，保证所有源 T 一致便于 stack
                    source_frames_list.append(np.zeros((T, C, h, w), dtype=np.float32))
                    continue
                n_use = min(n_avail, self.max_frames)
                if self.training and n_avail > n_use:
                    use_indices = sorted(random.sample(range(n_avail), n_use))
                else:
                    step = max(1, n_avail / n_use)
                    use_indices = [min(int(i * step), n_avail - 1) for i in range(n_use)]
                selected = frames[use_indices]
                selected_ts = ts[use_indices]
                # pad 到 T
                if n_use < T:
                    pad = np.zeros((T - n_use, C, h, w), dtype=np.float32)
                    selected = np.concatenate([selected, pad], axis=0)
                    pad_ts = np.zeros(T - n_use, dtype=np.float64)
                    selected_ts = np.concatenate([selected_ts, pad_ts])
                source_frames_list.append(selected)
                source_ts[s_idx, :] = selected_ts
                source_mask[s_idx, :n_use] = True
                all_monthly_ts.extend(float(t) for t in selected_ts[:n_use])
                source_type_ids[s_idx] = SOURCE_TYPE_MAP.get(src_name, 0)
            # 统一空间参考（用于增强 / mask）
            ref_h, ref_w = self.common_spatial_size
        else:
            source_frames = np.zeros((S_inp, T, C, H, W), dtype=np.float32)
            for s_idx, src_name in enumerate(self.input_sources):
                frames, ts = self._load_monthly_frames(patch_id, src_name, year, month)
                n_avail = len(frames)
                if n_avail == 0:
                    source_input_mask[s_idx] = False
                    continue
                n_use = min(n_avail, self.max_frames)
                if self.training and n_avail > n_use:
                    use_indices = sorted(random.sample(range(n_avail), n_use))
                else:
                    step = max(1, n_avail / n_use)
                    use_indices = [min(int(i * step), n_avail - 1) for i in range(n_use)]
                for i, fidx in enumerate(use_indices):
                    source_frames[s_idx, i] = frames[fidx]
                    source_ts[s_idx, i] = ts[fidx]
                    source_mask[s_idx, i] = True
                    all_monthly_ts.append(float(ts[fidx]))
                source_type_ids[s_idx] = SOURCE_TYPE_MAP.get(src_name, 0)
            ref_h, ref_w = H, W

        # 2. valid_period = 当月范围
        if all_monthly_ts:
            valid_start = float(min(all_monthly_ts))
            valid_end = float(max(all_monthly_ts))
        else:
            from datetime import datetime
            valid_start = datetime(year, month, 1).timestamp() * 1000
            import calendar
            valid_end = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59).timestamp() * 1000

        # 3. 构建目标
        target_relative_time = np.zeros(S_tgt, dtype=np.float32)
        target_metadata = np.zeros((S_tgt, M), dtype=np.float32)
        target_mask = np.zeros(S_tgt, dtype=bool)
        target_loss_type = np.zeros(S_tgt, dtype=np.int64)
        target_source_idx = np.zeros(S_tgt, dtype=np.int64)
        patch_label = self._get_worldcover_label(patch_id)

        if self.use_multires:
            target_images_list: list[np.ndarray] = []
            for t_idx, (tgt_name, loss_type, sensor_src) in enumerate(self.target_sources):
                is_categorical = loss_type == 1
                data = self._load_target_multires(patch_id, tgt_name, year, month_a, month_b, is_categorical)
                if data is not None:
                    target_images_list.append(data)
                    target_mask[t_idx] = True
                    target_loss_type[t_idx] = loss_type
                    target_source_idx[t_idx] = t_idx
                    gap_months = month_b - month_a
                    target_relative_time[t_idx] = gap_months / 12.0 + 0.5
                else:
                    # 占位空张量，保持 list 长度一致
                    h, w = self._get_source_shape(tgt_name) if tgt_name in self.source_gsd else (ref_h, ref_w)
                    ch = max(self.reconstruction_channels, self.num_classes)
                    target_images_list.append(np.zeros((ch, h, w), dtype=np.float32))
        else:
            target_res = H
            tgt_ch = max(self.reconstruction_channels, self.num_classes)
            target_images = np.zeros((S_tgt, tgt_ch, target_res, target_res), dtype=np.float32)
            for t_idx, (tgt_name, loss_type, sensor_src) in enumerate(self.target_sources):
                is_categorical = loss_type == 1
                data = self._load_target_legacy(patch_id, tgt_name, year, month_a, month_b, source_frames, source_mask, is_categorical)
                if data is not None:
                    data = self._pad_channels(data, self.reconstruction_channels)
                    data = self._resize_to_target(data, target_res, is_categorical=is_categorical)
                    target_images[t_idx, :data.shape[0]] = data
                    target_mask[t_idx] = True
                    target_loss_type[t_idx] = loss_type
                    target_source_idx[t_idx] = t_idx
                    gap_months = month_b - month_a
                    target_relative_time[t_idx] = gap_months / 12.0 + 0.5

        # 4. 空间增强（同步应用到所有源/目标）
        if self.training:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if self.use_multires:
                for i in range(len(source_frames_list)):
                    if flip_h:
                        source_frames_list[i] = source_frames_list[i][..., ::-1, :].copy()
                    if flip_v:
                        source_frames_list[i] = source_frames_list[i][..., ::-1].copy()
                for i in range(len(target_images_list)):
                    if flip_h:
                        target_images_list[i] = target_images_list[i][..., ::-1, :].copy()
                    if flip_v:
                        target_images_list[i] = target_images_list[i][..., ::-1].copy()
            else:
                if flip_h:
                    source_frames = source_frames[..., ::-1, :].copy()
                    target_images = target_images[..., ::-1, :].copy()
                if flip_v:
                    source_frames = source_frames[..., ::-1].copy()
                    target_images = target_images[..., ::-1].copy()

        # 5. 双时间窗口
        from datetime import datetime as _dt
        current_ts = sorted(int(t) for t in all_monthly_ts if t > 0)
        _m3_before_y, _m3_before_m = (year, month - 3) if month > 3 else (year - 1, month + 9)
        _m3_after_y, _m3_after_m = (year, month + 3) if month <= 9 else (year + 1, month - 9)
        _anchor_before = int(_dt(_m3_before_y, _m3_before_m, 15).timestamp() * 1000)
        _anchor_after = int(_dt(_m3_after_y, _m3_after_m, 15).timestamp() * 1000)
        ts_sorted = sorted([_anchor_before] + current_ts + [_anchor_after])
        w1_start, w1_end, w2_start, w2_end = self._sample_dual_windows(ts_sorted)

        # 6. 跨时相空间掩码 / 重建掩码
        spatial_mask = self._generate_spatial_mask() if self.training else None
        recon_mask = self._generate_recon_mask() if self.training else None

        # Round 2: 目标时间范围
        if month_b != month_a:
            from datetime import datetime
            import calendar
            target_valid_start = datetime(year, month_b, 1).timestamp() * 1000
            target_valid_end = datetime(year, month_b, calendar.monthrange(year, month_b)[1], 23, 59, 59).timestamp() * 1000
        else:
            target_valid_start = valid_start
            target_valid_end = valid_end

        result: dict[str, Any] = {
            "patch_id": patch_id,
            "year_month": (year, month),
            "source_timestamps_ms": torch.from_numpy(source_ts),
            "source_frame_mask": torch.from_numpy(source_mask),
            "source_input_mask": torch.from_numpy(source_input_mask),
            "source_type_ids": torch.from_numpy(source_type_ids),
            "valid_start_ms": torch.tensor(valid_start, dtype=torch.float64),
            "valid_end_ms": torch.tensor(valid_end, dtype=torch.float64),
            "valid_start_w1": torch.tensor(w1_start, dtype=torch.float64),
            "valid_end_w1": torch.tensor(w1_end, dtype=torch.float64),
            "valid_start_w2": torch.tensor(w2_start, dtype=torch.float64),
            "valid_end_w2": torch.tensor(w2_end, dtype=torch.float64),
            "target_valid_start_ms": torch.tensor(target_valid_start, dtype=torch.float32),
            "target_valid_end_ms": torch.tensor(target_valid_end, dtype=torch.float32),
            "spatial_mask": torch.from_numpy(spatial_mask) if spatial_mask is not None else torch.ones((ref_h, ref_w), dtype=torch.float32),
            "recon_mask": torch.from_numpy(recon_mask) if recon_mask is not None else torch.ones((ref_h, ref_w), dtype=torch.float32),
            "target_relative_time": torch.from_numpy(target_relative_time),
            "target_metadata": torch.from_numpy(target_metadata),
            "target_mask": torch.from_numpy(target_mask),
            "target_loss_type": torch.from_numpy(target_loss_type),
            "target_source_idx": torch.from_numpy(target_source_idx),
            "label": torch.tensor(patch_label, dtype=torch.long),
            "patch_index": torch.tensor(self._patch_to_idx.get(patch_id, 0), dtype=torch.long),
            **self._load_teacher_tokens(patch_id, year, month_a),
            **self._load_aef_embedding(patch_id),
        }
        if self.use_multires:
            result["source_frames"] = [torch.from_numpy(f) for f in source_frames_list]
            result["target_images"] = [torch.from_numpy(t) for t in target_images_list]
        else:
            result["source_frames"] = torch.from_numpy(source_frames)
            result["target_images"] = torch.from_numpy(target_images)
        return result
