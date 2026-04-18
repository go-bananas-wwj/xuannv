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
import torch
from torch.utils.data import Dataset

from src.data.transforms import (
    INPUT_SOURCES,
    TARGET_SOURCES,
    SOURCE_TYPE_MAP,
    label_to_timestamp_ms,
    read_tif,
    normalize_data,
)


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

        self.temporal_window_augmentation = d.temporal_window_augmentation
        self.temporal_window_prob = d.temporal_window_prob
        self.temporal_window_min_frames = d.temporal_window_min_frames
        self.temporal_window_max_frames = d.temporal_window_max_frames

        # 双窗口采样模式: "random_split" | "adjacent_month" | "non_overlap"
        self.window_mode = getattr(d, "window_mode", "random_split")
        # 跨时相掩码重建配置
        self.ct_mask_ratio = getattr(d, "ct_mask_ratio", 0.3)   # 掩码比例
        self.ct_mask_patch_size = getattr(d, "ct_mask_patch_size", 8)  # 掩码 patch 尺寸

        self.patches = self._discover_patches()
        stats_dir = Path(d.stats_dir) if d.stats_dir else None
        self.stats = self._load_stats(stats_dir)

        self._sample_weights: np.ndarray | None = None
        if self.variance_weighted:
            self._compute_sample_weights()

        # ★ 内存预加载: 避免每个 epoch 重复从磁盘读取 GeoTIFF
        self._cache: dict[str, dict[str, tuple]] = {}
        self._preload_all()

    def _discover_patches(self) -> list[str]:
        if self.data_root.suffix == ".json":
            with self.data_root.open("r") as f:
                manifest = json.load(f)
            return [r["sample_id"] for r in manifest]

        patch_ids: set[str] = set()
        if self.data_root.is_dir():
            for src_dir in self.data_root.iterdir():
                if not src_dir.is_dir():
                    continue
                for patch_dir in src_dir.iterdir():
                    if patch_dir.is_dir() and patch_dir.name.startswith("patch_"):
                        patch_ids.add(patch_dir.name)
        patches = sorted(patch_ids)
        if not patches:
            raise FileNotFoundError(f"No patches found in {self.data_root}")
        return patches

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
        s2_dir = self.data_root / "s2"
        if not s2_dir.exists():
            return
        variances = []
        for pid in self.patches:
            patch_s2 = s2_dir / pid
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
        cache_key = hashlib.md5(
            (str(self.data_root) + ",".join(self.patches)).encode()
        ).hexdigest()[:16]
        cache_file = Path("/workspace/outputs/aef_qwen_v4_cd_upgrade") / f"dataset_cache_{cache_key}.pt"

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
            ckpt = torch.load(cache_file, weights_only=False)
            self._cache = ckpt["cache"]
            print(f"[Dataset] Rank {rank} loaded cache in {time.time()-wait_start:.1f}s")
            return

        # rank 0 (或非 DDP): 执行预加载并保存 (原子写入，防止 rank 1 读到不完整文件)
        start = time.time()
        n_cached = 0
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
                if tgt_name in ("dem", "worldcover", "jrc_water"):
                    src_dir = self.data_root / tgt_name / patch_id
                    if src_dir.exists():
                        tif_files = sorted(src_dir.glob("*.tif"))
                        if tif_files:
                            data = read_tif(tif_files[0], 0)
                            if data is not None:
                                data = self._normalize(data, tgt_name)
                                data = self._pad_channels(data, self.reconstruction_channels)
                                self._cache[patch_id][tgt_name] = (data[np.newaxis, ...], np.array([0.0]))
                                n_cached += 1
                elif tgt_name == "dynamic_world":
                    src_dir = self.data_root / tgt_name / patch_id
                    if src_dir.exists():
                        tif_files = sorted(src_dir.glob("*.tif"))
                        frames_list = []
                        ts_list = []
                        for tf in tif_files:
                            d = read_tif(tf, self.image_size)
                            if d is not None:
                                d = self._normalize(d, tgt_name)
                                d = self._pad_channels(d, self.reconstruction_channels)
                                frames_list.append(d)
                                ts_list.append(float(label_to_timestamp_ms(tf.stem)))
                        if frames_list:
                            self._cache[patch_id][tgt_name] = (np.stack(frames_list), np.array(ts_list, dtype=np.float64))
                            n_cached += 1
        elapsed = time.time() - start
        print(f"[Dataset] Pre-loaded {len(self.patches)} patches, {n_cached} sources in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        save_start = time.time()
        tmp_file = cache_file.with_suffix(".tmp")
        torch.save({"cache": self._cache}, tmp_file)
        tmp_file.rename(cache_file)  # 原子重命名
        print(f"[Dataset] Saved cache to {cache_file} ({cache_file.stat().st_size/1e9:.1f}GB) in {time.time()-save_start:.1f}s")

    def _load_input_frames(self, patch_id: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        """带缓存的输入帧加载."""
        if patch_id in self._cache and source_name in self._cache[patch_id]:
            return self._cache[patch_id][source_name]
        return self._load_input_frames_impl(patch_id, source_name)

    def _load_input_frames_impl(self, patch_id: str, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        """从磁盘加载一个输入源的所有帧 (原始逻辑)."""
        source_dir = self.data_root / source_name / patch_id
        tif_files = sorted(source_dir.glob("*.tif")) if source_dir.exists() else []

        hr_name = None
        if self.merge_hr_into_lr and source_name == "s2":
            hr_name = "s2_hr"
        elif self.merge_hr_into_lr and source_name == "s1":
            hr_name = "s1_hr"

        hr_files = {}
        if hr_name:
            hr_dir = self.data_root / hr_name / patch_id
            if hr_dir.exists():
                for p in sorted(hr_dir.glob("*.tif")):
                    hr_files[p.stem] = p

        if not tif_files and not hr_files:
            return (np.zeros((0, self.input_dim, self.image_size, self.image_size), dtype=np.float32),
                    np.zeros(0, dtype=np.float64))

        if self.filter_2025_monthly:
            def _is_valid_monthly_2025(path: Path) -> bool:
                stem = path.stem
                if "Q" in stem.upper():
                    return False
                if len(stem) == 8 and stem.isdigit() and stem.startswith("2025"):
                    month = int(stem[4:6])
                    return 4 <= month <= 10
                return False
            tif_files = [f for f in tif_files if _is_valid_monthly_2025(f)]

        frames_list: list[np.ndarray] = []
        timestamps: list[float] = []

        for tif_path in tif_files:
            stem = tif_path.stem
            data = read_tif(tif_path, self.image_size)
            if data is None:
                continue
            data = self._normalize(data, source_name)

            if hr_name and stem in hr_files:
                hr_data = read_tif(hr_files[stem], self.image_size)
                if hr_data is not None:
                    hr_data = self._normalize(hr_data, hr_name)
                    data = np.concatenate([data, hr_data], axis=0)

            data = self._pad_channels(data, self.input_dim)
            frames_list.append(data)
            timestamps.append(float(label_to_timestamp_ms(stem)))

        if not frames_list:
            return (np.zeros((0, self.input_dim, self.image_size, self.image_size), dtype=np.float32),
                    np.zeros(0, dtype=np.float64))

        return np.stack(frames_list), np.array(timestamps, dtype=np.float64)

    def _load_target_frame(self, patch_id: str, source_name: str):
        """加载单个目标帧."""
        source_dir = self.data_root / source_name / patch_id
        if not source_dir.exists():
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

    def _sample_dual_windows(self, ts_sorted: list[int]) -> tuple[float, float, float, float]:
        """根据 window_mode 采样两个时间窗口.

        支持三种模式:
        - "adjacent_month": 选择相邻两个月的帧作为 w1/w2（与下游月度变化检测对齐）
        - "non_overlap": 非重叠长间隔窗口（原 v3 模式）
        - "random_split": 随机中点分割（默认）
        """
        if len(ts_sorted) < 2:
            t = float(ts_sorted[0]) if ts_sorted else 1672531200000.0
            return t, t, t, t

        w1_start = w1_end = w2_start = w2_end = float(ts_sorted[0])

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

        # 默认: 中点分割
        if len(ts_sorted) >= 4:
            mid = len(ts_sorted) // 2
            w1_start = float(ts_sorted[0])
            w1_end = float(ts_sorted[mid - 1])
            w2_start = float(ts_sorted[mid])
            w2_end = float(ts_sorted[-1])
        else:
            t = float(ts_sorted[0])
            w1_start = w1_end = w2_start = w2_end = t

        return w1_start, w1_end, w2_start, w2_end

    def _resize_to_target(self, data: np.ndarray, target_res: int, is_categorical: bool = False) -> np.ndarray:
        """将数据 resize 到 target_res.

        Args:
            data: (C, H, W) numpy array.
            target_res: 目标分辨率.
            is_categorical: 是否为分类数据（最近邻）.
        Returns:
            (C, target_res, target_res) numpy array.
        """
        C, H, W = data.shape
        if H == target_res and W == target_res:
            return data
        import torch.nn.functional as F
        import torch
        t = torch.from_numpy(data).unsqueeze(0)  # (1, C, H, W)
        if is_categorical:
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

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._get_item(idx)

    def _get_item(self, idx: int) -> dict[str, torch.Tensor]:
        patch_id = self.patches[idx]
        S_inp = self.num_input_sources
        T = self.max_frames
        C = self.input_dim
        H = W = self.image_size
        S_tgt = self.num_target_sources
        M = self.metadata_dim

        # 1. 加载输入
        all_timestamps: list[float] = []
        source_frames = np.zeros((S_inp, T, C, H, W), dtype=np.float32)
        source_ts = np.zeros((S_inp, T), dtype=np.float64)
        source_mask = np.zeros((S_inp, T), dtype=bool)
        source_input_mask = np.ones(S_inp, dtype=bool)
        source_type_ids = np.zeros(S_inp, dtype=np.int64)

        for s_idx, src_name in enumerate(self.input_sources):
            frames, ts = self._load_input_frames(patch_id, src_name)
            n_avail = len(frames)

            if n_avail == 0:
                source_input_mask[s_idx] = False
                continue

            frames, ts = self._subsample_frames(list(frames), list(ts))
            n_use = len(frames)
            source_frames[s_idx, :n_use] = frames
            source_ts[s_idx, :n_use] = ts
            source_mask[s_idx, :n_use] = True
            all_timestamps.extend(ts)
            source_type_ids[s_idx] = SOURCE_TYPE_MAP.get(src_name, 0)

        # 2. 确定 valid_period
        ts_sorted = sorted(set(int(t) for t in all_timestamps if t > 0))
        if len(ts_sorted) >= 2:
            duration = ts_sorted[-1] - ts_sorted[0]
            valid_start = float(ts_sorted[0] + int(0.1 * duration))
            valid_end = float(ts_sorted[-1] - int(0.1 * duration))
        elif len(ts_sorted) == 1:
            valid_start = valid_end = float(ts_sorted[0])
        else:
            valid_start = valid_end = 1672531200000.0

        # 3. 时序窗口增强
        if (self.training and self.temporal_window_augmentation
                and random.random() < self.temporal_window_prob and len(ts_sorted) >= 2):
            min_f = max(2, self.temporal_window_min_frames)
            max_f = min(self.temporal_window_max_frames, len(ts_sorted) - 1)
            if min_f <= max_f:
                n_sel = random.randint(min_f, max_f)
                max_start = max(0, len(ts_sorted) - n_sel)
                if max_start >= 0:
                    si = random.randint(0, max_start)
                    ei = min(si + n_sel, len(ts_sorted) - 1)
                    valid_start = float(ts_sorted[si])
                    valid_end = float(ts_sorted[ei])

        # 4. 选择目标帧
        valid_ts_set = set(int(t) for t in all_timestamps if t > 0)
        valid_in_range = [t for t in valid_ts_set if valid_start <= t <= valid_end]
        if valid_in_range and self.training:
            target_ts = float(random.choice(valid_in_range))
        elif valid_in_range:
            target_ts = float(valid_in_range[len(valid_in_range) // 2])
        else:
            target_ts = valid_start

        # 5. 构建目标
        target_res = H // 2
        tgt_ch = max(self.reconstruction_channels, self.num_classes)
        target_images = np.zeros((S_tgt, tgt_ch, target_res, target_res), dtype=np.float32)
        target_relative_time = np.zeros(S_tgt, dtype=np.float32)
        target_metadata = np.zeros((S_tgt, M), dtype=np.float32)
        target_mask = np.zeros(S_tgt, dtype=bool)
        target_loss_type = np.zeros(S_tgt, dtype=np.int64)
        target_source_idx = np.zeros(S_tgt, dtype=np.int64)

        support_duration = max(ts_sorted[-1] - ts_sorted[0], 1) if len(ts_sorted) >= 2 else 1

        for t_idx, (tgt_name, loss_type, sensor_src) in enumerate(self.target_sources):
            is_categorical = loss_type == 1
            if tgt_name in ("dem", "worldcover", "jrc_water"):
                if tgt_name in self._cache.get(patch_id, {}):
                    data = self._cache[patch_id][tgt_name][0][0]
                    data = self._resize_to_target(data, target_res, is_categorical=is_categorical)
                    target_images[t_idx, :data.shape[0]] = data
                    target_mask[t_idx] = True
                    target_loss_type[t_idx] = loss_type
                    target_source_idx[t_idx] = t_idx
                    target_relative_time[t_idx] = 0.5
                else:
                    src_dir = self.data_root / tgt_name / patch_id
                    if src_dir.exists():
                        tif_files = sorted(src_dir.glob("*.tif"))
                        if tif_files:
                            data = read_tif(tif_files[0], 0)
                            if data is not None:
                                data = self._normalize(data, tgt_name)
                                data = self._resize_to_target(data, target_res, is_categorical=is_categorical)
                                target_images[t_idx, :data.shape[0]] = data
                                target_mask[t_idx] = True
                                target_loss_type[t_idx] = loss_type
                                target_source_idx[t_idx] = t_idx
                                target_relative_time[t_idx] = 0.5
            elif tgt_name == "dynamic_world":
                if tgt_name in self._cache.get(patch_id, {}):
                    frames_arr, ts_arr = self._cache[patch_id][tgt_name]
                    closest_idx = int(np.argmin(np.abs(ts_arr - target_ts)))
                    data = frames_arr[closest_idx]
                    data = self._resize_to_target(data, target_res, is_categorical=is_categorical)
                    target_images[t_idx, :data.shape[0]] = data
                    target_mask[t_idx] = True
                    target_loss_type[t_idx] = loss_type
                    target_source_idx[t_idx] = t_idx
                    rel_t = (ts_arr[closest_idx] - ts_sorted[0]) / support_duration if len(ts_sorted) >= 2 else 0.5
                    target_relative_time[t_idx] = float(np.clip(rel_t, 0.0, 1.0))
                else:
                    src_dir = self.data_root / tgt_name / patch_id
                    if src_dir.exists():
                        tif_files = sorted(src_dir.glob("*.tif"))
                        if tif_files:
                            frames_list = []
                            ts_list = []
                            for tf in tif_files:
                                d = read_tif(tf, H)
                                if d is not None:
                                    d = self._normalize(d, tgt_name)
                                    d = self._resize_to_target(d, target_res, is_categorical=is_categorical)
                                    frames_list.append(d)
                                    ts_list.append(float(label_to_timestamp_ms(tf.stem)))
                            if frames_list:
                                frames_arr = np.stack(frames_list)
                                ts_arr = np.array(ts_list, dtype=np.float64)
                                closest_idx = int(np.argmin(np.abs(ts_arr - target_ts)))
                                target_images[t_idx, :frames_arr.shape[1]] = frames_arr[closest_idx]
                                target_mask[t_idx] = True
                                target_loss_type[t_idx] = loss_type
                                target_source_idx[t_idx] = t_idx
                                rel_t = (ts_arr[closest_idx] - ts_sorted[0]) / support_duration if len(ts_sorted) >= 2 else 0.5
                                target_relative_time[t_idx] = float(np.clip(rel_t, 0.0, 1.0))
            else:
                # s2/s1/landsat 作为目标: 复用已缓存的输入帧, 避免重复磁盘 IO
                frames, ts = self._load_input_frames(patch_id, tgt_name)
                if len(frames) > 0:
                    closest_idx = int(np.argmin(np.abs(ts - target_ts)))
                    data = frames[closest_idx]
                    data = self._pad_channels(data, self.reconstruction_channels)
                    data = self._resize_to_target(data, target_res, is_categorical=is_categorical)
                    target_images[t_idx, :data.shape[0]] = data
                    target_mask[t_idx] = True
                    target_loss_type[t_idx] = loss_type
                    target_source_idx[t_idx] = t_idx
                    rel_t = (ts[closest_idx] - ts_sorted[0]) / support_duration if len(ts_sorted) >= 2 else 0.5
                    target_relative_time[t_idx] = float(np.clip(rel_t, 0.0, 1.0))

            window_pos = (target_ts - valid_start) / max(valid_end - valid_start, 1)
            window_width = (valid_end - valid_start) / support_duration if support_duration > 0 else 0
            target_metadata[t_idx, 0] = float(np.clip(window_pos, 0, 1))
            target_metadata[t_idx, 1] = float(np.clip(window_width, 0, 1))

        # 6. 空间增强
        if self.training:
            if random.random() < 0.5:
                source_frames = source_frames[..., ::-1, :].copy()
                target_images = target_images[..., ::-1, :].copy()
            if random.random() < 0.5:
                source_frames = source_frames[..., ::-1].copy()
                target_images = target_images[..., ::-1].copy()

        # 7. 双时间窗口生成
        w1_start, w1_end, w2_start, w2_end = self._sample_dual_windows(ts_sorted)

        # 8. 跨时相空间掩码（用于掩码重建任务）
        spatial_mask = self._generate_spatial_mask() if self.training else None

        return {
            "patch_id": patch_id,
            "source_frames": torch.from_numpy(source_frames),
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
            "spatial_mask": torch.from_numpy(spatial_mask) if spatial_mask is not None else torch.ones((self.image_size, self.image_size), dtype=torch.float32),
            "target_images": torch.from_numpy(target_images),
            "target_relative_time": torch.from_numpy(target_relative_time),
            "target_metadata": torch.from_numpy(target_metadata),
            "target_mask": torch.from_numpy(target_mask),
            "target_loss_type": torch.from_numpy(target_loss_type),
            "target_source_idx": torch.from_numpy(target_source_idx),
            "label": torch.tensor(0, dtype=torch.long),
        }
