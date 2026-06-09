"""海淀区数据集适配 AEF 格式."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.aef.data.transforms import read_tif, normalize_source, parse_date_to_ms


class HaidianAEFDataset(Dataset):
    """
    海淀区多源数据集，适配 AEF 输入格式 (B, T, H, W, C)。
    同时加载预计算的 AEF 官方 64D embedding 作为蒸馏监督。
    """

    def __init__(
        self,
        data_root: str = "data_raw/haidian/scenes",
        planet_root: str = "data_raw/beijing/planetscene",
        stats_dir: str = "statistics/haidian",
        cache_dir: str = "src/aef/.cache",
        image_size: int = 128,
        source_names: list[str] | None = None,
        required_sources: list[str] | None = None,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        max_frames: int = 16,
        aef_embedding_root: str = "data_raw/haidian/aef_embeddings/haidian_2025_patches",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.planet_root = Path(planet_root)
        self.image_size = image_size
        self.start_date = start_date
        self.end_date = end_date
        self.source_names = source_names or ["tianyi_sar", "s2", "landsat", "planet"]
        self.split = split
        self.max_frames = max_frames
        self.aef_embedding_root = Path(aef_embedding_root)

        # 加载时间映射表
        mapping_path = Path(cache_dir) / "temporal_mapping.json"
        if not mapping_path.exists():
            print(f"[Dataset] Temporal mapping not found, building...")
            import fcntl
            lock_path = mapping_path.with_suffix(".json.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lockfile:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                if not mapping_path.exists():
                    from src.aef.scripts.build_temporal_mapping import build_mapping
                    build_mapping(
                        data_root=str(self.data_root),
                        planet_root=str(self.planet_root),
                        output_path=str(mapping_path),
                    )
                fcntl.flock(lockfile, fcntl.LOCK_UN)
        with open(mapping_path) as f:
            self.mapping = json.load(f)

        self.patch_ids = sorted(self.mapping.keys())

        # 划分 train/val
        rng = random.Random(seed)
        rng.shuffle(self.patch_ids)
        n_train = int(len(self.patch_ids) * train_ratio)
        if split == "train":
            self.patch_ids = self.patch_ids[:n_train]
        else:
            self.patch_ids = self.patch_ids[n_train:]

        # 加载统计量
        self.stats = {}
        for src in self.source_names:
            stats_path = Path(stats_dir) / f"{src}_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    self.stats[src] = json.load(f)
            else:
                self.stats[src] = {}

        self.required_sources = required_sources or self.source_names

        # 快速预检查：过滤掉在时间窗口内缺少必需源的 patch
        self.samples = []
        for patch_id in self.patch_ids:
            has_all = True
            for src in self.required_sources:
                if src == "planet":
                    src_dir = self.planet_root / patch_id
                else:
                    src_dir = self.data_root / patch_id / src
                if not src_dir.exists():
                    has_all = False
                    break
                files = sorted(src_dir.glob("*.tif"))
                has_in_window = False
                for f in files:
                    date_str = f.stem
                    if self.start_date and date_str < self.start_date:
                        continue
                    if self.end_date and date_str > self.end_date:
                        continue
                    has_in_window = True
                    break
                if not has_in_window:
                    has_all = False
                    break
            if has_all:
                self.samples.append({"patch_id": patch_id})

        print(f"[{split}] {len(self.patch_ids)} patches, {len(self.samples)} valid samples (required: {self.required_sources})")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _remap_worldcover(data: np.ndarray) -> np.ndarray:
        """ESA WorldCover 编码 -> 0-based 类别索引."""
        mapping = {
            10: 0,   # Tree cover
            20: 1,   # Shrubland
            30: 2,   # Grassland
            40: 3,   # Cropland
            50: 4,   # Built-up
            60: 5,   # Bare / sparse vegetation
            80: 6,   # Permanent water bodies
            90: 7,   # Herbaceous wetland
            95: 8,   # Mangroves
            100: 9,  # Moss and lichen
        }
        out = np.full_like(data, 255, dtype=np.int64)
        for esa_val, cls_idx in mapping.items():
            out[data == esa_val] = cls_idx
        return out

    def _load_source_frames(self, patch_id: str, source_name: str) -> tuple[np.ndarray, list[float]] | None:
        """加载单个源的所有帧数据，返回 (data, timestamps_ms)."""
        if source_name == "planet":
            src_dir = self.planet_root / patch_id
        else:
            src_dir = self.data_root / patch_id / source_name

        if not src_dir.exists():
            return None

        tif_files = sorted(src_dir.glob("*.tif"))
        if len(tif_files) == 0:
            return None

        frames = []
        timestamps_ms = []
        for tif_path in tif_files:
            # 时间筛选
            date_str = tif_path.stem
            if self.start_date and date_str < self.start_date:
                continue
            if self.end_date and date_str > self.end_date:
                continue

            if len(frames) >= self.max_frames:
                break

            # 根据源分辨率差异选择重采样策略
            if source_name == "planet":
                # Planet 3m 高分辨率大图，需下采样以保持完整地理覆盖
                data = read_tif(tif_path, self.image_size, resize_mode="area")
            elif source_name == "landsat":
                # Landsat 30m 低分辨率小图，需真正上采样而非 edge pad
                data = read_tif(tif_path, self.image_size, resize_mode="bilinear")
            elif source_name in ("worldcover", "dynamic_world"):
                # 分类目标：最近邻插值保持类别值
                data = read_tif(tif_path, self.image_size, resize_mode="nearest")
            else:
                # S1, S2, tianyi_sar 等接近目标分辨率：center crop
                data = read_tif(tif_path, self.image_size)
            if data is None:
                continue
            # 过滤包含 NaN/Inf 的帧
            if np.isnan(data).any() or np.isinf(data).any():
                continue

            if source_name in ("worldcover", "dynamic_world"):
                # 分类目标：不做 mean/std 归一化，保持类别索引
                data = data.astype(np.int64)
                if source_name == "worldcover":
                    # ESA WorldCover 编码 -> 0-based 映射
                    data = self._remap_worldcover(data)
                elif source_name == "dynamic_world":
                    # Dynamic World 有效值 0-8；>=9 设为 ignore_index
                    data = np.where(data >= 9, 255, data)
            else:
                stats = self.stats.get(source_name, {})
                data = normalize_source(data, source_name, stats)
                # 归一化后再次检查 NaN
                if np.isnan(data).any() or np.isinf(data).any():
                    continue
            frames.append(data)
            # 从文件名解析日期并转为 ms
            ts_ms = parse_date_to_ms(date_str)
            timestamps_ms.append(ts_ms)

        if len(frames) == 0:
            return None

        # Stack: (T, C, H, W) -> (T, H, W, C)
        data = np.stack(frames, axis=0)
        data = np.transpose(data, (0, 2, 3, 1))  # (T, H, W, C)
        return data, timestamps_ms

    def _load_aef_embedding(self, patch_id: str) -> torch.Tensor | None:
        """加载预计算的 AEF 官方 64D embedding."""
        emb_path = self.aef_embedding_root / f"{patch_id}.npy"
        if not emb_path.exists():
            return None
        emb = np.load(emb_path)  # (64, 128, 128)
        return torch.from_numpy(emb).float()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        patch_id = sample["patch_id"]

        source_data: dict[str, torch.Tensor] = {}
        timestamps: dict[str, torch.Tensor] = {}

        all_timestamps: list[float] = []
        for source_name in self.source_names:
            result = self._load_source_frames(patch_id, source_name)
            if result is not None:
                data, ts_ms = result
                if source_name in ("worldcover", "dynamic_world"):
                    source_data[source_name] = torch.from_numpy(data).long()
                else:
                    source_data[source_name] = torch.from_numpy(data).float()
                timestamps[source_name] = torch.tensor(ts_ms, dtype=torch.float32)
                all_timestamps.extend(ts_ms)

        if len(all_timestamps) == 0:
            raise ValueError(f"No data loaded for patch {patch_id}")

        # Valid period: 数据的最早到最晚时间
        valid_start_ms = min(all_timestamps)
        valid_end_ms = max(all_timestamps)

        # 加载官方 AEF embedding
        aef_embedding = self._load_aef_embedding(patch_id)

        # 训练时随机垂直翻转，破坏数据固有的南北梯度
        if self.split == "train" and np.random.rand() < 0.5:
            for src in source_data:
                data = source_data[src].numpy()
                data = np.flip(data, axis=1)  # flip H dimension
                if source_data[src].dtype == torch.long:
                    source_data[src] = torch.from_numpy(data.copy()).long()
                else:
                    source_data[src] = torch.from_numpy(data.copy()).float()
            if aef_embedding is not None:
                aef_embedding = torch.flip(aef_embedding, dims=[1])  # (64, H, W) -> flip H

        return {
            "source_data": source_data,
            "timestamps": timestamps,
            "valid_period": (valid_start_ms, valid_end_ms),
            "patch_id": patch_id,
            "aef_embedding": aef_embedding,  # (64, 128, 128) 或 None
        }


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """合并 batch，处理变长序列，输出 AEF 格式."""
    batch_size = len(batch)

    # 获取所有源名称
    all_sources = set()
    for b in batch:
        all_sources.update(b["source_data"].keys())

    # 先计算全局最大时间步数（所有源之间统一）
    global_max_t = 0
    for b in batch:
        for src, tensor in b["source_data"].items():
            global_max_t = max(global_max_t, tensor.shape[0])

    collated_sources: dict[str, torch.Tensor] = {}
    collated_timestamps: dict[str, torch.Tensor] = {}

    for source in all_sources:
        # 收集该源的所有样本
        tensors = []
        ts_list = []
        for b in batch:
            if source in b["source_data"]:
                tensors.append(b["source_data"][source])
                ts_list.append(b["timestamps"][source])
            else:
                tensors.append(None)
                ts_list.append(None)

        # 过滤 None
        valid_tensors = [t for t in tensors if t is not None]
        valid_ts = [t for t in ts_list if t is not None]

        if len(valid_tensors) == 0:
            continue

        max_t = global_max_t  # 使用全局最大时间步
        _, H, W, C = valid_tensors[0].shape

        # 判断是否为分类目标（long dtype）
        is_categorical = valid_tensors[0].dtype == torch.long
        pad_value = 255 if is_categorical else 0
        pad_dtype = valid_tensors[0].dtype

        padded_tensors = []
        padded_ts = []
        for t, ts in zip(tensors, ts_list):
            if t is None:
                # 填充张量：分类目标用 255(ignore_index)，连续目标用 0
                padded_tensors.append(torch.full((max_t, H, W, C), pad_value, dtype=pad_dtype))
                last_valid_ts = valid_ts[-1][-1].item() if valid_ts else 0.0
                padded_ts.append(torch.full((max_t,), last_valid_ts, dtype=torch.float32))
            else:
                if t.shape[0] < max_t:
                    pad_t = max_t - t.shape[0]
                    t_pad = torch.cat([t, torch.full((pad_t, H, W, C), pad_value, dtype=t.dtype)], dim=0)
                    ts_pad = torch.cat([ts, ts[-1:].repeat(pad_t)], dim=0) if ts.numel() > 0 else torch.zeros(max_t, dtype=torch.float32)
                    padded_tensors.append(t_pad)
                    padded_ts.append(ts_pad)
                else:
                    padded_tensors.append(t)
                    padded_ts.append(ts)

        collated_sources[source] = torch.stack(padded_tensors)
        collated_timestamps[source] = torch.stack(padded_ts)

    valid_periods = torch.tensor([b["valid_period"] for b in batch], dtype=torch.float32)
    patch_ids = [b["patch_id"] for b in batch]

    # Collate AEF embeddings
    aef_embeddings = []
    has_aef = False
    for b in batch:
        emb = b.get("aef_embedding")
        if emb is not None:
            aef_embeddings.append(emb)
            has_aef = True
        else:
            aef_embeddings.append(None)

    result: dict[str, Any] = {
        "source_data": collated_sources,
        "timestamps": collated_timestamps,
        "valid_periods": valid_periods,
        "patch_ids": patch_ids,
    }

    if has_aef:
        # Stack valid embeddings, zero-pad missing ones
        valid_embs = [e for e in aef_embeddings if e is not None]
        if valid_embs:
            C, H, W = valid_embs[0].shape
            stacked = []
            for e in aef_embeddings:
                if e is None:
                    stacked.append(torch.zeros(C, H, W, dtype=torch.float32))
                else:
                    stacked.append(e)
            result["aef_embedding"] = torch.stack(stacked)  # (B, 64, 128, 128)
            result["aef_embedding_valid"] = torch.tensor([e is not None for e in aef_embeddings], dtype=torch.bool)

    return result
