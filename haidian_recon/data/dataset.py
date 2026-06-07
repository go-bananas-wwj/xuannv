"""数据集 — 时间聚合 + 四层Mask."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from haidian_recon.data.transforms import read_tif, normalize_source


class HaidianReconDataset(Dataset):
    """
    海淀区多源重建数据集。
    """

    def __init__(
        self,
        data_root: str = "data_raw/haidian/scenes",
        planet_root: str = "data_raw/beijing/planetscene",
        stats_dir: str = "statistics/haidian",
        cache_dir: str = "haidian_recon/.cache",
        image_size: int = 128,
        source_names: list[str] | None = None,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        aef_embedding_root: str | None = "data_raw/haidian/aef_embeddings/haidian_2025_patches",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.planet_root = Path(planet_root)
        self.image_size = image_size
        self.source_names = source_names or ["tianyi_sar", "s2", "landsat", "planet"]
        self.cfg_anchor_source = "tianyi_sar"  # 锚点源
        self.split = split
        self.aef_embedding_root = Path(aef_embedding_root) if aef_embedding_root else None

        # 加载时间映射表（不存在时自动生成）
        mapping_path = Path(cache_dir) / "temporal_mapping.json"
        if not mapping_path.exists():
            print(f"[Dataset] Temporal mapping not found, building...")
            from haidian_recon.scripts.build_temporal_mapping import build_mapping
            build_mapping(
                data_root=str(self.data_root),
                planet_root=str(self.planet_root),
                output_path=str(mapping_path),
            )
        with open(mapping_path) as f:
            self.mapping = json.load(f)

        self.patch_ids = sorted(self.mapping.keys())

        # 划分train/val
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

        # 构建样本列表
        self.samples = []
        for patch_id in self.patch_ids:
            for entry in self.mapping[patch_id]:
                self.samples.append({
                    "patch_id": patch_id,
                    "anchor_date": entry["anchor_date"],
                    "sources": entry["sources"],
                })

        print(f"[{split}] {len(self.patch_ids)} patches, {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_source(self, patch_id: str, source_name: str, file_date: str | None) -> np.ndarray | None:
        """加载单个源的单帧数据."""
        if file_date is None:
            return None

        if source_name == "planet":
            path = self.planet_root / patch_id / f"{file_date}.tif"
        else:
            path = self.data_root / patch_id / source_name / f"{file_date}.tif"

        if not path.exists():
            return None

        data = read_tif(path, self.image_size)
        if data is None:
            return None

        # 归一化
        stats = self.stats.get(source_name, {})
        data = normalize_source(data, source_name, stats)
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | None]:
        sample = self.samples[idx]
        patch_id = sample["patch_id"]
        sources_files = sample["sources"]

        batch = {}
        for source_name in self.source_names:
            # 锚点源（tianyi_sar）的file_date就是anchor_date
            if source_name == self.cfg_anchor_source:
                file_date = sample["anchor_date"]
            else:
                file_date = sources_files.get(source_name)
            data = self._load_source(patch_id, source_name, file_date)
            if data is not None:
                # [1, 1, C, H, W] — batch维度=1, 时间维度=1
                batch[source_name] = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
            else:
                batch[source_name] = None

        # 加载预计算的 AEF embedding (64, 128, 128) -> GAP -> 64D
        if self.aef_embedding_root is not None:
            aef_path = self.aef_embedding_root / f"{patch_id}.npy"
            if aef_path.exists():
                aef_emb = np.load(aef_path)  # (64, 128, 128)
                # 全局平均池化 -> (64,)
                aef_emb = aef_emb.mean(axis=(1, 2))  # (64,)
                batch["aef_embedding"] = torch.from_numpy(aef_emb).float()
            else:
                batch["aef_embedding"] = None
        else:
            batch["aef_embedding"] = None

        # 记录 patch_id 供 trainer 使用
        batch["patch_id"] = patch_id

        return batch


def collate_fn(batch_list: list[dict]) -> dict[str, torch.Tensor | None]:
    """
    合并batch。缺失的源填充零张量，并记录valid_mask供masking模块使用。
    """
    keys = batch_list[0].keys()
    batch_size = len(batch_list)
    result = {}

    for key in keys:
        values = [b[key] for b in batch_list]
        if all(v is None for v in values):
            result[key] = None
            continue

        # 特殊处理 aef_embedding: 一维向量 (64,)
        if key == "aef_embedding":
            dim = next(v for v in values if v is not None).shape[0]
            stacked = torch.zeros(batch_size, dim, dtype=torch.float32)
            valid_mask = torch.zeros(batch_size, dtype=torch.bool)
            for i, v in enumerate(values):
                if v is not None:
                    stacked[i] = v
                    valid_mask[i] = True
            result[key] = stacked
            result[f"{key}_valid"] = valid_mask
            continue

        # 特殊处理 patch_id: 字符串列表
        if key == "patch_id":
            result[key] = [v for v in values]
            continue

        # 获取第一个非None的shape作为参考 [1, 1, C, H, W]
        first = next(v for v in values if v is not None)
        _, _, C, H, W = first.shape

        # 填充到统一batch
        stacked = torch.zeros(batch_size, 1, C, H, W, dtype=first.dtype)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)
        for i, v in enumerate(values):
            if v is not None:
                stacked[i] = v  # v是[1,1,C,H,W]
                valid_mask[i] = True

        result[key] = stacked
        # 将valid_mask附加到result中供masking使用
        result[f"{key}_valid"] = valid_mask

    return result
