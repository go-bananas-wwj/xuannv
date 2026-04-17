"""AEF_qwen Gradio Demo — 数据缓存和工具函数."""
from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rasterio

# ──────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────
RAW_DIR = Path("/workspace/raw/harbin_scenes")
STATS_DIR = Path("/workspace/statistics/harbin_scenes")
GRID_GEOJSON = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
OUTPUT_DIR = Path("/workspace/outputs/aef_qwen_v1")

# 模型注册
MODEL_REGISTRY = {
    "qwen_v1 (3-input, 7-target)": {
        "output_dir": str(OUTPUT_DIR),
        "checkpoint": str(OUTPUT_DIR / "epoch_399.pt"),
        "config": "/workspace/xuannv/configs/qwen_v1_scenes.yaml",
        "embedding_dim": 128,
        "status": "ready",
        "description": "3类输入(S2+S1+Landsat), 7类目标, skip_l2训练, raw_uniformity反坍缩",
    },
}

# 输入源 (按时间特性排序)
INPUT_SOURCES = ["s2", "s1", "landsat"]

# 所有可用源
ALL_SOURCES = ["s2", "s1", "landsat", "dem", "worldcover", "dynamic_world", "jrc_water"]

SOURCE_DISPLAY_NAMES = {
    "s2": "Sentinel-2 (光学)",
    "s1": "Sentinel-1 (SAR)",
    "landsat": "Landsat 8/9",
    "dem": "DEM 高程",
    "worldcover": "WorldCover 土地覆盖",
    "dynamic_world": "Dynamic World",
    "jrc_water": "JRC 水体",
}

# WorldCover 颜色映射
WC_COLORS = {
    0: (65, 155, 223),   # 水体
    1: (57, 125, 73),    # 树林
    2: (136, 176, 83),   # 灌木
    3: (255, 187, 34),   # 草地
    4: (255, 255, 76),   # 作物
    5: (187, 85, 29),    # 建筑
    6: (222, 222, 222),  # 裸地
    7: (170, 170, 170),  # 冰雪
    8: (120, 80, 20),    # 湿地
    9: (140, 140, 140),  # 苔原
    10: (100, 100, 100), # 红树林
}


@dataclass
class PatchMeta:
    """单个 patch 的元数据."""
    patch_id: str
    bounds: dict[str, float] = field(default_factory=dict)
    crs: str = "EPSG:32652"
    sources: dict[str, list[str]] = field(default_factory=dict)
    ix: int = 0
    iy: int = 0


class DataCache:
    """全局数据缓存."""

    def __init__(self):
        self.patch_metas: list[PatchMeta] = []
        self.embedding_maps: dict[str, np.ndarray] = {}  # model_name -> [N, D, H, W]
        self.embedding_map_ids: dict[str, list[str]] = {}  # model_name -> [patch_id, ...]
        self.grid_data: dict | None = None
        self._loaded = False

    def load(self):
        """加载 patch 元数据和 grid."""
        if self._loaded:
            return

        # 加载 grid GeoJSON
        if GRID_GEOJSON.exists():
            with open(GRID_GEOJSON) as f:
                self.grid_data = json.load(f)
            for feat in self.grid_data.get("features", []):
                p = feat["properties"]
                pid = p.get("patch_id", "")
                bounds = {}
                if "geometry" in feat:
                    coords = feat["geometry"]["coordinates"][0]
                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]
                    bounds = {"minx": min(xs), "maxx": max(xs), "miny": min(ys), "maxy": max(ys)}
                pm = PatchMeta(
                    patch_id=pid,
                    bounds=bounds,
                    ix=p.get("ix", 0),
                    iy=p.get("iy", 0),
                )
                self.patch_metas.append(pm)

        # 扫描数据源
        if not self.patch_metas:
            # 从目录扫描
            s2_dir = RAW_DIR / "s2"
            if s2_dir.exists():
                for patch_dir in sorted(s2_dir.iterdir()):
                    if patch_dir.is_dir() and patch_dir.name.startswith("patch_"):
                        pm = PatchMeta(patch_id=patch_dir.name)
                        self.patch_metas.append(pm)

        # 统计每个 patch 的数据源
        for pm in self.patch_metas:
            sources = {}
            for src in ALL_SOURCES:
                src_dir = RAW_DIR / src / pm.patch_id
                if src_dir.exists():
                    tifs = sorted([f.stem for f in src_dir.glob("*.tif")])
                    if tifs:
                        sources[src] = tifs
            pm.sources = sources

        # 加载 embedding maps
        self._load_embeddings()

        self._loaded = True

    def _load_embeddings(self):
        """加载预计算的 embedding maps."""
        for model_name, reg in MODEL_REGISTRY.items():
            out_dir = Path(reg["output_dir"])
            emb_path = out_dir / "embeddings" / "embedding_maps.npy"
            ids_path = out_dir / "embeddings" / "patch_ids.json"

            if emb_path.exists() and ids_path.exists():
                self.embedding_maps[model_name] = np.load(emb_path)
                with open(ids_path) as f:
                    self.embedding_map_ids[model_name] = json.load(f)

    def get_patch_index(self, patch_id: str) -> int:
        """根据 patch_id 找索引."""
        for i, pm in enumerate(self.patch_metas):
            if pm.patch_id == patch_id:
                return i
        return -1

    def load_s2_rgb(self, patch_id: str, frame_idx: int = -1) -> np.ndarray | None:
        """加载 S2 RGB 图像."""
        src_dir = RAW_DIR / "s2" / patch_id
        if not src_dir.exists():
            return None
        tifs = sorted(src_dir.glob("*.tif"))
        if not tifs:
            return None
        idx = frame_idx if frame_idx >= 0 else len(tifs) - 1
        idx = min(idx, len(tifs) - 1)
        try:
            with rasterio.open(str(tifs[idx])) as src:
                data = src.read().astype(np.float32)
            # 假设波段顺序: B2, B3, B4, B8, B11, ... → 取 B4, B3, B2 作为 RGB
            if data.shape[0] >= 3:
                rgb = data[[2, 1, 0], :, :]  # B4, B3, B2
            else:
                rgb = data[:3, :, :]
            # 归一化到 0-255
            for c in range(min(3, rgb.shape[0])):
                ch = rgb[c]
                p2, p98 = np.percentile(ch, (2, 98))
                if p98 > p2:
                    rgb[c] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)
                else:
                    rgb[c] = 0
            return rgb.astype(np.uint8)
        except Exception:
            return None

    def load_source_thumbnail(self, patch_id: str, source: str, frame_idx: int = -1,
                               target_size: int = 128) -> np.ndarray | None:
        """加载任意源缩略图."""
        src_dir = RAW_DIR / source / patch_id
        if not src_dir.exists():
            return None
        tifs = sorted(src_dir.glob("*.tif"))
        if not tifs:
            return None
        idx = frame_idx if frame_idx >= 0 else len(tifs) - 1
        idx = min(idx, len(tifs) - 1)
        try:
            with rasterio.open(str(tifs[idx])) as src:
                data = src.read().astype(np.float32)
            if data.shape[0] >= 3:
                return data[[2, 1, 0], :, :]
            elif data.shape[0] == 1:
                # 单通道 → 复制3次
                ch = np.clip((data[0] - data[0].min()) / max(data[0].max() - data[0].min(), 1e-8) * 255, 0, 255)
                return np.stack([ch, ch, ch], axis=0).astype(np.uint8)
            else:
                return data[:3, :, :]
        except Exception:
            return None

    def colorize_worldcover(self, patch_id: str) -> np.ndarray | None:
        """渲染 WorldCover 彩色地图."""
        src_dir = RAW_DIR / "worldcover" / patch_id
        if not src_dir.exists():
            return None
        tifs = sorted(src_dir.glob("*.tif"))
        if not tifs:
            return None
        try:
            with rasterio.open(str(tifs[0])) as src:
                data = src.read(1).astype(np.int32)
            # 映射到颜色
            h, w = data.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for orig_val, color in WC_COLORS.items():
                mask = data == orig_val
                rgb[mask] = color
            return rgb
        except Exception:
            return None

    def compute_s2_ndvi(self, patch_id: str, frame_idx: int = -1) -> np.ndarray | None:
        """计算 S2 NDVI."""
        src_dir = RAW_DIR / "s2" / patch_id
        if not src_dir.exists():
            return None
        tifs = sorted(src_dir.glob("*.tif"))
        if not tifs:
            return None
        idx = min(frame_idx if frame_idx >= 0 else len(tifs) - 1, len(tifs) - 1)
        try:
            with rasterio.open(str(tifs[idx])) as src:
                data = src.read().astype(np.float32)
            # B4 (red), B8 (NIR) — 假设索引 0,3
            if data.shape[0] >= 4:
                red = data[0]
                nir = data[3]
                ndvi = (nir - red) / (nir + red + 1e-8)
                # 归一化到 0-255 用于显示
                ndvi_vis = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                # 绿色调色板
                rgb = np.zeros((ndvi_vis.shape[0], ndvi_vis.shape[1], 3), dtype=np.uint8)
                rgb[:, :, 1] = ndvi_vis  # 绿色通道
                return rgb
        except Exception:
            pass
        return None


# 全局单例
cache = DataCache()
