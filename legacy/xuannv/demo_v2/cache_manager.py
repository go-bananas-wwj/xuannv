"""统一缓存管理器 — 预计算 embedding maps / S2 RGB / Grid 元数据."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import rasterio
from PIL import Image

# 将项目根目录加入 sys.path 以导入 src 模块
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/AEF")

from demo_v2.utils.constants import (
    RAW_DIR,
    GRID_PATH,
    MODEL_REGISTRY,
    SOURCE_DISPLAY_NAMES,
)


class PatchMeta:
    """单个 patch 的元数据."""

    def __init__(self, patch_id: str, bounds: Tuple[float, float, float, float], crs: str):
        self.patch_id = patch_id
        self.bounds = bounds  # (minx, miny, maxx, maxy)
        self.crs = crs
        self.sources: Dict[str, int] = {}  # source_name -> frame_count

    def __repr__(self) -> str:
        return f"PatchMeta({self.patch_id}, sources={list(self.sources.keys())})"


class CacheManager:
    """全局缓存管理器（单例模式由模块级变量保证）."""

    def __init__(self):
        self.patch_ids: List[str] = []
        self.patch_metas: List[PatchMeta] = []
        self._meta_by_id: Dict[str, PatchMeta] = {}
        self._grid_data: Optional[dict] = None

        # version -> embedding_maps [N, D, H, W]
        self.embedding_maps: Dict[str, np.ndarray] = {}
        # version -> [patch_id, ...]
        self.embedding_map_patch_ids: Dict[str, List[str]] = {}
        # version -> PCA cache
        self._pca_cache: Dict[str, Tuple[Any, np.ndarray, np.ndarray]] = {}

    # ──────────────────────────────────────────
    # 加载
    # ──────────────────────────────────────────
    def load(self) -> None:
        """启动时加载所有静态数据."""
        print("[CacheManager] Loading grid and patch metadata...")
        self._load_grid()
        self._scan_sources()
        self._load_embedding_maps()
        print(f"[CacheManager] Loaded {len(self.patch_ids)} patches.")
        print(f"[CacheManager] Embedding maps: {list(self.embedding_maps.keys())}")

    def _load_grid(self) -> None:
        with open(GRID_PATH) as f:
            self._grid_data = json.load(f)

        for feat in self._grid_data["features"]:
            pid = feat["properties"]["patch_id"]
            coords = feat["geometry"]["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            crs = feat["properties"].get("crs", "EPSG:32652")
            meta = PatchMeta(pid, (min(xs), min(ys), max(xs), max(ys)), crs)
            self.patch_metas.append(meta)
            self._meta_by_id[pid] = meta
        self.patch_ids = [m.patch_id for m in self.patch_metas]

    def _scan_sources(self) -> None:
        """扫描每个 patch 下各数据源的帧数."""
        for src_dir in sorted(RAW_DIR.iterdir()):
            if not src_dir.is_dir():
                continue
            for meta in self.patch_metas:
                pid = meta.patch_id
                patch_src_dir = src_dir / pid
                if patch_src_dir.is_dir():
                    n_files = len(list(patch_src_dir.glob("*.tif")))
                    if n_files > 0:
                        meta.sources[src_dir.name] = n_files

    def _load_embedding_maps(self) -> None:
        """加载已预计算的 embedding maps."""
        for ver, info in MODEL_REGISTRY.items():
            emb_dir = info["embeddings_dir"]
            emb_path = emb_dir / "embedding_maps.npy"
            ids_path = emb_dir / "patch_ids.json"
            if not emb_path.exists() or not ids_path.exists():
                continue
            try:
                maps = np.load(emb_path)
                with open(ids_path) as f:
                    ids = json.load(f)
                self.embedding_maps[ver] = maps
                self.embedding_map_patch_ids[ver] = ids
                print(f"[CacheManager] Loaded {ver} embeddings: {maps.shape}")
            except Exception as e:
                print(f"[CacheManager] Failed to load {ver} embeddings: {e}")

    # ──────────────────────────────────────────
    # 查询接口
    # ──────────────────────────────────────────
    def get_meta(self, patch_id: str) -> Optional[PatchMeta]:
        return self._meta_by_id.get(patch_id)

    def get_patch_dir(self, patch_id: str) -> Path:
        return RAW_DIR / patch_id

    def get_embedding_map(self, version: str, patch_id: str) -> Optional[np.ndarray]:
        """获取指定 version + patch_id 的 embedding map [D, H, W]."""
        maps = self.embedding_maps.get(version)
        ids = self.embedding_map_patch_ids.get(version)
        if maps is None or ids is None:
            return None
        try:
            idx = ids.index(patch_id)
        except ValueError:
            return None
        if idx >= maps.shape[0]:
            return None
        return maps[idx]

    def get_s2_rgb(self, patch_id: str, frame_idx: int = -1) -> Optional[np.ndarray]:
        """加载 S2 RGB [H, W, 3] uint8."""
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
            rgb = data[[2, 1, 0], :, :] if data.shape[0] >= 3 else data[:3]
            for c in range(min(3, rgb.shape[0])):
                p2, p98 = np.percentile(rgb[c], (2, 98))
                if p98 > p2:
                    rgb[c] = np.clip((rgb[c] - p2) / (p98 - p2) * 255, 0, 255)
            return np.transpose(rgb.astype(np.uint8), (1, 2, 0))
        except Exception:
            return None

    def get_worldcover(self, patch_id: str) -> Optional[np.ndarray]:
        """加载 WorldCover 标签 [H, W] int32."""
        src_dir = RAW_DIR / "worldcover" / patch_id
        if not src_dir.exists():
            return None
        tifs = sorted(src_dir.glob("*.tif"))
        if not tifs:
            return None
        try:
            with rasterio.open(str(tifs[0])) as src:
                return src.read(1).astype(np.int32)
        except Exception:
            return None

    # ──────────────────────────────────────────
    # PCA 缓存
    # ──────────────────────────────────────────
    @property
    def num_patches(self) -> int:
        return len(self.patch_ids)

    def get_global_pca(self, version: str) -> Tuple[Any, np.ndarray, np.ndarray]:
        """对所有 patch embedding map 拟合全局 PCA(3)."""
        if version in self._pca_cache:
            return self._pca_cache[version]

        from sklearn.decomposition import PCA

        maps = self.embedding_maps.get(version)
        if maps is None:
            raise ValueError(f"No embedding maps for {version}")

        N, D, H, W = maps.shape
        flat = maps.transpose(0, 2, 3, 1).reshape(-1, D)
        rng = np.random.RandomState(42)
        n_samples = min(200_000, flat.shape[0])
        idx = rng.choice(flat.shape[0], n_samples, replace=False)
        pca = PCA(n_components=3, random_state=42)
        pca.fit(flat[idx])

        projected = pca.transform(flat[idx])
        vmin = np.percentile(projected, 1, axis=0)
        vmax = np.percentile(projected, 99, axis=0)

        self._pca_cache[version] = (pca, vmin, vmax)
        return self._pca_cache[version]


# 全局单例
cache = CacheManager()
