"""多区域混合数据集 — 支持哈尔滨 + 海淀.

核心修改:
1. 从manifest加载多区域数据
2. 区域感知的源目录解析
3. 区域感知的统计量/AEF/OlmoEarth加载
4. 缺失静态目标自动跳过
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.dataset import HarbinPatchDataset


class MultiRegionPatchDataset(HarbinPatchDataset):
    """多区域Patch数据集 — 支持跨区域混合训练."""

    def __init__(self, cfg) -> None:
        # 先加载manifest
        self.region_manifest = self._load_manifest(cfg)
        
        # 临时修改cfg的manifest_path为第一个区域路径（用于父类初始化）
        original_manifest = cfg.data.manifest_path
        if self.region_manifest:
            first_region = next(iter(self.region_manifest))
            first_root = self.region_manifest[first_region]['data_root']
            cfg.data.manifest_path = first_root
        
        # 调用父类初始化（大部分逻辑保持不变）
        super().__init__(cfg)
        
        # 恢复原始配置
        cfg.data.manifest_path = original_manifest
        
        # 覆盖patches为所有区域的合并列表
        # 格式: "{region}_{local_patch_id}"
        self.all_patch_ids = []
        for region, info in self.region_manifest.items():
            for patch_id in info['patches']:
                self.all_patch_ids.append(f"{region}_{patch_id}")
        
        # 更新父类的patches列表
        self.patches = self.all_patch_ids
        self.num_samples = len(self.patches)
        
        # ★ 重建月度样本索引（父类初始化时只用了第一个区域）
        self.monthly_samples = self._build_monthly_samples()
        print(f"[Dataset] 月度样本数: {len(self.monthly_samples)} (来自 {len(self.patches)} patches)")
        
        # ★ 重新预加载 AEF 嵌入（父类初始化时 self.patches 只有第一个区域）
        self._preload_aef_embeddings()
    
    def _load_manifest(self, cfg) -> dict:
        """加载多区域manifest."""
        manifest_path = getattr(cfg.data, 'multi_region_manifest', None)
        if not manifest_path:
            return {}
        with open(manifest_path) as f:
            return json.load(f)
    
    def _resolve_source_dir(self, source_name: str, patch_id: str) -> Path | None:
        """区域感知的源目录解析.
        
        patch_id格式: "{region}_{local_id}" 或单区域格式。
        """
        # 检查是否是多区域格式
        if isinstance(patch_id, str) and '_' in patch_id:
            parts = patch_id.split('_', 1)
            # 处理 region 为多个单词的情况（如 harbin_newarea）
            for region_name in self.region_manifest:
                prefix = f"{region_name}_"
                if patch_id.startswith(prefix):
                    local_id = patch_id[len(prefix):]
                    return self._resolve_region_source_dir(region_name, source_name, local_id)
        
        # 回退到父类行为（单区域）
        return super()._resolve_source_dir(source_name, patch_id)
    
    def _resolve_region_source_dir(self, region: str, source_name: str, patch_id: str) -> Path | None:
        """解析特定区域的源目录."""
        info = self.region_manifest[region]
        region_root = Path(info['data_root'])

        # source_roots 覆盖（per-source 根目录）
        source_roots = info.get('source_roots', {})
        if source_name in source_roots:
            override_root = Path(source_roots[source_name])
            src_dir = override_root / source_name / patch_id
            if src_dir.exists():
                return src_dir

        # 标准路径: data_root / source_name / patch_id
        src_dir = region_root / source_name / patch_id
        if src_dir.exists():
            return src_dir
        
        # 回退：在 data_root 的直接子目录中搜索
        for sub_dir in region_root.iterdir():
            if not sub_dir.is_dir():
                continue
            candidate = sub_dir / source_name / patch_id
            if candidate.exists():
                return candidate
        
        return None
    
    def _preload_teacher_tokens(self) -> None:
        """多区域 OlmoEarth teacher tokens 预加载."""
        for region, info in self.region_manifest.items():
            tokens_root = info.get('olmoearth_tokens_root')
            if tokens_root is None:
                continue
            root = Path(tokens_root)
            if not root.exists():
                continue
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
                        # 合并到已有月份（如果不同区域有相同月份）
                        if key not in self._teacher_tokens:
                            self._teacher_tokens[key] = zd["tokens"].astype(np.float16)
                            self._teacher_tok_pid2row[key] = {}
                        else:
                            # 追加到已有数组
                            existing = self._teacher_tokens[key]
                            new_tokens = zd["tokens"].astype(np.float16)
                            self._teacher_tokens[key] = np.concatenate([existing, new_tokens], axis=0)
                        
                        if "patch_ids" in zd.files:
                            offset = 0 if key not in self._teacher_tok_pid2row or not self._teacher_tok_pid2row[key] else max(self._teacher_tok_pid2row[key].values()) + 1
                            for i, p in enumerate(zd["patch_ids"]):
                                prefixed = f"{region}_{str(p)}"
                                self._teacher_tok_pid2row[key][prefixed] = offset + i
                    except Exception:
                        continue
                emb_path = d / "emb_all.npz"
                if emb_path.exists():
                    try:
                        ed = np.load(str(emb_path))
                        if "embeddings" in ed.files:
                            if key not in self._teacher_global:
                                self._teacher_global[key] = ed["embeddings"].astype(np.float32)
                                self._teacher_glb_pid2row[key] = {}
                            else:
                                existing = self._teacher_global[key]
                                new_emb = ed["embeddings"].astype(np.float32)
                                self._teacher_global[key] = np.concatenate([existing, new_emb], axis=0)
                            
                            if "patch_ids" in ed.files:
                                offset = 0 if key not in self._teacher_glb_pid2row or not self._teacher_glb_pid2row[key] else max(self._teacher_glb_pid2row[key].values()) + 1
                                for i, p in enumerate(ed["patch_ids"]):
                                    prefixed = f"{region}_{str(p)}"
                                    self._teacher_glb_pid2row[key][prefixed] = offset + i
                    except Exception:
                        pass
        self._teacher_months = sorted(self._teacher_tokens.keys())
        if self._teacher_months:
            total_patches = sum(a.shape[0] for a in self._teacher_tokens.values())
            gb = sum(a.nbytes for a in self._teacher_tokens.values()) / 1e9
            print(f"[Dataset] OlmoEarth teacher tokens 预加载: {len(self._teacher_months)} 个月, "
                  f"{total_patches} patches, {gb:.2f}GB (fp16, 内存常驻)")
    
    def _preload_aef_embeddings(self) -> None:
        """多区域 AEF 嵌入预加载."""
        self._aef_embeds: dict[str, np.ndarray] = {}
        loaded = 0
        for region, info in self.region_manifest.items():
            aef_dir = info.get('aef_embed_dir')
            if aef_dir is None:
                continue
            root = Path(aef_dir)
            if not root.exists():
                continue
            for patch_id in self.patches:
                if not patch_id.startswith(f"{region}_"):
                    continue
                local_pid = patch_id[len(f"{region}_"):]
                fpath = root / f"{local_pid}.npy"
                if fpath.exists():
                    try:
                        self._aef_embeds[patch_id] = np.load(fpath).astype(np.float32)
                        loaded += 1
                    except Exception:
                        pass
        if loaded > 0:
            print(f"[Dataset] AEF 嵌入预加载: {loaded}/{len(self.patches)} patches")
    
    def _load_stats(self, stats_dir: Path | None) -> dict:
        """多区域统计量加载 — 合并所有区域的 stats."""
        stats: dict[str, dict[str, dict[str, float]]] = {}
        for region, info in self.region_manifest.items():
            region_stats_dir = info.get('stats_dir')
            if region_stats_dir is None:
                continue
            region_stats_path = Path(region_stats_dir)
            if not region_stats_path.exists():
                continue
            for stats_file in region_stats_path.glob("*_stats.json"):
                source_name = stats_file.stem.replace("_stats", "")
                with stats_file.open("r") as f:
                    region_source_stats = json.load(f)
                if source_name not in stats:
                    stats[source_name] = region_source_stats
                else:
                    # 简单合并：如果已有，保留第一个区域的统计量
                    pass
        return stats
    
    def _load_target_frame(self, patch_id: str, source_name: str):
        """加载目标帧，处理跨区域缺失的静态目标."""
        # 解析region
        region = None
        if isinstance(patch_id, str) and '_' in patch_id:
            for region_name in self.region_manifest:
                if patch_id.startswith(f"{region_name}_"):
                    region = region_name
                    break
        
        if region is None:
            return super()._load_target_frame(patch_id, source_name)
        
        # 检查该区域是否有此目标源
        info = self.region_manifest.get(region, {})
        available_sources = info.get('sources', [])
        
        if source_name not in available_sources:
            # 目标源不可用，返回空张量 + False标记
            out_ch = self._get_target_channels(source_name)
            empty = np.zeros((out_ch, self.image_size, self.image_size), dtype=np.float32)
            return empty, False
        
        # 目标源可用，正常加载
        return super()._load_target_frame(patch_id, source_name)
    
    def _get_target_channels(self, source_name: str) -> int:
        """获取目标源的输出通道数."""
        channels_map = {
            's2': 6,
            's1': 2,
            'landsat': 6,
            'dem': 1,
            'worldcover': 1,
            'dynamic_world': 1,
            'jrc_water': 1,
            'sar': 1,
        }
        return channels_map.get(source_name, 1)
