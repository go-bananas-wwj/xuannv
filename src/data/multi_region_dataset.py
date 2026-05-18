"""多区域混合数据集 — 支持哈尔滨 + 大庆 + 海淀.

核心修改:
1. 从manifest加载多区域数据
2. 区域感知的源目录解析（S2云筛选路径）
3. 区域感知的统计量加载
4. 缺失静态目标（phase2无DW/JRC）自动跳过
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
        
        # 临时修改cfg的manifest_path为哈尔滨路径（用于父类初始化）
        original_manifest = cfg.data.manifest_path
        if self.region_manifest:
            harbin_root = self.region_manifest['harbin']['data_root']
            cfg.data.manifest_path = harbin_root
        
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
            if len(parts) == 2 and parts[0] in self.region_manifest:
                region, local_id = parts
                return self._resolve_region_source_dir(region, source_name, local_id)
        
        # 回退到父类行为（单区域）
        return super()._resolve_source_dir(source_name, patch_id)
    
    def _resolve_region_source_dir(self, region: str, source_name: str, patch_id: str) -> Path | None:
        """解析特定区域的源目录."""
        info = self.region_manifest[region]
        region_root = Path(info['data_root'])
        
        # S2特殊处理：优先使用云筛选后的目录
        if source_name == 's2':
            cloud_dir = region_root / 's2_cloud_filtered' / patch_id
            if cloud_dir.exists():
                return cloud_dir
            # fallback到原始s2
            orig_dir = region_root / 's2' / patch_id
            if orig_dir.exists():
                return orig_dir
            return None
        
        # 其他源
        src_dir = region_root / source_name / patch_id
        if src_dir.exists():
            return src_dir
        return None
    
    def _load_target_frame(self, patch_id: str, source_name: str):
        """加载目标帧，处理跨区域缺失的静态目标.
        
        phase2数据没有DynamicWorld和JRCWater，对这些patch返回空结果+False标记。
        """
        # 解析region
        region = 'harbin'
        if isinstance(patch_id, str) and '_' in patch_id:
            parts = patch_id.split('_', 1)
            if len(parts) == 2 and parts[0] in self.region_manifest:
                region = parts[0]
        
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
        }
        return channels_map.get(source_name, 1)
