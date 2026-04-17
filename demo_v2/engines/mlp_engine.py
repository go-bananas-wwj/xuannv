"""MLP 下游变化检测结果引擎."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image

from demo_v2.utils.constants import MLP_OUTPUT_BASE


class MLPEngine:
    """扫描并读取 MLP 下游输出的概率图、SHP 叠加图和指标."""

    # 模型版本 -> 输出根目录 映射
    VERSION_ROOTS = {
        "v2": MLP_OUTPUT_BASE / "aef_qwen_v2" / "shp_maps_mlp_monthly",
        "v2_hr_finetune": MLP_OUTPUT_BASE / "aef_qwen_v2_hr_finetune" / "shp_maps_mlp_monthly",
        "v2_hr_from_scratch": MLP_OUTPUT_BASE / "aef_qwen_v2_hr_from_scratch" / "shp_maps_mlp_monthly",
    }

    # 模型版本 -> metrics json 路径
    VERSION_METRICS = {
        "v2": MLP_OUTPUT_BASE / "aef_qwen_v2" / "mlp_downstream" / "mlp_training_summary.json",
        "v2_hr_finetune": MLP_OUTPUT_BASE / "aef_qwen_v2_hr_finetune" / "mlp_downstream" / "mlp_training_summary.json",
        "v2_hr_from_scratch": MLP_OUTPUT_BASE / "aef_qwen_v2_hr_from_scratch" / "mlp_downstream" / "mlp_training_summary.json",
    }

    def __init__(self):
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}

    def list_available_versions(self) -> List[str]:
        """返回有实际输出数据的版本列表."""
        return [v for v, p in self.VERSION_ROOTS.items() if p.exists()]

    def list_categories(self, version: str) -> List[str]:
        """返回某版本下所有可用的变化类别目录名."""
        root = self.VERSION_ROOTS.get(version)
        if root is None or not root.exists():
            return []
        return sorted([d.name for d in root.iterdir() if d.is_dir()])

    def list_time_windows(self, version: str, category: str) -> List[str]:
        """返回某类别下的所有时间窗口子目录."""
        cat_dir = self.VERSION_ROOTS.get(version, Path()) / category
        if not cat_dir.exists():
            return []
        return sorted([d.name for d in cat_dir.iterdir() if d.is_dir()])

    def list_pages(self, version: str, category: str, time_window: str) -> List[Path]:
        """返回某时间窗口下所有结果页面图片路径（按 page_01, page_02 排序）."""
        tw_dir = self.VERSION_ROOTS.get(version, Path()) / category / time_window
        if not tw_dir.exists():
            return []
        return sorted(tw_dir.glob("page_*.png"))

    def get_metrics(self, version: str) -> Dict[str, Any]:
        """读取某版本的 MLP 训练汇总指标."""
        if version in self._metrics_cache:
            return self._metrics_cache[version]
        path = self.VERSION_METRICS.get(version)
        if path is None or not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._metrics_cache[version] = data
            return data
        except Exception:
            return {}

    def get_category_metrics(self, version: str, category: str) -> Dict[str, Any]:
        """获取指定类别的汇总指标."""
        metrics = self.get_metrics(version)
        # 类别名称可能需要映射：SAR建筑工地 -> 建筑工地 等
        # 但 JSON key 通常与目录名一致或接近
        if category in metrics:
            return metrics[category]
        # 尝试去掉 SAR 前缀
        alt = category.replace("SAR", "")
        if alt in metrics:
            return metrics[alt]
        return {}

    def load_page_image(self, version: str, category: str, time_window: str, page_index: int) -> Image.Image | None:
        """加载指定页码的图片（1-based）."""
        pages = self.list_pages(version, category, time_window)
        if not pages or page_index < 1 or page_index > len(pages):
            return None
        return Image.open(pages[page_index - 1])
