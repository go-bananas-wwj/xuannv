"""下载器抽象基类。"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDownloader(ABC):
    """
    所有下载器的基类。

    子类必须实现 download()，负责将某个数据源的所有 patch 下载到
    output_dir/{source_name}/{patch_id}/{YYYYMMDD}.tif 结构。
    """

    def __init__(self, region_cfg: dict, source_name: str) -> None:
        self.region_cfg = region_cfg
        self.source_name = source_name
        self.source_cfg = region_cfg["sources"][source_name]
        self.region_name = region_cfg["region_name"]
        self.output_dir = Path(region_cfg["output_dir"]) / source_name

    # ------------------------------------------------------------------ #
    # 抽象接口
    # ------------------------------------------------------------------ #

    @abstractmethod
    def download(self, patches: list[dict], *, workers: int = 4) -> dict[str, int]:
        """
        下载所有 patch 的数据到 output_dir。

        Args:
            patches : 由 patchify 生成的 patch 列表（含 utm_bounds/id）
            workers : 并发线程数

        Returns:
            dict {"downloaded": N, "skipped": N, "failed": N}
        """

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    def patch_out_dir(self, patch_id: int | str) -> Path:
        """返回某 patch 的输出目录，自动创建。"""
        d = self.output_dir / f"patch_{patch_id:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_download_report(self, stats: dict[str, Any]) -> None:
        """保存下载统计报告到 output_dir/download_report.json。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rpt = self.output_dir / "download_report.json"
        rpt.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    @staticmethod
    def is_complete(patch_dir: Path, min_files: int = 1) -> bool:
        """快速检查某 patch 目录是否已有足够的 tif 文件（用于断点续传）。"""
        if not patch_dir.exists():
            return False
        return len(list(patch_dir.glob("*.tif"))) >= min_files
