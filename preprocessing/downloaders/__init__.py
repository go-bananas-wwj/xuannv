"""下载器工厂。"""
from __future__ import annotations

from preprocessing.downloaders.base import BaseDownloader


def get_downloader(region_cfg: dict, source_name: str) -> BaseDownloader:
    """
    根据区域配置中 sources[source_name].download_source 选择下载器。

    download_source 取值:
        "planetary_computer"  → PlanetaryComputerDownloader
        "gee"                 → GEEDownloader
        "local"               → LocalSARImporter
    """
    source_cfg = region_cfg["sources"].get(source_name, {})
    download_source = source_cfg.get("download_source", "")

    if download_source == "planetary_computer":
        from preprocessing.downloaders.planetary_computer import PlanetaryComputerDownloader
        return PlanetaryComputerDownloader(region_cfg, source_name)
    elif download_source == "gee":
        from preprocessing.downloaders.gee import GEEDownloader
        return GEEDownloader(region_cfg, source_name)
    elif download_source == "local":
        from preprocessing.downloaders.local_importer import LocalSARImporter
        return LocalSARImporter(region_cfg, source_name)
    else:
        raise ValueError(
            f"不支持的 download_source='{download_source}' "
            f"(source={source_name}, region={region_cfg['region_name']})"
        )
