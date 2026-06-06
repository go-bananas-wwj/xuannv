"""多区域训练 manifest 生成器。

从预处理配置文件中读取元数据，生成 configs/multi_region_manifest.json，
使 MultiRegionPatchDataset 能够加载多区域数据进行训练。

manifest 格式:
    {
        "<region_name>": {
            "data_root"   : str,          # 默认数据根，通常为 cloud_filtered_dir
            "stats_dir"   : str,          # 统计量目录
            "source_roots": {             # 可选：per-source 数据根覆盖
                "sar": "/workspace/xuannv/data_raw/phase2_haidian/haidian_scenes"
            },
            "sources"     : ["s2","s1",...],  # 该区域实际存在的源
            "patches"     : ["patch_000000", ...]
        }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from preprocessing.utils.logging import get_logger

logger = get_logger(__name__)


def generate_manifest(
    region_configs: list[dict],
    output_path: str | Path = "/workspace/xuannv/configs/multi_region_manifest.json",
) -> dict:
    """
    从多个区域配置生成合并 manifest。

    对每个区域：
        1. 从 patches_meta.json 读取 patch id 列表
        2. 确定各源是否可用（目录存在 + enabled）
        3. 记录 source_roots 覆盖（当某源数据在非默认目录时）

    Args:
        region_configs : 已加载的区域配置字典列表
        output_path    : 输出 manifest 路径

    Returns:
        manifest 字典
    """
    manifest: dict = {}

    for region_cfg in region_configs:
        region_name = region_cfg["region_name"]
        output_dir = Path(region_cfg["output_dir"])
        cloud_filtered_dir = Path(region_cfg.get("cloud_filtered_dir", region_cfg["output_dir"]))
        stats_dir = Path(region_cfg["statistics_dir"])

        # 加载 patch 列表
        meta_path = output_dir / "patches_meta.json"
        if not meta_path.exists():
            logger.warning(f"[{region_name}] patches_meta.json 不存在，跳过")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        patches = [f"patch_{p['id']:06d}" for p in meta["patches"]]

        # 过滤：只保留实际存在数据的 patch（至少有一个源有数据）
        valid_patches = []
        for pid in patches:
            # 检查 S2 是否存在（主要信号源）
            s2_dir = cloud_filtered_dir / "s2" / pid
            if s2_dir.exists() and any(s2_dir.glob("*.tif")):
                valid_patches.append(pid)
        if not valid_patches:
            valid_patches = patches  # fallback：若数据还在下载中，全部保留

        # 确定各源是否可用
        available_sources: list[str] = []
        source_roots: dict[str, str] = {}

        for src_name, src_cfg in region_cfg["sources"].items():
            if not src_cfg.get("enabled", False):
                continue
            dl_src = src_cfg.get("download_source", "")
            # 确定数据根目录
            if src_name == "s2":
                src_root = cloud_filtered_dir / "s2"
            elif dl_src == "local":
                # 本地 SAR：数据在 output_dir（不在 cloud_filtered_dir）
                src_root = output_dir / src_name
            else:
                # 其他源优先在 cloud_filtered_dir，fallback 到 output_dir
                src_root = cloud_filtered_dir / src_name
                if not src_root.exists():
                    src_root = output_dir / src_name

            if src_root.exists():
                available_sources.append(src_name)
                # 若与 cloud_filtered_dir 不同，记录覆盖
                expected_root = cloud_filtered_dir / src_name
                if str(src_root) != str(expected_root):
                    source_roots[src_name] = str(src_root.parent)  # 到 region root

        entry: dict = {
            "data_root": str(cloud_filtered_dir),
            "stats_dir": str(stats_dir),
            "sources": available_sources,
            "patches": valid_patches,
        }
        if source_roots:
            entry["source_roots"] = source_roots

        manifest[region_name] = entry
        logger.info(
            f"[{region_name}] {len(valid_patches)} patches, "
            f"可用源: {available_sources}, "
            f"source_roots 覆盖: {source_roots}"
        )

    # 写入文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"[Manifest] 已写入 {output_path}")
    return manifest
