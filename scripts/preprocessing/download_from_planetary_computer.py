#!/usr/bin/env python3
from __future__ import annotations
import sys; sys.stdout.reconfigure(line_buffering=True)
"""
Planetary Computer 批量下载脚本（备用方案）
支持 S2 / S1 / Landsat / DEM / WorldCover

特点:
- 使用 STAC API 搜索 + COG 按需裁剪，无 GEE export 配额限制
- S2/S1/Landsat 自动重投影到 patch 所在 UTM zone
- 支持断点续传（已存在的 .tif 跳过）
- 多线程并行下载
- S2 可选 divide(10000) 以保持与现有 GEE 数据格式一致

依赖:
    conda run -n xuannv pip install pystac-client planetary-computer stackstac odc-stac rioxarray

用法:
    # 下载齐齐哈尔全部 patch 的 S2
    python download_from_planetary_computer.py \
        --patches /workspace/raw/heilongjiang_new/qiqihar/patches_meta.json \
        --output /workspace/raw/heilongjiang_new/qiqihar \
        --sources s2 --workers 4

    # 下载大庆全部 patch 的 S1 + Landsat
    python download_from_planetary_computer.py \
        --patches /workspace/raw/heilongjiang_new/daqing/patches_meta.json \
        --output /workspace/raw/heilongjiang_new/daqing \
        --sources s1 landsat --workers 4
"""
import sys; sys.stdout.reconfigure(line_buffering=True)

# 限制底层库线程数，避免 ThreadPoolExecutor + Dask 嵌套导致死锁
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 设置 Dask 同步调度器，避免额外线程池
import dask
dask.config.set(scheduler="synchronous")

import argparse
import json
import time
import sys
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# 全局 STAC Catalog（每个 worker 线程独立连接更稳定）
# ---------------------------------------------------------------------------

DATE_START = "2023-01-01"
DATE_END = "2025-10-31"

COLLECTION_MAP = {
    "s2": "sentinel-2-l2a",
    "s1": "sentinel-1-grd",
    "landsat": "landsat-c2-l2",
    "dem": "cop-dem-glo-30",
    "worldcover": "esa-worldcover",
}

ASSET_MAP = {
    "s2": ["B02", "B03", "B04", "B05", "B06", "B07"],
    "s1": ["vv", "vh"],
    "landsat": ["red", "green", "blue", "nir08", "swir16", "lwir11"],
    "dem": ["data"],
    "worldcover": ["map"],
}

RESOLUTION_MAP = {
    "s2": 10,
    "s1": 10,
    "landsat": 30,
    "dem": 30,
    "worldcover": 10,
}

# 部分 source 不需要时间范围（静态数据）
STATIC_SOURCES = {"dem", "worldcover"}


def get_catalog():
    """每个线程独立的 catalog 连接"""
    import pystac_client
    import planetary_computer as pc
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )


def compute_bbox(center_lon: float, center_lat: float, patch_size_m: int = 1280) -> list[float]:
    """从中心点和 patch 大小计算 lonlat bbox"""
    # 在纬度 lat 处，1° lon ≈ 111km * cos(lat)
    lat_buf = patch_size_m / 2 / 111_000
    lon_buf = patch_size_m / 2 / (111_000 * np.cos(np.radians(center_lat)))
    return [
        center_lon - lon_buf,
        center_lat - lat_buf,
        center_lon + lon_buf,
        center_lat + lat_buf,
    ]


def determine_utm_epsg(center_lon: float, center_lat: float) -> int:
    """根据中心点计算 UTM zone EPSG code"""
    zone = int((center_lon + 180) / 6) + 1
    if center_lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def search_items(catalog, source: str, bbox: list[float], date_start: str, date_end: str):
    """STAC 搜索"""
    collection = COLLECTION_MAP[source]
    assets = ASSET_MAP[source]

    kwargs: dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox,
    }
    if source not in STATIC_SOURCES:
        kwargs["datetime"] = f"{date_start}/{date_end}"
    if source in ("s2", "landsat"):
        kwargs["query"] = {"eo:cloud_cover": {"lt": 20}}


    try:
        search = catalog.search(**kwargs)
        items = search.item_collection()
        return items
    except Exception as e:
        print(f"    [WARN] STAC 搜索失败 {source}: {e}")
        return []


def download_source_stackstac(
    item,
    source: str,
    bbox: list[float],
    epsg: int,
    divide_10000: bool = False,
):
    """使用 stackstac 下载并裁剪（适用于 S2, Landsat, DEM, WorldCover）"""
    import stackstac

    assets = ASSET_MAP[source]
    resolution = RESOLUTION_MAP[source]

    # 统一用 float64 读取（避免 fill_value 与 uint16 冲突），保存时再转 dtype
    stack = stackstac.stack(
        [item],
        assets=assets,
        bounds_latlon=bbox,
        resolution=resolution,
        rescale=False,
        dtype="float64",
        epsg=epsg,
        fill_value=0,
    )
    data = stack.compute()  # [time=1, band, y, x]
    data_np = np.asarray(data[0])  # [band, y, x]

    # 检查 band 数量（Landsat 偶发某个 band 缺失）
    expected_bands = len(assets)
    if data_np.shape[0] != expected_bands:
        raise ValueError(f"Band mismatch: expected {expected_bands}, got {data_np.shape[0]} for item {item.id}")

    if divide_10000 and source == "s2":
        data_np = data_np.astype(np.float32) / 10000.0
    elif source == "s2":
        data_np = data_np.astype(np.uint16)

    return data_np


def download_source_odcstac(
    item,
    source: str,
    bbox: list[float],
    epsg: int,
):
    """使用 odc-stac 下载并裁剪（适用于 S1，stackstac 对 S1 CRS 支持不完整）"""
    import odc.stac

    assets = ASSET_MAP[source]
    resolution = RESOLUTION_MAP[source]

    ds = odc.stac.load(
        [item],
        bands=assets,
        bbox=bbox,
        resolution=resolution,
        crs=f"EPSG:{epsg}",
    )
    data = ds.compute()

    # odc-stac 返回 Dataset，按 band 名提取
    bands_list = []
    for band in assets:
        if band not in data.data_vars:
            raise ValueError(f"Band '{band}' missing in item {item.id}")
        band_data = np.asarray(data[band][0])  # [y, x]
        bands_list.append(band_data)
    data_np = np.stack(bands_list, axis=0)  # [band, y, x]

    expected_bands = len(assets)
    if data_np.shape[0] != expected_bands:
        raise ValueError(f"Band mismatch: expected {expected_bands}, got {data_np.shape[0]} for item {item.id}")

    return data_np


def save_geotiff(path: Path, data: np.ndarray, bbox: list[float], epsg: int):
    """保存为 GeoTIFF"""
    _, h, w = data.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    dtype = data.dtype
    # rasterio 不支持 float64，降级到 float32
    if dtype == np.float64:
        data = data.astype(np.float32)
        dtype = np.float32

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=data.shape[0],
        dtype=dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data)


def download_patch(args) -> dict:
    """单 patch 多源下载（供 ThreadPoolExecutor 调用）"""
    patch, city_name, output_root, sources, date_start, date_end, divide_10000 = args

    patch_id = patch["id"]
    center_lon, center_lat = patch["center_lonlat"]
    bbox = compute_bbox(center_lon, center_lat)
    epsg = determine_utm_epsg(center_lon, center_lat)

    catalog = get_catalog()
    results = {"patch_id": patch_id, "sources": {}}

    for source in sources:
        t0 = time.time()
        patch_dir = Path(output_root) / city_name / source / f"patch_{patch_id:06d}"
        patch_dir.mkdir(parents=True, exist_ok=True)

        items = search_items(catalog, source, bbox, date_start, date_end)
        n_items = len(items)
        
        # 手动过滤 Landsat 7（PC STAC API 不支持 platform query 过滤）
        if source == "landsat":
            items = [item for item in items if item.properties.get("platform") in ("landsat-8", "landsat-9")]
            n_items = len(items)
        
        downloaded = 0
        skipped = 0
        failed_items = 0

        for item in items:
            try:
                # 日期命名
                dt = item.datetime
                if dt is None:
                    # fallback: 从 id 中解析日期
                    date_str = item.id.split("_")[-1][:8]
                else:
                    date_str = dt.strftime("%Y%m%d")

                out_path = patch_dir / f"{date_str}.tif"
                if out_path.exists():
                    skipped += 1
                    continue

                # 下载
                if source == "s1":
                    data_np = download_source_odcstac(item, source, bbox, epsg)
                else:
                    data_np = download_source_stackstac(item, source, bbox, epsg, divide_10000)

                save_geotiff(out_path, data_np, bbox, epsg)
                downloaded += 1
            except Exception as e:
                failed_items += 1
                # 打印到 stderr 以便调试
                print(f"    [WARN] patch_{patch_id:06d} {source} item {item.id if hasattr(item, 'id') else 'unknown'} failed: {e}", file=sys.stderr)
                continue

        status = "ok" if downloaded > 0 else ("failed" if n_items > 0 else "no_data")
        results["sources"][source] = {
            "status": status,
            "n_items": n_items,
            "downloaded": downloaded,
            "skipped": skipped,
            "failed_items": failed_items,
            "time_s": round(time.time() - t0, 1),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="从 Planetary Computer 批量下载遥感数据")
    parser.add_argument("--patches", required=True, help="patches_meta.json 路径")
    parser.add_argument("--output", required=True, help="输出根目录")
    parser.add_argument("--sources", nargs="+", default=["s2"],
                        choices=["s2", "s1", "landsat", "dem", "worldcover"],
                        help="要下载的数据源")
    parser.add_argument("--workers", type=int, default=4, help="并行 worker 数")
    parser.add_argument("--date-start", default=DATE_START, help="起始日期")
    parser.add_argument("--date-end", default=DATE_END, help="结束日期")
    parser.add_argument("--divide-10000", action="store_true",
                        help="S2 数据除以 10000（保持与现有 GEE 数据格式一致）")
    parser.add_argument("--limit", type=int, default=0, help="限制下载 patch 数量（0=全部）")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 个 patch")
    args = parser.parse_args()

    # 加载 patches
    with open(args.patches) as f:
        meta = json.load(f)

    city_name = meta.get("city", "unknown")
    patches = meta["patches"]
    total = len(patches)

    patches = patches[args.offset:]
    if args.limit > 0:
        patches = patches[:args.limit]

    print(f"城市: {city_name} ({meta.get('city_name', '')})")
    print(f"总 patch 数: {total}, 本次处理: {len(patches)} (offset={args.offset}, limit={args.limit})")
    print(f"数据源: {args.sources}")
    print(f"时间范围: {args.date_start} ~ {args.date_end}")
    print(f"S2 divide(10000): {args.divide_10000}")
    print(f"并行 workers: {args.workers}")
    print("=" * 60)

    # 并行下载
    task_args = [
        (p, city_name, args.output, args.sources, args.date_start, args.date_end, args.divide_10000)
        for p in patches
    ]

    start_time = time.time()
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_patch, a): a for a in task_args}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                patch_id = result["patch_id"]

                # 统计
                has_error = any(v.get("status") == "error" for v in result["sources"].values())
                if has_error:
                    error_count += 1
                else:
                    success_count += 1

                # 打印进度（每 10 个或出错时打印）
                if (i + 1) % 10 == 0 or has_error:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(futures) - (i + 1))
                    status_parts = []
                    for src, info in result["sources"].items():
                        if info.get("status") == "ok":
                            status_parts.append(f"{src}:{info['downloaded']}/{info['n_items']}")
                        else:
                            status_parts.append(f"{src}:ERR({info.get('error','')[:20]})")
                    print(f"[{i+1}/{len(futures)}] patch_{patch_id:06d} | "
                          f"{' | '.join(status_parts)} | "
                          f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")

            except Exception as e:
                error_count += 1
                print(f"[{i+1}/{len(futures)}] 异常: {e}")
                traceback.print_exc()

    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"下载完成: 成功 {success_count}, 失败 {error_count}, 总计 {len(futures)}")
    print(f"总耗时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"平均每个 patch: {total_time/len(futures):.1f}s")


if __name__ == "__main__":
    main()
