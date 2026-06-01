#!/usr/bin/env python3
"""
Planetary Computer 批量下载脚本 v2（优化版）
==============================================

相比 v1 的核心优化:
1. **去除按月拆分搜索** → STAC API 调用减少 ~18 倍
2. **批量 stackstac** → 将同一 patch 的所有 items 一次性 stack，减少 overhead
3. **减少 token 重新签名** → search 返回的 items 预签名一次，下载时复用
4. **快速跳过检查** → 用 os.path.getsize 替代 rasterio.open，避免 I/O 瓶颈
5. **批量保存** → 批量读取后逐个保存，减少 stackstac 调用次数

优化效果预估（基于 v1 实测）:
- 搜索阶段: 18x 加速（按月拆分 → 一次性搜索）
- 下载阶段: 2-3x 加速（批量 stack + 减少签名）
- 整体: 约 5-10x 加速

用法:
    python download_from_planetary_computer_v2.py \
        --patches /workspace/raw/national_china/patches_meta.json \
        --output /workspace/raw/national_china \
        --sources s2 --workers 16 \
        --date-start 2025-01-01 --date-end 2026-06-01
"""
from __future__ import annotations
import sys; sys.stdout.reconfigure(line_buffering=True)

import os
# 清除代理环境变量，确保直连 Planetary Computer（国内可直连，避免消耗梯子流量）
for proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(proxy_key, None)
os.environ["no_proxy"] = "*"  # 禁用所有代理

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dask
dask.config.set(scheduler="synchronous")

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

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
    "s2": 10, "s1": 10, "landsat": 30, "dem": 30, "worldcover": 10,
}

STATIC_SOURCES = {"dem", "worldcover"}

# 缓存的 catalog 连接（每个线程独立）
_CATALOG_CACHE = {}


def get_catalog():
    """每个线程独立的 catalog 连接（带缓存）"""
    import threading
    tid = threading.get_ident()
    if tid not in _CATALOG_CACHE:
        import pystac_client
        import planetary_computer as pc
        _CATALOG_CACHE[tid] = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
    return _CATALOG_CACHE[tid]


def compute_bbox(center_lon: float, center_lat: float, patch_size_m: int = 1280) -> list[float]:
    lat_buf = patch_size_m / 2 / 111_000
    lon_buf = patch_size_m / 2 / (111_000 * np.cos(np.radians(center_lat)))
    return [center_lon - lon_buf, center_lat - lat_buf, center_lon + lon_buf, center_lat + lat_buf]


def determine_utm_epsg(center_lon: float, center_lat: float) -> int:
    zone = int((center_lon + 180) / 6) + 1
    return 32600 + zone if center_lat >= 0 else 32700 + zone


def _search_with_retry(catalog, source, bbox, date_start, date_end):
    """一次性搜索整个时间范围（带重试）"""
    collection = COLLECTION_MAP[source]
    # WorldCover 2021 数据，强制使用 2021 年时间范围
    if source == "worldcover":
        date_start, date_end = "2021-01-01", "2021-12-31"
    kwargs = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
    }
    if source in ("s2", "landsat"):
        kwargs["query"] = {"eo:cloud_cover": {"lt": 20}}

    for attempt in range(3):
        try:
            search = catalog.search(**kwargs)
            items = search.item_collection()
            return list(items)
        except Exception as e:
            if attempt < 2:
                wait = 5 + attempt * 5
                print(f"    [WARN] STAC 搜索失败 {source} (尝试 {attempt+1}/3): {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    [WARN] STAC 搜索失败 {source} (3次均失败): {e}")
                return []
    return []


def pre_sign_items(items):
    """批量预签名所有 items（只需调用一次 planetary_computer.sign）"""
    import planetary_computer as pc
    return [pc.sign(item) for item in items]


def batch_download_stackstac(
    items: list,
    source: str,
    bbox: list[float],
    epsg: int,
    divide_10000: bool = False,
) -> dict[str, np.ndarray]:
    """
    批量下载同一 patch 的所有 items。
    将多个 item 一次性 stack，然后逐个提取，大幅减少 stackstac overhead。
    
    Returns: {date_str: data_np}
    """
    import stackstac

    if not items:
        return {}

    assets = ASSET_MAP[source]
    resolution = RESOLUTION_MAP[source]

    # 过滤掉缺少目标 assets 的 items（S1 EW 模式缺少 vv/vh，跳过）
    valid_items = []
    for item in items:
        available = set(item.assets.keys())
        if all(a in available for a in assets):
            valid_items.append(item)
        else:
            missing = [a for a in assets if a not in available]
            print(f"      [SKIP] {source} item {item.id} missing bands: {missing}")
    items = valid_items
    if not items:
        return {}

    # 一次性 stack 所有 items
    stack = stackstac.stack(
        items,
        assets=assets,
        bounds_latlon=bbox,
        resolution=resolution,
        rescale=False,
        dtype="float64",
        epsg=epsg,
        fill_value=0,
    )
    data = stack.compute()  # [time, band, y, x]

    results = {}
    expected_bands = len(assets)

    for i, item in enumerate(items):
        data_np = np.asarray(data[i])  # [band, y, x]

        if data_np.shape[0] != expected_bands:
            print(f"      [WARN] Band mismatch: expected {expected_bands}, got {data_np.shape[0]} for {item.id}")
            continue

        # 全0检测
        if source in ("s2", "landsat", "s1") and np.all(data_np == 0):
            print(f"      [WARN] All-zero data for {source} item {item.id}")
            continue

        # 数据转换
        if divide_10000 and source == "s2":
            data_np = data_np.astype(np.float32) / 10000.0
        elif source == "s2":
            data_np = data_np.astype(np.uint16)
        elif source == "landsat":
            data_np = data_np.astype(np.float32) * 0.0000275 - 0.2
            data_np = np.clip(data_np, 0.0, 1.0)

        # 日期命名
        dt = item.datetime
        if dt is None:
            date_str = item.id.split("_")[-1][:8]
        else:
            date_str = dt.strftime("%Y%m%d")

        results[date_str] = data_np

    return results


def batch_download_odcstac(
    items: list,
    source: str,
    bbox: list[float],
    epsg: int,
) -> dict[str, np.ndarray]:
    """批量下载 S1（odc-stac，逐个处理但复用连接）"""
    import odc.stac
    import planetary_computer as pc

    assets = ASSET_MAP[source]
    resolution = RESOLUTION_MAP[source]
    results = {}

    for item in items:
        try:
            item = pc.sign(item)
            ds = odc.stac.load(
                [item],
                bands=assets,
                bbox=bbox,
                resolution=resolution,
                crs=f"EPSG:{epsg}",
            )
            data = ds.compute()

            bands_list = []
            for band in assets:
                if band not in data.data_vars:
                    raise ValueError(f"Band '{band}' missing")
                bands_list.append(np.asarray(data[band][0]))
            data_np = np.stack(bands_list, axis=0)

            # 全0检测
            if np.all(data_np == 0):
                print(f"      [WARN] All-zero S1 data for {item.id}")
                continue

            dt = item.datetime
            date_str = dt.strftime("%Y%m%d") if dt else item.id.split("_")[-1][:8]
            results[date_str] = data_np

        except Exception as e:
            print(f"      [WARN] S1 item {item.id} failed: {e}")
            continue

    return results


def save_geotiff(path: Path, data: np.ndarray, bbox: list[float], epsg: int):
    _, h, w = data.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    dtype = data.dtype
    if dtype == np.float64:
        data = data.astype(np.float32)
        dtype = np.float32

    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=data.shape[0], dtype=dtype,
        crs=f"EPSG:{epsg}", transform=transform, compress="lzw",
    ) as dst:
        dst.write(data)


def quick_check_exists(out_path: Path, source: str) -> bool:
    """
    快速检查文件是否已存在且有效（不打开 rasterio）。
    只检查文件大小，避免昂贵的 I/O。
    """
    if not out_path.exists():
        return False

    try:
        size = out_path.stat().st_size
        # 极小文件 = 损坏/空
        if size < 2048:
            out_path.unlink()
            print(f"      [FIX] 删除损坏/空文件 ({size}b): {out_path.name}")
            return False
        return True
    except Exception:
        return False


def download_patch_v2(args) -> dict:
    """优化版单 patch 多源下载"""
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

        # === 1. 一次性搜索整个时间范围 ===
        items = _search_with_retry(catalog, source, bbox, date_start, date_end)
        n_items = len(items)

        # 过滤 Landsat 7
        if source == "landsat":
            items = [it for it in items if it.properties.get("platform") in ("landsat-8", "landsat-9")]
            n_items = len(items)

        if not items:
            results["sources"][source] = {
                "status": "no_data", "n_items": 0, "downloaded": 0,
                "skipped": 0, "failed_items": 0, "time_s": round(time.time() - t0, 1),
            }
            continue

        # === 2. 预签名所有 items（一次调用）===
        items = pre_sign_items(items)

        # === 3. 过滤已存在的文件 ===
        items_to_download = []
        skipped = 0
        for item in items:
            dt = item.datetime
            date_str = dt.strftime("%Y%m%d") if dt else item.id.split("_")[-1][:8]
            out_path = patch_dir / f"{date_str}.tif"
            if quick_check_exists(out_path, source):
                skipped += 1
            else:
                items_to_download.append(item)

        # === 4. 批量下载 ===
        downloaded = 0
        failed_items = 0

        if items_to_download:
            if source == "s1":
                batch_results = batch_download_odcstac(items_to_download, source, bbox, epsg)
            else:
                batch_results = batch_download_stackstac(
                    items_to_download, source, bbox, epsg, divide_10000
                )

            # 保存
            for date_str, data_np in batch_results.items():
                try:
                    out_path = patch_dir / f"{date_str}.tif"
                    save_geotiff(out_path, data_np, bbox, epsg)
                    downloaded += 1
                except Exception as e:
                    failed_items += 1
                    print(f"    [WARN] patch_{patch_id:06d} {source} save {date_str} failed: {e}", file=sys.stderr)

            failed_items += len(items_to_download) - len(batch_results) - failed_items

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
    parser = argparse.ArgumentParser(description="Planetary Computer 批量下载 v2（优化版）")
    parser.add_argument("--patches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sources", nargs="+", default=["s2"],
                        choices=["s2", "s1", "landsat", "dem", "worldcover"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--date-start", default="2023-01-01")
    parser.add_argument("--date-end", default="2025-10-31")
    parser.add_argument("--no-divide-10000", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    with open(args.patches) as f:
        meta = json.load(f)

    city_name = meta.get("city", "unknown")
    patches = meta["patches"]
    patches = patches[args.offset:]
    if args.limit > 0:
        patches = patches[:args.limit]

    print(f"城市: {city_name}")
    print(f"总 patch 数: {len(patches)} (offset={args.offset}, limit={args.limit})")
    print(f"数据源: {args.sources}")
    print(f"时间范围: {args.date_start} ~ {args.date_end}")
    print(f"S2 divide(10000): {not args.no_divide_10000}")
    print(f"并行 workers: {args.workers}")
    print("=" * 60)
    print("【优化版 v2】一次性搜索 + 批量 stack + 快速跳过")
    print("=" * 60)

    task_args = [
        (p, city_name, args.output, args.sources, args.date_start, args.date_end, not args.no_divide_10000)
        for p in patches
    ]

    start_time = time.time()
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_patch_v2, a): a for a in task_args}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                patch_id = result["patch_id"]
                has_error = any(v.get("status") == "error" for v in result["sources"].values())
                if has_error:
                    error_count += 1
                else:
                    success_count += 1

                if (i + 1) % 10 == 0 or has_error:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(futures) - (i + 1))
                    status_parts = []
                    for src, info in result["sources"].items():
                        status_parts.append(f"{src}:{info['downloaded']}/{info['n_items']}")
                    print(f"[{i+1}/{len(futures)}] patch_{patch_id:06d} | {' | '.join(status_parts)} | elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
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
