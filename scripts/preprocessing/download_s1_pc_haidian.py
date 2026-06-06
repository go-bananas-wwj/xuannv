#!/usr/bin/env python3
"""
海淀区 S1 RTC 下载脚本（Planetary Computer）
============================================
适配现有 patches_meta.json 格式，使用 stackstac 批量下载。
目标：替换现有 GEE S1 数据，解决与 S2 不对齐的问题。
"""
from __future__ import annotations
import sys; sys.stdout.reconfigure(line_buffering=True)

import os
# 清除代理
for proxy_key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(proxy_key, None)
os.environ["no_proxy"] = "*"
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
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_bounds

import stackstac
import pystac_client
import planetary_computer as pc


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
COLLECTION = "sentinel-1-rtc"
ASSETS = ["vv", "vh"]
RESOLUTION = 10

# 缓存 catalog 连接
_CATALOG_CACHE = {}


def get_catalog():
    tid = __import__("threading").get_ident()
    if tid not in _CATALOG_CACHE:
        _CATALOG_CACHE[tid] = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
    return _CATALOG_CACHE[tid]


def search_s1_items(catalog, bbox_wgs84: list[float], date_start: str, date_end: str):
    """搜索 S1 RTC items，带重试"""
    kwargs = {
        "collections": [COLLECTION],
        "bbox": bbox_wgs84,
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "max_items": 50,
    }
    for attempt in range(3):
        try:
            search = catalog.search(**kwargs)
            items = list(search.items())
            # 修复缺少 proj:epsg 的问题
            for item in items:
                if item.properties.get("proj:epsg") is None:
                    proj_code = item.properties.get("proj:code")
                    if proj_code and isinstance(proj_code, str) and proj_code.upper().startswith("EPSG:"):
                        try:
                            item.properties["proj:epsg"] = int(proj_code.split(":")[1])
                        except ValueError:
                            pass
            return items
        except Exception as e:
            if attempt < 2:
                wait = 5 + attempt * 5
                print(f"    [WARN] STAC 搜索失败 (尝试 {attempt+1}/3): {e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"    [WARN] STAC 搜索失败 (3次均失败): {e}")
                return []
    return []


def download_patch_s1(args) -> dict:
    """单 patch S1 下载"""
    patch, output_root, date_start, date_end = args

    patch_id = patch["patch_id"]
    bbox_wgs84 = patch["bounds_wgs84"]  # [west, south, east, north]
    bbox_utm = patch["bounds"]          # [left, bottom, right, top]
    crs = patch["crs"]                  # e.g. EPSG:32650
    epsg = int(crs.split(":")[1])

    patch_dir = Path(output_root) / patch_id / "s1"
    patch_dir.mkdir(parents=True, exist_ok=True)

    catalog = get_catalog()
    t0 = time.time()

    # 1. 搜索
    items = search_s1_items(catalog, bbox_wgs84, date_start, date_end)
    if not items:
        return {"patch_id": patch_id, "status": "no_data", "downloaded": 0, "time_s": round(time.time()-t0, 1)}

    # 2. 过滤缺少 vv/vh 的 items
    valid_items = []
    for item in items:
        available = set(item.assets.keys())
        if all(a in available for a in ASSETS):
            valid_items.append(item)
        else:
            missing = [a for a in ASSETS if a not in available]
            print(f"      [SKIP] {item.id} missing bands: {missing}")
    items = valid_items
    if not items:
        return {"patch_id": patch_id, "status": "no_assets", "downloaded": 0, "time_s": round(time.time()-t0, 1)}

    # 3. 预签名
    items = [pc.sign(it) for it in items]

    # 4. 过滤已存在文件
    items_to_download = []
    skipped = 0
    for item in items:
        dt = item.datetime
        date_str = dt.strftime("%Y%m%d") if dt else item.id.split("_")[-1][:8]
        out_path = patch_dir / f"{date_str}.tif"
        if out_path.exists() and out_path.stat().st_size > 2048:
            skipped += 1
        else:
            items_to_download.append((item, date_str, out_path))

    downloaded = 0
    failed = 0

    if not items_to_download:
        return {"patch_id": patch_id, "status": "skipped", "downloaded": 0, "skipped": skipped, "time_s": round(time.time()-t0, 1)}

    # 5. 批量下载（mini-batch stackstac）
    mini_batch = 3
    for mb_start in range(0, len(items_to_download), mini_batch):
        mb = items_to_download[mb_start:mb_start + mini_batch]
        mb_items = [it[0] for it in mb]

        try:
            # 为 S1 设置更短的 GDAL HTTP 超时
            from stackstac.rio_env import LayeredEnv
            gdal_env = LayeredEnv(
                always=dict(
                    GDAL_HTTP_MULTIRANGE="YES",
                    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                    GDAL_HTTP_TIMEOUT="15",
                    GDAL_HTTP_MAX_RETRY="2",
                    GDAL_HTTP_RETRY_DELAY="2",
                ),
                open=dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True),
                read=dict(VSI_CACHE=False),
            )

            stack = stackstac.stack(
                mb_items,
                assets=ASSETS,
                bounds=bbox_utm,
                resolution=RESOLUTION,
                rescale=False,
                dtype="float64",
                epsg=epsg,
                fill_value=0,
                gdal_env=gdal_env,
            )
            data = stack.compute()  # [time, band, y, x]
        except Exception as e:
            err_str = str(e)
            if "403" in err_str or "HTTP response code" in err_str:
                print(f"      [RETRY] mini-batch {mb_start} token expired, re-signing...")
                mb_items = [pc.sign(it) for it in mb_items]
                try:
                    stack = stackstac.stack(
                        mb_items,
                        assets=ASSETS,
                        bounds=bbox_utm,
                        resolution=RESOLUTION,
                        rescale=False,
                        dtype="float64",
                        epsg=epsg,
                        fill_value=0,
                        gdal_env=gdal_env,
                    )
                    data = stack.compute()
                except Exception as e2:
                    print(f"      [WARN] mini-batch {mb_start} re-sign failed: {e2}")
                    failed += len(mb)
                    continue
            else:
                print(f"      [WARN] mini-batch {mb_start} failed: {e}")
                # fallback: 逐个尝试
                for item, date_str, out_path in mb:
                    try:
                        single_stack = stackstac.stack(
                            [item],
                            assets=ASSETS,
                            bounds=bbox_utm,
                            resolution=RESOLUTION,
                            rescale=False,
                            dtype="float64",
                            epsg=epsg,
                            fill_value=0,
                            gdal_env=gdal_env,
                        )
                        single_data = single_stack.compute()
                        data_np = np.asarray(single_data[0])  # [band, y, x]
                        if data_np.shape[0] != len(ASSETS):
                            print(f"      [WARN] Band mismatch for {item.id}")
                            failed += 1
                            continue
                        if np.all(data_np == 0):
                            print(f"      [WARN] All-zero data for {item.id}")
                            failed += 1
                            continue
                        _save(out_path, data_np, bbox_utm, epsg)
                        downloaded += 1
                    except Exception as e2:
                        print(f"      [WARN] item {item.id} fallback failed: {e2}")
                        failed += 1
                continue

        # 处理 batch 结果
        for i, (item, date_str, out_path) in enumerate(mb):
            try:
                data_np = np.asarray(data[i])  # [band, y, x]
                if data_np.shape[0] != len(ASSETS):
                    print(f"      [WARN] Band mismatch for {item.id}: expected {len(ASSETS)}, got {data_np.shape[0]}")
                    failed += 1
                    continue
                if np.all(data_np == 0):
                    print(f"      [WARN] All-zero data for {item.id}")
                    failed += 1
                    continue
                _save(out_path, data_np, bbox_utm, epsg)
                downloaded += 1
            except Exception as e:
                print(f"      [WARN] item {item.id} failed: {e}")
                failed += 1

    status = "ok" if downloaded > 0 else ("failed" if len(items_to_download) > 0 else "no_data")
    return {
        "patch_id": patch_id,
        "status": status,
        "n_items": len(items),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "time_s": round(time.time() - t0, 1),
    }


def _save(path: Path, data: np.ndarray, bbox_utm: list[float], epsg: int):
    _, h, w = data.shape
    transform = from_bounds(bbox_utm[0], bbox_utm[1], bbox_utm[2], bbox_utm[3], w, h)
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


def main():
    parser = argparse.ArgumentParser(description="海淀区 S1 RTC PC 下载")
    parser.add_argument("--patches", required=True, help="patches_meta.json 路径")
    parser.add_argument("--output", required=True, help="输出根目录")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--date-start", default="2025-01-01")
    parser.add_argument("--date-end", default="2026-04-30")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    with open(args.patches) as f:
        patches = json.load(f)

    # 支持两种格式: list 或 dict with 'patches' key
    if isinstance(patches, dict):
        patches = patches["patches"]

    patches = patches[args.offset:]
    if args.limit > 0:
        patches = patches[:args.limit]

    print(f"总 patch 数: {len(patches)} (offset={args.offset}, limit={args.limit})")
    print(f"时间范围: {args.date_start} ~ {args.date_end}")
    print(f"并行 workers: {args.workers}")
    print(f"输出目录: {args.output}")
    print("=" * 60)

    task_args = [
        (p, args.output, args.date_start, args.date_end)
        for p in patches
    ]

    start_time = time.time()
    success = 0
    error = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_patch_s1, a): a for a in task_args}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                pid = result["patch_id"]
                st = result["status"]
                if st in ("ok", "skipped"):
                    success += 1
                else:
                    error += 1
                if (i + 1) % 10 == 0 or st not in ("ok", "skipped"):
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (len(futures) - (i + 1))
                    print(f"[{i+1}/{len(futures)}] {pid} | status={st} | dl={result.get('downloaded',0)} | skip={result.get('skipped',0)} | elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
            except Exception as e:
                error += 1
                print(f"[{i+1}/{len(futures)}] 异常: {e}")
                traceback.print_exc()

    total = time.time() - start_time
    print()
    print("=" * 60)
    print(f"下载完成: 成功 {success}, 失败 {error}, 总计 {len(futures)}")
    print(f"总耗时: {total/60:.1f} 分钟 ({total/3600:.2f} 小时)")


if __name__ == "__main__":
    main()
