#!/usr/bin/env python3
"""
Planetary Computer 下载速度测试脚本
测试 S2 / S1 / Landsat 从 STAC 搜索到裁剪下载的完整链路速度
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PATCH_META = Path("/workspace/raw/heilongjiang_new/qiqihar/patches_meta.json")
OUTPUT_LOG = Path("/workspace/xuannv/pc_speed_test_result.json")

# 选一个 patch（齐齐哈尔中心区域，urban + rural 混合）
PATCH_ID = 100

# S2/Landsat 测试时间范围
TEST_DATETIME = "2023-06-01/2023-08-31"
# S1 用更宽的时间范围（该 patch S1 覆盖不均匀）
TEST_DATETIME_S1 = "2023-01-01/2025-10-31"

# 裁剪区域大小（米）
PATCH_SIZE_M = 1280

# ---------------------------------------------------------------------------
# 初始化 PC
# ---------------------------------------------------------------------------
import pystac_client
import planetary_computer as pc
import stackstac

print("=" * 70)
print("Planetary Computer 下载速度测试")
print(f"时间: {datetime.now().isoformat()}")
print("=" * 70)

# 连接 STAC API（匿名访问，无需 API Key）
t0 = time.time()
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)
print(f"STAC 连接耗时: {time.time() - t0:.2f}s")
print()

# ---------------------------------------------------------------------------
# 读取 patch 坐标
# ---------------------------------------------------------------------------
with open(PATCH_META) as f:
    meta = json.load(f)

patch = meta["patches"][PATCH_ID]
patch_id = patch["id"]
center_lon, center_lat = patch["center_lonlat"]

# 粗略换算：1° lon ≈ 111km * cos(lat), 1° lat ≈ 111km
# 在纬度 47°，1° lon ≈ 75.8km
lat_buffer = PATCH_SIZE_M / 2 / 111_000  # ~0.0058°
lon_buffer = PATCH_SIZE_M / 2 / (111_000 * np.cos(np.radians(center_lat)))  # ~0.0086°

bbox = [
    center_lon - lon_buffer,
    center_lat - lat_buffer,
    center_lon + lon_buffer,
    center_lat + lat_buffer,
]
print(f"Patch ID: {patch_id}")
print(f"Center: ({center_lon:.6f}, {center_lat:.6f})")
print(f"Bbox: [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}]")
print(f"测试时间范围: {TEST_DATETIME}")
print()

# ---------------------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------------------

def test_source(name: str, collection: str, assets: list[str], resolution: int,
                    cloud_filter: dict | None = None, datetime_override: str | None = None):
    """测试单个数据源的搜索+裁剪下载速度"""
    dt_range = datetime_override or TEST_DATETIME
    print(f"{'─' * 70}")
    print(f"测试数据源: {name} (collection={collection})")
    print(f"  assets={assets}, resolution={resolution}m, datetime={dt_range}")
    
    # --- STAC 搜索 ---
    t_search_start = time.time()
    try:
        kwargs = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": dt_range,
        }
        if cloud_filter:
            kwargs["query"] = cloud_filter
        
        search = catalog.search(**kwargs)
        items = search.item_collection()
        t_search = time.time() - t_search_start
        n_items = len(items)
        print(f"  STAC 搜索: 找到 {n_items} 景, 耗时 {t_search:.2f}s")
        
        if n_items == 0:
            print(f"  ⚠️  无数据，跳过下载测试")
            return {
                "source": name,
                "collection": collection,
                "n_items": 0,
                "search_time_s": t_search,
                "error": "no_items",
            }
    except Exception as e:
        print(f"  ❌ STAC 搜索失败: {e}")
        return {
            "source": name,
            "collection": collection,
            "error": f"search_failed: {e}",
        }
    
    # --- 裁剪下载第一景 ---
    # 限制只取最近 3 景，避免 stackstac 内存爆炸
    items_limited = items[:3]
    
    # 读取 EPSG（从第一个 item 的 proj:code）
    first_item = items_limited[0]
    proj_code = first_item.properties.get("proj:code", "")
    if proj_code.startswith("EPSG:"):
        epsg = int(proj_code.replace("EPSG:", ""))
    else:
        epsg = 4326  # fallback
    
    # S1 GRD 在 4326 下 resolution 单位是度，stackstac 会报错。
    # 统一使用 patch 所在 UTM zone (EPSG:32651) 重投影。
    if epsg == 4326:
        epsg = 32651
        print(f"  原始 EPSG:4326 → 强制使用 EPSG:32651 (UTM 51N) 以支持米单位裁剪")
    else:
        print(f"  使用 EPSG:{epsg} (from {proj_code})")
    
    t_dl_start = time.time()
    try:
        stack = stackstac.stack(
            items_limited,
            assets=assets,
            bounds_latlon=bbox,
            resolution=resolution,
            rescale=False,
            dtype="float64",
            epsg=epsg,
            fill_value=0,
        )
        print(f"  stackstac 构建: shape={stack.shape}, dims={list(stack.dims)}, 耗时 {time.time() - t_dl_start:.2f}s")
        
        # 触发实际下载
        t_compute_start = time.time()
        data = stack.compute()
        t_compute = time.time() - t_compute_start
        
        # 估算下载数据量
        data_bytes = data.nbytes
        data_mb = data_bytes / 1_048_576
        speed_mbps = data_mb / t_compute if t_compute > 0 else float('inf')
        
        print(f"  实际下载: shape={data.shape}, dtype={data.dtype}")
        print(f"  数据量: {data_mb:.2f} MB")
        print(f"  下载耗时: {t_compute:.2f}s")
        print(f"  平均速度: {speed_mbps:.2f} MB/s")
        print(f"  ✅ {name} 测试通过")
        
        return {
            "source": name,
            "collection": collection,
            "n_items": n_items,
            "n_items_used": len(items_limited),
            "search_time_s": round(t_search, 2),
            "build_time_s": round(time.time() - t_dl_start - t_compute, 2),
            "download_time_s": round(t_compute, 2),
            "data_mb": round(data_mb, 2),
            "speed_mbps": round(speed_mbps, 2),
            "data_shape": list(data.shape),
            "dtype": str(data.dtype),
        }
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return {
            "source": name,
            "collection": collection,
            "n_items": n_items,
            "search_time_s": round(t_search, 2),
            "error": f"download_failed: {e}",
        }


# ---------------------------------------------------------------------------
# 运行测试
# ---------------------------------------------------------------------------
results = []

# S2: 搜索 + 裁剪 (B02, B03, B04, B08, SCL)
r = test_source(
    name="S2",
    collection="sentinel-2-l2a",
    assets=["B02", "B03", "B04", "B08", "SCL"],
    resolution=10,
    cloud_filter={"eo:cloud_cover": {"lt": 20}},
)
results.append(r)

# S1: 搜索 + 裁剪 (VV, VH)
r = test_source(
    name="S1",
    collection="sentinel-1-grd",
    assets=["vv", "vh"],
    resolution=10,
    datetime_override=TEST_DATETIME_S1,
)
results.append(r)

# Landsat 8/9: 搜索 + 裁剪 (red, green, blue, nir08, swir16, lwir11)
r = test_source(
    name="Landsat",
    collection="landsat-c2-l2",
    assets=["red", "green", "blue", "nir08", "swir16", "lwir11"],
    resolution=30,
    cloud_filter={"eo:cloud_cover": {"lt": 20}},
)
results.append(r)

# ---------------------------------------------------------------------------
# 汇总输出
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("测试结果汇总")
print("=" * 70)

summary = {
    "timestamp": datetime.now().isoformat(),
    "patch_id": patch_id,
    "center": [center_lon, center_lat],
    "bbox": bbox,
    "datetime_range": TEST_DATETIME,
    "results": results,
}

for r in results:
    name = r["source"]
    if "error" in r:
        print(f"  {name}: ❌ {r['error']}")
    else:
        print(f"  {name}: {r['n_items']} 景 | "
              f"搜索 {r['search_time_s']:.1f}s | "
              f"下载 {r['data_mb']:.1f}MB/{r['download_time_s']:.1f}s = "
              f"{r['speed_mbps']:.2f} MB/s")

with open(OUTPUT_LOG, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print()
print(f"详细结果已保存到: {OUTPUT_LOG}")
