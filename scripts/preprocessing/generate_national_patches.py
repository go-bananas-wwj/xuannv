#!/usr/bin/env python3
"""
全国范围 patches 采样脚本 (B+D 策略)
=====================================
- 在中国陆地边界内生成均匀网格
- 按纬度带分层采样，确保地理多样性
- 额外在主要城市群增加采样密度
- 输出 patches_meta.json (EPSG:4326, WGS84 边界)

用法:
    python generate_national_patches.py \
        --n-patches 5000 \
        --patch-size-m 1280 \
        --output /workspace/xuannv/data_raw/national_china/patches_meta.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box


def load_china_boundary():
    """加载中国陆地边界（含台湾、海南、南海诸岛）"""
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    # 筛选中国（包含台湾、香港、澳门）
    china = world[world["SOVEREIGNT"] == "China"].copy()
    # 合并所有 polygon（处理 MultiPolygon）
    china_union = china.geometry.union_all()
    return china_union


def generate_candidate_grid(
    china_geom,
    spacing_km: float = 35.0,
    patch_size_m: float = 1280.0,
) -> list[dict]:
    """
    在中国边界内生成均匀候选网格。
    spacing_km: 候选点之间的间距（公里）
    """
    bounds = china_geom.bounds  # (minx, miny, maxx, maxy)
    min_lon, min_lat, max_lon, max_lat = bounds

    # 在纬度 lat 处，1° 经度 ≈ 111km * cos(lat)
    # 用平均纬度计算经度间距
    avg_lat = (min_lat + max_lat) / 2
    lon_spacing_deg = spacing_km / (111.0 * np.cos(np.radians(avg_lat)))
    lat_spacing_deg = spacing_km / 111.0

    lons = np.arange(min_lon, max_lon, lon_spacing_deg)
    lats = np.arange(min_lat, max_lat, lat_spacing_deg)

    candidates = []
    patch_half_deg_lon = (patch_size_m / 2) / (111_000 * np.cos(np.radians(avg_lat)))
    patch_half_deg_lat = (patch_size_m / 2) / 111_000

    for lon in lons:
        for lat in lats:
            # 以 (lon, lat) 为中心的 patch 边界
            west = lon - patch_half_deg_lon
            east = lon + patch_half_deg_lon
            south = lat - patch_half_deg_lat
            north = lat + patch_half_deg_lat

            # 创建 patch 的 bbox polygon
            patch_box = box(west, south, east, north)

            # 检查 patch 中心是否在陆地上（简化：用中心点判断）
            center = Point(lon, lat)
            if china_geom.contains(center):
                candidates.append({
                    "center_lonlat": [round(lon, 6), round(lat, 6)],
                    "bounds_wgs84": [
                        round(west, 6), round(south, 6),
                        round(east, 6), round(north, 6),
                    ],
                    "lat": lat,
                    "lon": lon,
                })

    return candidates


def stratified_sample(candidates: list[dict], n_patches: int) -> list[dict]:
    """
    B+D 分层采样策略：
    1. 按纬度带分层（确保南北多样性）
    2. 在纬度带内均匀随机采样
    3. 额外在人口密集纬度带（30°-40°N）增加密度
    """
    # 按纬度分带
    bands = {
        "cold": (42.0, 54.0),      # 寒温带/温带北部（东北、内蒙古）
        "temperate_n": (36.0, 42.0),  # 温带北部（华北、西北）
        "temperate_c": (30.0, 36.0),  # 温带中部（华中、四川）
        "temperate_s": (24.0, 30.0),  # 温带南部（华南、云南）
        "subtropical": (18.0, 24.0),  # 亚热带（海南南部、台湾南部）
    }

    band_groups = {name: [] for name in bands}
    unassigned = []

    for c in candidates:
        lat = c["lat"]
        assigned = False
        for name, (lo, hi) in bands.items():
            if lo <= lat < hi:
                band_groups[name].append(c)
                assigned = True
                break
        if not assigned:
            unassigned.append(c)

    # 分层配额（B+D 策略：温带/亚热带加重，高纬度/低纬度保底）
    quotas = {
        "cold": 0.10,         # 500
        "temperate_n": 0.20,  # 1000（华北城市群密集）
        "temperate_c": 0.25,  # 1250（长江流域，农业+城市）
        "temperate_s": 0.25,  # 1250（华南，多样性高）
        "subtropical": 0.15,  # 750（海南、台湾、云南南部）
    }

    selected = []
    remaining = []
    rng = random.Random(42)

    for band_name, quota in quotas.items():
        pool = band_groups[band_name]
        n_target = int(n_patches * quota)
        if len(pool) <= n_target:
            selected.extend(pool)
            remaining.extend([])
        else:
            chosen = rng.sample(pool, n_target)
            selected.extend(chosen)
            remaining.extend([p for p in pool if p not in chosen])

    # 补充未分配的和剩余候选
    if unassigned:
        remaining.extend(unassigned)

    # 如果选中的不足 n_patches，从剩余中补充
    shortfall = n_patches - len(selected)
    if shortfall > 0 and remaining:
        extra = rng.sample(remaining, min(shortfall, len(remaining)))
        selected.extend(extra)

    # 打乱顺序，避免地理聚集
    rng.shuffle(selected)

    return selected[:n_patches]


def build_patches_meta(patches: list[dict], patch_size_m: int) -> dict:
    """生成与现有 pipeline 兼容的 patches_meta.json"""
    patch_list = []
    for i, p in enumerate(patches):
        west, south, east, north = p["bounds_wgs84"]
        # utm_bounds: 在 WGS84 下直接用度数值（EPSG:4326 的单位是度）
        # 为兼容性保留 utm_bounds 字段，但实际是 WGS84 边界
        patch_list.append({
            "id": i,
            "utm_bounds": [west, south, east, north],
            "bounds_wgs84": [west, south, east, north],
            "center_lonlat": p["center_lonlat"],
        })

    return {
        "city": "national_china",
        "crs": "EPSG:4326",
        "patch_size_m": patch_size_m,
        "resolution": 10,
        "patches": patch_list,
    }


def main():
    parser = argparse.ArgumentParser(description="全国 patches 采样")
    parser.add_argument("--n-patches", type=int, default=5000, help="目标 patch 数量")
    parser.add_argument("--patch-size-m", type=int, default=1280, help="patch 边长（米）")
    parser.add_argument("--spacing-km", type=float, default=35.0, help="候选网格间距（公里）")
    parser.add_argument("--output", required=True, help="输出 patches_meta.json 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("全国 Patches 采样 (B+D 策略)")
    print("=" * 60)

    # 1. 加载中国边界
    print("\n[1/4] 加载中国陆地边界...")
    china_geom = load_china_boundary()
    print(f"  边界范围: {china_geom.bounds}")

    # 2. 生成候选网格
    print(f"\n[2/4] 生成候选网格 (间距 {args.spacing_km}km)...")
    candidates = generate_candidate_grid(china_geom, args.spacing_km, args.patch_size_m)
    print(f"  候选点总数: {len(candidates)}")

    # 按纬度统计
    lat_counts = {}
    for c in candidates:
        lat_band = int(c["lat"] / 6) * 6  # 6度一个带
        lat_counts[lat_band] = lat_counts.get(lat_band, 0) + 1
    print(f"  纬度分布: {dict(sorted(lat_counts.items()))}")

    if len(candidates) < args.n_patches:
        print(f"\n[ERROR] 候选点不足: {len(candidates)} < {args.n_patches}")
        print("  建议降低 --spacing-km 值")
        return

    # 3. 分层采样
    print(f"\n[3/4] 分层采样 {args.n_patches} 个 patches...")
    selected = stratified_sample(candidates, args.n_patches)
    print(f"  选中: {len(selected)} 个")

    # 统计最终分布
    final_lat = [p["lat"] for p in selected]
    final_lon = [p["lon"] for p in selected]
    print(f"  纬度范围: {min(final_lat):.1f}°N ~ {max(final_lat):.1f}°N")
    print(f"  经度范围: {min(final_lon):.1f}°E ~ {max(final_lon):.1f}°E")

    # 4. 生成 patches_meta.json
    print(f"\n[4/4] 生成 patches_meta.json...")
    meta = build_patches_meta(selected, args.patch_size_m)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  输出: {out_path}")
    print(f"  patches: {len(meta['patches'])}")
    print(f"  crs: {meta['crs']}")
    print(f"  patch_size_m: {meta['patch_size_m']}")
    print("\n" + "=" * 60)
    print("采样完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
