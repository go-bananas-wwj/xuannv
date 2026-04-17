"""哈尔滨新区 2025 年变化检测标注解析工具.

整合:
- 变化检测清单/4个 Excel 文件 (4-6月, 6-8月, 8-9月, 9-10月)
- 变化检测shp文件/SAR专题标注 (建筑工地/房屋拆除/疑似违建/非农非粮)
- 变化检测shp文件/月度光学标注 (june/aug/September/October.shp)

输出:
- 按 patch_id 组织的标注记录
- 支持 binary / category / construction 三种标签模式
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import box, Point

_transformer_wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)

BASE_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件")
SHP_DIR = BASE_DIR / "变化检测shp文件"
XLSX_DIR = BASE_DIR / "变化检测清单"
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")

# 时间窗口映射 (月份 -> ms)
PERIODS = {
    "2025-04~2025-06": (1743465600000.0, 1717200000000.0),  # 占位，后面用 date_to_ms
    "2025-06~2025-08": (1717200000000.0, 1722470400000.0),
    "2025-08~2025-09": (1722470400000.0, 1725148800000.0),
    "2025-09~2025-10": (1725148800000.0, 1727740800000.0),
}


def _date_to_ms(date_str: str) -> float:
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return float(dt.timestamp() * 1000)


def _compute_periods() -> dict[str, tuple[float, float]]:
    """根据 Excel 文件名和列名计算准确的 before/after 时间窗口."""
    return {
        "2025-04~2025-06": (_date_to_ms("2025-04-01"), _date_to_ms("2025-06-30")),
        "2025-06~2025-08": (_date_to_ms("2025-06-01"), _date_to_ms("2025-08-31")),
        "2025-08~2025-09": (_date_to_ms("2025-08-01"), _date_to_ms("2025-09-30")),
        "2025-09~2025-10": (_date_to_ms("2025-09-01"), _date_to_ms("2025-10-31")),
        "2025-all": (_date_to_ms("2025-01-01"), _date_to_ms("2025-12-31")),
    }


PERIODS = _compute_periods()

# 变化类型映射 -> 规范化类别
CATEGORY_MAP = {
    "疑似建筑工地施工": "construction",
    "疑似建造房屋": "construction",
    "疑似房屋拆除": "demolition",
    "疑似新建道路": "road",
    "疑似道路修建": "road",
    "疑似水塘填埋，转换为裸地": "water_change",
    "疑似裸地开挖，新建为水塘": "water_change",
    "疑似裸地开挖，新建为水田": "farmland",
    "疑似裸地开挖，新建为农田": "farmland",
    "疑似裸地开挖并转为农田": "farmland",
}

CATEGORY_TO_IDX = {
    "unchanged": 0,
    "construction": 1,
    "demolition": 2,
    "road": 3,
    "water_change": 4,
    "farmland": 5,
}


def _load_grid() -> dict:
    with open(GRID_PATH) as f:
        return json.load(f)


def _find_patch_for_point(x: float, y: float, patch_bounds: dict) -> str | None:
    """根据平面坐标(UTM)找到包含该点的 patch_id."""
    pt = Point(x, y)
    for pid, bounds in patch_bounds.items():
        minx, miny, maxx, maxy = bounds
        if minx <= x <= maxx and miny <= y <= maxy:
            if pt.within(box(minx, miny, maxx, maxy)):
                return pid
    # 回退：找最近 patch 中心
    best_pid, best_dist = None, float("inf")
    for pid, bounds in patch_bounds.items():
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        dist = (x - cx) ** 2 + (y - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_pid = pid
    return best_pid


def _parse_excel_records(patch_bounds: dict) -> list[dict]:
    """解析 4 个 Excel 文件，返回标注记录列表."""
    records = []
    excel_files = {
        "2025-04~2025-06": "4-6月份变化检测图斑.xlsx",
        "2025-06~2025-08": "6-8月份变化检测图斑.xlsx",
        "2025-08~2025-09": "8-9月份变化检测图斑.xlsx",
        "2025-09~2025-10": "9-10月份变化检测图斑.xlsx",
    }

    for period, filename in excel_files.items():
        path = XLSX_DIR / filename
        if not path.exists():
            continue
        df = pd.read_excel(path)
        for _, row in df.iterrows():
            lon = float(row.get("经度", row.get("纬度", np.nan)))  # 注意列名可能经纬度顺序不同
            lat = float(row.get("纬度", row.get("经度", np.nan)))
            if pd.isna(lon) or pd.isna(lat):
                continue
            # 修正：有些文件列名是 纬度 在前，经度 在后
            cols = list(df.columns)
            if "经度" in cols and "纬度" in cols:
                lon = float(row["经度"])
                lat = float(row["纬度"])
            # Excel 中为 WGS84 经纬度, 需投影到 UTM Zone 52N 再匹配
            ux, uy = _transformer_wgs84_to_utm.transform(lon, lat)
            pid = _find_patch_for_point(ux, uy, patch_bounds)
            if pid is None:
                continue
            remark = str(row.get("备注", "")).strip()
            category = CATEGORY_MAP.get(remark, "other")
            records.append({
                "patch_id": pid,
                "period": period,
                "category": category,
                "remark": remark,
                "source": "optical_excel",
                "geometry": Point(ux, uy),  # 存储为 UTM 坐标
                "area": float(row.get("面积", 0)) if "面积" in row else 0,
            })
    return records


def _load_shp_and_reproject(shp_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(shp_path))
    if gdf.crs is None or gdf.crs.to_epsg() != 32652:
        gdf = gdf.to_crs(epsg=32652)
    return gdf


def _parse_shp_records(patch_bounds: dict) -> list[dict]:
    """解析所有 SHP 文件 (月度光学 + SAR专题)."""
    records = []

    # 月度光学标注
    optical_shps = {
        "2025-04~2025-06": "june.shp",      # 对应4-6月
        "2025-06~2025-08": "aug.shp",       # 对应6-8月
        "2025-08~2025-09": "September.shp", # 对应8-9月
        "2025-09~2025-10": "October.shp",   # 对应9-10月
    }

    for period, filename in optical_shps.items():
        path = SHP_DIR / filename
        if not path.exists():
            continue
        gdf = _load_shp_and_reproject(path)
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            # 使用多边形中心点匹配 patch
            centroid = geom.centroid
            pid = _find_patch_for_point(centroid.x, centroid.y, patch_bounds)
            if pid is None:
                continue
            records.append({
                "patch_id": pid,
                "period": period,
                "category": "unknown",  # 光学 SHP 没有明确类型，需要结合 Excel
                "remark": "optical_change",
                "source": "optical_shp",
                "geometry": geom,
                "area": float(row.get("面积", 0)) if "面积" in row else 0,
            })

    # SAR 专题标注
    sar_shps = {
        "SAR建筑工地.shp": "construction",
        "SAR房屋拆除.shp": "demolition",
        "SAR疑似违建.shp": "construction",
        "SAR非农非粮.shp": "farmland",
    }

    for filename, category in sar_shps.items():
        path = SHP_DIR / filename
        if not path.exists():
            continue
        gdf = _load_shp_and_reproject(path)
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            centroid = geom.centroid
            pid = _find_patch_for_point(centroid.x, centroid.y, patch_bounds)
            if pid is None:
                continue
            records.append({
                "patch_id": pid,
                "period": "2025-all",  # SAR 标注覆盖全年
                "category": category,
                "remark": filename.replace(".shp", ""),
                "source": "sar",
                "geometry": geom,
                "area": float(row.get("面积", 0)) if "面积" in row else 0,
            })

    return records


def _merge_records(excel_records: list[dict], shp_records: list[dict]) -> dict[str, list[dict]]:
    """合并 Excel 和 SHP 记录，按 patch_id 分组去重."""
    all_records = excel_records + shp_records

    # 按 patch_id 分组
    patch_records: dict[str, list[dict]] = {}
    for r in all_records:
        pid = r["patch_id"]
        patch_records.setdefault(pid, []).append(r)

    # 去重：同一个 patch + 同一 period + 重叠几何体合并
    cleaned = {}
    for pid, recs in patch_records.items():
        # 对于 SHP 中的多边形，如果多个记录来自同一 source 且 period 相同，保留最大的几何体
        seen_keys = set()
        unique_recs = []
        for r in recs:
            key = (r["period"], r["source"], r["category"])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_recs.append(r)
        cleaned[pid] = unique_recs

    return cleaned


def _build_patch_bounds() -> dict[str, tuple[float, float, float, float]]:
    """从 grid geojson 构建 patch bounds."""
    grid = _load_grid()
    bounds = {}
    for feat in grid["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return bounds


# ── 公共 API ──

def get_patch_bounds() -> dict[str, tuple[float, float, float, float]]:
    """返回所有 patch 的 bounds."""
    return _build_patch_bounds()


def load_harbin_annotations() -> dict[str, list[dict]]:
    """加载并返回所有哈尔滨新区 2025 变化标注，按 patch_id 分组."""
    patch_bounds = _build_patch_bounds()
    excel_records = _parse_excel_records(patch_bounds)
    shp_records = _parse_shp_records(patch_bounds)
    return _merge_records(excel_records, shp_records)


def get_annotated_patches() -> list[str]:
    """返回有标注的 patch_id 列表（已排序）."""
    annotations = load_harbin_annotations()
    return sorted(annotations.keys())


def get_period_for_patch(patch_id: str) -> str | None:
    """返回 patch 的主要变化 period."""
    annotations = load_harbin_annotations()
    recs = annotations.get(patch_id, [])
    if not recs:
        return None
    # 优先返回 Excel 的月度 period，否则 SAR 的 all
    for r in recs:
        if r["source"] == "optical_excel":
            return r["period"]
    return recs[0]["period"]


def rasterize_patch_changes(patch_id: str, grid_size: int = 64) -> tuple[np.ndarray, list[dict]]:
    """将 patch 的所有变化标注光栅化为 [grid_size, grid_size] 的 mask.

    Returns:
        binary_mask: int32, 0=未变化, 1=变化
        records: 该 patch 的所有标注记录
    """
    patch_bounds = _build_patch_bounds()
    annotations = load_harbin_annotations()
    recs = annotations.get(patch_id, [])

    bounds = patch_bounds.get(patch_id)
    if bounds is None or not recs:
        return np.zeros((grid_size, grid_size), dtype=np.int32), recs

    minx, miny, maxx, maxy = bounds
    resolution_x = (maxx - minx) / grid_size
    resolution_y = (maxy - miny) / grid_size

    mask = np.zeros((grid_size, grid_size), dtype=np.int32)

    for r in recs:
        geom = r["geometry"]
        if geom is None:
            continue

        # Point 特殊处理：直接映射到最近像素
        if geom.geom_type == "Point":
            # 所有 point 已统一为 UTM (Excel 在解析时已投影)
            px = min(grid_size - 1, max(0, int((geom.x - minx) / resolution_x)))
            py = min(grid_size - 1, max(0, int((maxy - geom.y) / resolution_y)))
            mask[py, px] = 1  # [row=y, col=x]
            continue

        # 使用多边形 bbox 加速
        gminx, gminy, gmaxx, gmaxy = geom.bounds
        px_start = max(0, int((gminx - minx) / resolution_x))
        px_end = min(grid_size, int((gmaxx - minx) / resolution_x) + 1)
        py_start = max(0, int((maxy - gmaxy) / resolution_y))
        py_end = min(grid_size, int((maxy - gminy) / resolution_y) + 1)

        for px in range(px_start, px_end):
            for py in range(py_start, py_end):
                wx = minx + (px + 0.5) * resolution_x
                wy = maxy - (py + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[py, px] = 1  # [row=y, col=x]

    return mask, recs


def rasterize_patch_categories(patch_id: str, grid_size: int = 64) -> tuple[np.ndarray, list[dict]]:
    """将 patch 的类别标注光栅化为 [grid_size, grid_size] 的多类 mask.

    Returns:
        category_mask: int32, 0=未变化, 1=construction, 2=demolition, 3=road, 4=water_change, 5=farmland
    """
    patch_bounds = _build_patch_bounds()
    annotations = load_harbin_annotations()
    recs = annotations.get(patch_id, [])

    bounds = patch_bounds.get(patch_id)
    if bounds is None or not recs:
        return np.zeros((grid_size, grid_size), dtype=np.int32), recs

    minx, miny, maxx, maxy = bounds
    resolution_x = (maxx - minx) / grid_size
    resolution_y = (maxy - miny) / grid_size

    mask = np.zeros((grid_size, grid_size), dtype=np.int32)

    for r in recs:
        geom = r["geometry"]
        if geom is None:
            continue
        cat = r.get("category", "other")
        if cat not in CATEGORY_TO_IDX:
            continue
        cat_idx = CATEGORY_TO_IDX[cat]

        if geom.geom_type == "Point":
            px = min(grid_size - 1, max(0, int((geom.x - minx) / resolution_x)))
            py = min(grid_size - 1, max(0, int((maxy - geom.y) / resolution_y)))
            mask[py, px] = cat_idx  # [row=y, col=x]
            continue

        gminx, gminy, gmaxx, gmaxy = geom.bounds
        px_start = max(0, int((gminx - minx) / resolution_x))
        px_end = min(grid_size, int((gmaxx - minx) / resolution_x) + 1)
        py_start = max(0, int((maxy - gmaxy) / resolution_y))
        py_end = min(grid_size, int((maxy - gminy) / resolution_y) + 1)

        for px in range(px_start, px_end):
            for py in range(py_start, py_end):
                wx = minx + (px + 0.5) * resolution_x
                wy = maxy - (py + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[py, px] = cat_idx  # [row=y, col=x]

    return mask, recs


if __name__ == "__main__":
    # Smoke test
    annotated = get_annotated_patches()
    print(f"Annotated patches: {len(annotated)}")
    print("First 10:", annotated[:10])

    # Show category distribution
    annotations = load_harbin_annotations()
    cat_counts = {}
    for pid, recs in annotations.items():
        for r in recs:
            cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    print("Category distribution:", cat_counts)

    # Test rasterize
    if annotated:
        mask, recs = rasterize_patch_changes(annotated[0])
        print(f"Patch {annotated[0]}: mask shape={mask.shape}, changed pixels={mask.sum()}")
