"""地理工具：UTM 转换、patch 网格生成。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def get_utm_epsg(lon: float, lat: float) -> int:
    """根据经纬度返回合适的 UTM EPSG 代码。"""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def bbox_to_utm_patches(
    bbox_deg: dict,
    patch_size_m: float,
    step_m: float | None = None,
    crs_override: str | None = None,
    utm_grid: dict | None = None,
) -> tuple[list[dict], str]:
    """
    将经纬度 bbox 转为 UTM 网格，返回 (patch_list, crs_str)。

    每个 patch 包含:
        id           : int
        utm_bounds   : [left, bottom, right, top]
        bounds_wgs84 : [west, south, east, north]  -- WGS84 度
        center_lonlat: [lon, lat]

    Args:
        bbox_deg    : {"west", "south", "east", "north"} 字典
        patch_size_m: patch 边长（米）
        step_m      : 步长，默认与 patch_size_m 相同（无重叠）
        crs_override: 若不为 None，强制使用指定 CRS（如 "EPSG:32650"）
        utm_grid    : 精确 UTM 网格定义，优先级高于 bbox_deg。
                      格式: {"origin_x": float, "origin_y": float,
                             "cols": int, "rows": int}
                      直接从 UTM 整数坐标生成网格，避免浮点往返误差。
    """
    from pyproj import Transformer

    step_m = step_m or patch_size_m
    w, s, e, n = bbox_deg["west"], bbox_deg["south"], bbox_deg["east"], bbox_deg["north"]
    center_lon = (w + e) / 2
    center_lat = (s + n) / 2

    if crs_override:
        crs = crs_override
    else:
        epsg = get_utm_epsg(center_lon, center_lat)
        crs = f"EPSG:{epsg}"

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    if utm_grid is not None:
        # 使用精确 UTM 网格
        ox = float(utm_grid["origin_x"])
        oy = float(utm_grid["origin_y"])
        cols = int(utm_grid["cols"])
        rows = int(utm_grid["rows"])
        xs_iter = [ox + c * step_m for c in range(cols)]
        ys_iter = [oy + r * step_m for r in range(rows)]
    else:
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        _xs, _ys = to_utm.transform([w, e], [s, n])
        left, right = _xs[0], _xs[1]
        bottom, top = _ys[0], _ys[1]
        xs_iter_raw, ys_iter_raw = [], []
        x = left
        while x + patch_size_m <= right + 1e-3:
            xs_iter_raw.append(x)
            x += step_m
        y = bottom
        while y + patch_size_m <= top + 1e-3:
            ys_iter_raw.append(y)
            y += step_m
        xs_iter = xs_iter_raw
        ys_iter = ys_iter_raw

    patches: list[dict] = []
    patch_id = 0
    for x in xs_iter:
        for y in ys_iter:
            cx = x + patch_size_m / 2
            cy = y + patch_size_m / 2
            lon, lat = to_wgs.transform(cx, cy)
            pw, ps = to_wgs.transform(x, y)
            pe, pn = to_wgs.transform(x + patch_size_m, y + patch_size_m)
            patches.append({
                "id": patch_id,
                "utm_bounds": [x, y, x + patch_size_m, y + patch_size_m],
                "bounds_wgs84": [round(pw, 6), round(ps, 6), round(pe, 6), round(pn, 6)],
                "center_lonlat": [round(lon, 6), round(lat, 6)],
            })
            patch_id += 1

    return patches, crs
