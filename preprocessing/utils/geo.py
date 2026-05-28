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
) -> tuple[list[dict], str]:
    """
    将经纬度 bbox 转为 UTM 网格，返回 (patch_list, crs_str)。

    每个 patch 包含:
        id          : int
        utm_bounds  : [left, bottom, right, top]
        center_lonlat: [lon, lat]

    Args:
        bbox_deg    : {"west", "south", "east", "north"} 字典
        patch_size_m: patch 边长（米）
        step_m      : 步长，默认与 patch_size_m 相同（无重叠）
        crs_override: 若不为 None，强制使用指定 CRS（如 "EPSG:32650"）
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

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    _xs, _ys = to_utm.transform([w, e], [s, n])
    left, right = _xs[0], _xs[1]
    bottom, top = _ys[0], _ys[1]

    patches: list[dict] = []
    patch_id = 0
    x = left
    while x + patch_size_m <= right + 1e-3:
        y = bottom
        while y + patch_size_m <= top + 1e-3:
            cx = x + patch_size_m / 2
            cy = y + patch_size_m / 2
            lon, lat = to_wgs.transform(cx, cy)
            patches.append({
                "id": patch_id,
                "utm_bounds": [x, y, x + patch_size_m, y + patch_size_m],
                "center_lonlat": [lon, lat],
            })
            patch_id += 1
            y += step_m
        x += step_m

    return patches, crs
