#!/usr/bin/env python3
"""重新生成三个城市的 patches_meta.json"""
import json
from pathlib import Path

CITIES = {
    "qiqihar": {
        "bbox_deg": [123.5, 47.0, 124.5, 47.7],
        "patch_size_m": 1280,
    },
    "daqing": {
        "bbox_deg": [124.5, 46.2, 125.6, 47.0],
        "patch_size_m": 1280,
    },
    "haidian": {
        "bbox_deg": [115.9, 39.7, 116.7, 40.2],
        "patch_size_m": 1280,
    },
}


def get_utm_crs(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def degree_bbox_to_utm_patches(bbox_deg, patch_size_m):
    from pyproj import Transformer
    w, s, e, n = bbox_deg
    center_lon = (w + e) / 2
    center_lat = (s + n) / 2
    crs = get_utm_crs(center_lon, center_lat)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    xs, ys = transformer.transform([w, e], [s, n])
    left, right = xs
    bottom, top = ys

    patches = []
    x = left
    patch_id = 0
    while x + patch_size_m <= right + 1e-6:
        y = bottom
        while y + patch_size_m <= top + 1e-6:
            patch_bounds = [x, y, x + patch_size_m, y + patch_size_m]
            cx = x + patch_size_m / 2
            cy = y + patch_size_m / 2
            lon, lat = transformer.transform(cx, cy, direction="INVERSE")
            patches.append({
                "id": patch_id,
                "utm_bounds": patch_bounds,
                "center_lonlat": [lon, lat],
            })
            patch_id += 1
            y += patch_size_m
        x += patch_size_m

    return patches, crs


def main():
    base = Path("/workspace/raw/phase2_heilongjiang")
    base.mkdir(parents=True, exist_ok=True)

    for city, cfg in CITIES.items():
        patches, crs = degree_bbox_to_utm_patches(cfg["bbox_deg"], cfg["patch_size_m"])
        # 对齐之前的数据规模：每个城市限制 400 patches
        patches = patches[:400]
        meta = {
            "city": city,
            "crs": crs,
            "patch_size_m": cfg["patch_size_m"],
            "patches": patches,
        }
        out_dir = base / city
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "patches_meta.json"
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{city}] 生成 {len(patches)} 个 patches, CRS={crs}, 保存到 {out_path}")


if __name__ == "__main__":
    main()
