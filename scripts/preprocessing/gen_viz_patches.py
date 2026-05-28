#!/usr/bin/env python3
"""
从 preprocessing/configs/*.json 读取区域配置，生成可视化用的 patches 数据。
输出到 preprocessing/viz/data/{region_name}_patches.json

用法:
    conda run -n xuannv python3 scripts/preprocessing/gen_viz_patches.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preprocessing.utils.geo import bbox_to_utm_patches


# SAR 点位 bbox（用于 haidian 地图标注）
HAIDIAN_SAR_SITES = {
    "点位1_干涉（海淀/昌平）": {
        "color": "#f59e0b",
        "bbox_wgs84": [116.0298, 39.9279, 116.3922, 40.1927],
    },
    "点位2_干涉（丰台/大兴）": {
        "color": "#10b981",
        "bbox_wgs84": [116.2723, 39.7036, 116.6509, 39.9832],
    },
    "朝阳角反": {
        "color": "#f43f5e",
        "bbox_wgs84": [116.2233, 39.8159, 116.6032, 40.1160],
    },
    "门头沟大台_干涉": {
        "color": "#8b5cf6",
        "bbox_wgs84": [115.7915, 39.7923, 116.1323, 40.0552],
    },
}

REGION_EXTRAS: dict[str, dict] = {
    "haidian": {
        "sar_sites": HAIDIAN_SAR_SITES,
    }
}


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def make_viz_data(cfg: dict) -> dict:
    region_name = cfg["region_name"]
    crs = cfg.get("crs", "EPSG:32652")
    patch_cfg = cfg["patch"]
    patch_size_m = patch_cfg["size_m"]
    step_m = patch_cfg.get("step_m", patch_size_m)
    utm_grid = patch_cfg.get("utm_grid")
    bbox = cfg["bbox"]

    patches, resolved_crs = bbox_to_utm_patches(
        bbox, patch_size_m, step_m=step_m,
        crs_override=crs, utm_grid=utm_grid
    )

    viz_patches = [
        {
            "id": p["id"],
            "bounds_wgs84": p["bounds_wgs84"],
            "center_lonlat": p["center_lonlat"],
        }
        for p in patches
    ]

    all_w = [p["bounds_wgs84"][0] for p in patches]
    all_s = [p["bounds_wgs84"][1] for p in patches]
    all_e = [p["bounds_wgs84"][2] for p in patches]
    all_n = [p["bounds_wgs84"][3] for p in patches]
    overall_bbox = [
        round(min(all_w), 6),
        round(min(all_s), 6),
        round(max(all_e), 6),
        round(max(all_n), 6),
    ]

    out: dict = {
        "region_name": region_name,
        "display_name": cfg.get("display_name", region_name),
        "crs": resolved_crs,
        "patch_size_m": patch_size_m,
        "total_patches": len(viz_patches),
        "overall_bbox_wgs84": overall_bbox,
        "patches": viz_patches,
    }

    extras = REGION_EXTRAS.get(region_name, {})
    out.update(extras)

    return out


def main() -> None:
    configs_dir = ROOT / "preprocessing" / "configs"
    out_dir = ROOT / "preprocessing" / "viz" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_files = sorted(configs_dir.glob("*.json"))
    if not config_files:
        print("未找到配置文件")
        return

    summary = []
    for cfg_path in config_files:
        try:
            cfg = load_config(cfg_path)
        except Exception as e:
            print(f"[SKIP] {cfg_path.name}: {e}")
            continue
        if "region_name" not in cfg or "bbox" not in cfg or "patch" not in cfg:
            print(f"[SKIP] {cfg_path.name}: 缺少必要字段")
            continue

        print(f"[{cfg['region_name']}] 正在生成...", end=" ", flush=True)
        viz_data = make_viz_data(cfg)

        out_path = out_dir / f"{cfg['region_name']}_patches.json"
        with open(out_path, "w") as f:
            json.dump(viz_data, f, ensure_ascii=False, separators=(",", ":"))

        print(f"{viz_data['total_patches']} patches → {out_path.relative_to(ROOT)}")
        summary.append({
            "region": viz_data["region_name"],
            "display": viz_data["display_name"],
            "patches": viz_data["total_patches"],
            "bbox": viz_data["overall_bbox_wgs84"],
        })

    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总索引 → {index_path.relative_to(ROOT)}")
    print(f"\n共 {len(summary)} 个区域:")
    for s in summary:
        print(f"  {s['display']:24s}  {s['patches']:5d} patches")


if __name__ == "__main__":
    main()
