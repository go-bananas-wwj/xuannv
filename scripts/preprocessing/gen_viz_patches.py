#!/usr/bin/env python3
"""
从 preprocessing/configs/*.json 读取区域配置，生成可视化用的 patches JSON。

优先级：
  1. config 里有 patches_meta_path  → 直接读取真实训练 patch 数据
  2. config 里有 patch.utm_grid     → 从精确 UTM 网格坐标生成
  3. 兜底                            → bbox + CRS 浮点推算

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
    "haidian": {"sar_sites": HAIDIAN_SAR_SITES},
}


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return json.load(f)


def utm_bounds_to_wgs84(utm_bounds: list, crs: str) -> list[float]:
    """[left, bottom, right, top] UTM → [west, south, east, north] WGS84."""
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    pw, ps = to_wgs.transform(utm_bounds[0], utm_bounds[1])
    pe, pn = to_wgs.transform(utm_bounds[2], utm_bounds[3])
    return [round(pw, 6), round(ps, 6), round(pe, 6), round(pn, 6)]


def center_from_wgs84(bwgs84: list[float]) -> list[float]:
    cx = round((bwgs84[0] + bwgs84[2]) / 2, 6)
    cy = round((bwgs84[1] + bwgs84[3]) / 2, 6)
    return [cx, cy]


def read_patches_meta(meta_path: Path, crs: str) -> list[dict]:
    """读取真实 patches_meta.json，统一转换为可视化格式。

    兼容两种格式:
      - list of {patch_id, bounds/utm_bounds, bounds_wgs84?, center_lonlat, ...}
      - dict {city, patches: [...{id, utm_bounds, center_lonlat}]}
    """
    raw = json.loads(meta_path.read_text())

    # 展开 dict 包装
    if isinstance(raw, dict):
        raw_patches = raw.get("patches", [])
        meta_crs = raw.get("crs", crs)
    else:
        raw_patches = raw
        meta_crs = crs

    result = []
    for i, p in enumerate(raw_patches):
        pid = p.get("id", p.get("patch_id", i))

        # 获取 UTM bounds
        utm = p.get("utm_bounds") or p.get("bounds")

        # 如果已经有 bounds_wgs84 直接用, 否则转换
        bwgs84 = p.get("bounds_wgs84")
        if not bwgs84:
            if utm:
                bwgs84 = utm_bounds_to_wgs84(utm, meta_crs)
            else:
                continue  # 无坐标信息，跳过

        # 中心点
        center = p.get("center_lonlat")
        if not center:
            center = center_from_wgs84(bwgs84)

        result.append({
            "id": pid,
            "bounds_wgs84": [round(v, 6) for v in bwgs84],
            "center_lonlat": [round(v, 6) for v in center],
        })

    return result


def make_viz_data_from_meta(cfg: dict, meta_path: Path) -> dict:
    """从真实 patches_meta.json 生成可视化数据。"""
    region_name = cfg["region_name"]
    crs = cfg.get("crs", "EPSG:32652")

    print(f"  ↳ 读取真实 patches: {meta_path}", end=" ", flush=True)
    viz_patches = read_patches_meta(meta_path, crs)

    all_w = [p["bounds_wgs84"][0] for p in viz_patches]
    all_s = [p["bounds_wgs84"][1] for p in viz_patches]
    all_e = [p["bounds_wgs84"][2] for p in viz_patches]
    all_n = [p["bounds_wgs84"][3] for p in viz_patches]
    overall_bbox = [
        round(min(all_w), 6), round(min(all_s), 6),
        round(max(all_e), 6), round(max(all_n), 6),
    ]

    out: dict = {
        "region_name": region_name,
        "display_name": cfg.get("display_name", region_name),
        "crs": crs,
        "patch_size_m": cfg.get("patch", {}).get("size_m", 1280),
        "total_patches": len(viz_patches),
        "overall_bbox_wgs84": overall_bbox,
        "source": "patches_meta",
        "patches": viz_patches,
    }
    out.update(REGION_EXTRAS.get(region_name, {}))
    return out


def make_viz_data_from_config(cfg: dict) -> dict:
    """从 bbox/utm_grid 配置生成可视化数据（无真实patch文件时）。"""
    region_name = cfg["region_name"]
    crs = cfg.get("crs", "EPSG:32652")
    patch_cfg = cfg["patch"]
    patch_size_m = patch_cfg["size_m"]
    step_m = patch_cfg.get("step_m", patch_size_m)
    utm_grid = patch_cfg.get("utm_grid")

    patches, resolved_crs = bbox_to_utm_patches(
        cfg["bbox"], patch_size_m, step_m=step_m,
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
        round(min(all_w), 6), round(min(all_s), 6),
        round(max(all_e), 6), round(max(all_n), 6),
    ]

    out: dict = {
        "region_name": region_name,
        "display_name": cfg.get("display_name", region_name),
        "crs": resolved_crs,
        "patch_size_m": patch_size_m,
        "total_patches": len(viz_patches),
        "overall_bbox_wgs84": overall_bbox,
        "source": "config_generated",
        "patches": viz_patches,
    }
    out.update(REGION_EXTRAS.get(region_name, {}))
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

        region_name = cfg["region_name"]
        print(f"[{region_name}] 正在生成...", end=" ", flush=True)

        # 优先使用真实 patches_meta 文件
        meta_path_str = cfg.get("patches_meta_path")
        if meta_path_str:
            meta_path = Path(meta_path_str)
            if meta_path.exists():
                viz_data = make_viz_data_from_meta(cfg, meta_path)
            else:
                print(f"\n  ⚠ patches_meta_path 不存在: {meta_path}，回退到 config 生成")
                viz_data = make_viz_data_from_config(cfg)
        else:
            print(" (config_generated)", end=" ", flush=True)
            viz_data = make_viz_data_from_config(cfg)

        out_path = out_dir / f"{region_name}_patches.json"
        with open(out_path, "w") as f:
            json.dump(viz_data, f, ensure_ascii=False, separators=(",", ":"))

        src_tag = "✓ 真实数据" if viz_data.get("source") == "patches_meta" else "⚙ config生成"
        print(f" [{src_tag}]  {viz_data['total_patches']} patches → {out_path.relative_to(ROOT)}")
        summary.append({
            "region": viz_data["region_name"],
            "display": viz_data["display_name"],
            "patches": viz_data["total_patches"],
            "bbox": viz_data["overall_bbox_wgs84"],
            "source": viz_data.get("source", "unknown"),
        })

    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总索引 → {index_path.relative_to(ROOT)}")
    print(f"\n{'区域':<28} {'Patches':>7}  {'来源'}")
    print("-" * 55)
    for s in summary:
        tag = "✓真实训练数据" if s["source"] == "patches_meta" else "⚙格网推算"
        print(f"  {s['display']:<26} {s['patches']:>7}  {tag}")


if __name__ == "__main__":
    main()
