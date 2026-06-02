#!/usr/bin/env python3
"""OlmoEarth 变化检测 AUC 评估。
用各月份全局 embedding (project_and_aggregate, 768D) 计算 patch-level cosine distance，
与哈尔滨变化标注 shp 做 ROC AUC。

时间段:
  june      : 2025-04 → 2025-06  shp: june.shp
  aug       : 2025-06 → 2025-08  shp: aug.shp
  September : 2025-08 → 2025-09  shp: September.shp
  October   : 2025-09 → 2025-10  shp: October.shp
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from sklearn.metrics import roc_auc_score

ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")
GRID      = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
EMB_ROOT  = Path("/workspace/outputs/olmoearth_harbin")
META_FILE = Path("/workspace/xuannv_show/data/harbin/patches_meta.json")
OUT_FILE  = Path("/workspace/outputs/olmoearth_harbin/eval/auc_olmoearth.json")

PERIODS = {
    "june":      (4,  6,  "june.shp"),
    "aug":       (6,  8,  "aug.shp"),
    "September": (8,  9,  "September.shp"),
    "October":   (9,  10, "October.shp"),
}


def load_month_emb(month: int) -> dict[str, np.ndarray]:
    """加载某月份的 patch embedding，返回 {patch_id: (768,)}。"""
    f = EMB_ROOT / f"{month:02d}" / "emb_all.npz"
    if not f.exists():
        raise FileNotFoundError(f"找不到 {f}，请先运行 olmoearth_embed_months.py")
    c = np.load(f)
    embs = c["embeddings"]      # (N, 768)
    ids  = list(c["patch_ids"]) # ['patch_000000', ...]
    return {pid: embs[i] for i, pid in enumerate(ids)}


def load_patch_bounds() -> dict[str, tuple]:
    """从 patches_meta.json 读取 patch 的 WGS84 bounds。"""
    meta = json.load(open(META_FILE))
    # bounds_wgs84: [lon_min, lat_min, lon_max, lat_max]
    return {p["patch_id"]: tuple(p["bounds_wgs84"]) for p in meta}


def patches_intersect_shp(shp_path: Path, patch_bounds: dict[str, tuple]) -> set[str]:
    """返回与变化多边形有交集的 patch_id 集合。"""
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"  读取 shp 失败: {e}")
        return set()
    if gdf.empty:
        return set()
    # 统一到 WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    changed = set()
    for pid, (lon0, lat0, lon1, lat1) in patch_bounds.items():
        patch_box = box(lon0, lat0, lon1, lat1)
        if gdf.intersects(patch_box).any():
            changed.add(pid)
    return changed


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - a @ b)


def evaluate_period(name: str, m_before: int, m_after: int, shp_name: str,
                    patch_bounds: dict) -> dict:
    print(f"\n  [加载] 月份 {m_before:02d} 和 {m_after:02d}")
    try:
        emb_before = load_month_emb(m_before)
        emb_after  = load_month_emb(m_after)
    except FileNotFoundError as e:
        print(f"  跳过: {e}")
        return None

    shp_path = ANNOT_DIR / shp_name
    if not shp_path.exists():
        print(f"  标注文件不存在: {shp_path}")
        return None

    changed_ids = patches_intersect_shp(shp_path, patch_bounds)
    print(f"  变化patch: {len(changed_ids)} / {len(patch_bounds)}")

    common = set(emb_before) & set(emb_after)
    distances, labels = [], []
    for pid in sorted(common):
        d = cosine_dist(emb_before[pid], emb_after[pid])
        distances.append(d)
        labels.append(1 if pid in changed_ids else 0)

    distances = np.array(distances)
    labels    = np.array(labels)
    n_changed = labels.sum()
    n_total   = len(labels)

    print(f"  共 {n_total} patches，变化={n_changed}，未变化={n_total-n_changed}")
    print(f"  变化patch余弦距离: {distances[labels==1].mean():.4f}")
    print(f"  未变化patch余弦距离: {distances[labels==0].mean():.4f}")

    if n_changed == 0 or n_changed == n_total:
        print(f"  ⚠️  类别单一，无法计算 AUC")
        return {"n_changed": int(n_changed), "n_total": int(n_total), "auc": None}

    auc = float(roc_auc_score(labels, distances))
    print(f"  AUC = {auc:.4f}")
    return {
        "n_changed": int(n_changed), "n_total": int(n_total), "auc": auc,
        "mean_dist_changed":   float(distances[labels==1].mean()),
        "mean_dist_unchanged": float(distances[labels==0].mean()),
    }


def main():
    print("=== OlmoEarth 变化检测 AUC 评估 ===")
    patch_bounds = load_patch_bounds()
    print(f"Patches: {len(patch_bounds)}")

    results = {}
    for name, (m_before, m_after, shp_name) in PERIODS.items():
        print(f"\n{'='*50}")
        print(f"时间段: {name}  ({m_before:02d}月 → {m_after:02d}月)  shp={shp_name}")
        r = evaluate_period(name, m_before, m_after, shp_name, patch_bounds)
        results[name] = r

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果保存: {OUT_FILE}")

    # 打印汇总
    print("\n=== 汇总 ===")
    for name, r in results.items():
        if r and r.get("auc") is not None:
            print(f"  {name:12s}: AUC={r['auc']:.4f}  变化={r['n_changed']}/{r['n_total']}")
        else:
            print(f"  {name:12s}: 无法计算")


if __name__ == "__main__":
    main()
