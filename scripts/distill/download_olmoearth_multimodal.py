#!/usr/bin/env python3
"""下载哈尔滨新区 424 patch 的 OlmoEarth 5 模态输入数据。

区域: xuannv_show/data/harbin/patches_meta.json 的 424 patch (新区, EPSG:32652, 1280m/128px)
时间: 2025-01-01 ~ 今天 (时序模态); 静态模态取最新可用
模态(全部 Planetary Computer):
  - s2       : sentinel-2-l2a   12 波段 [B02 B03 B04 B08 | B05 B06 B07 B8A B11 B12 | B01 B09]
  - s1       : sentinel-1-rtc    2 波段 [vv vh]
  - landsat  : landsat-c2-l2    11 波段(OlmoEarth顺序 B8 B1..B11); L2 SR 仅 8 个可用,其余填0
  - worldcover: esa-worldcover    1 波段 (静态)
  - dem      : cop-dem-glo-30     1 波段 (静态, OlmoEarth 的 srtm)
输出: /workspace/raw/harbin_newarea_olmoearth/<模态>/patch_XXXXXX/<日期或static>.tif
      uint16(光学/分类) 或 float32(SAR/DEM), lzw 压缩, 与现有数据隔离。

借鉴 run_download.py: token 过期重签重试 / 搜索退避重试 / 全零校验。
时序模态按月限额(默认每月最多 max_per_month 景, S2 额外按 eo:cloud_cover 过滤)。

用法:
  conda run -n xuannv python scripts/distill/download_olmoearth_multimodal.py --limit 2 --sources s2 s1
  conda run -n xuannv python scripts/distill/download_olmoearth_multimodal.py --workers 8
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(line_buffering=True)
# 不走代理梯子: Planetary Computer (Azure) 国内直连可达, 避免消耗 VPN 流量。
for _p in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
    os.environ.pop(_p, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"
for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import dask
dask.config.set(scheduler="synchronous")

import argparse, json, time, traceback
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer

PATCHES_META = "/workspace/xuannv_show/data/harbin/patches_meta.json"
OUT_ROOT     = Path("/workspace/raw/harbin_newarea_olmoearth")
CRS = "EPSG:32652"
DATE_START = "2025-01-01"
DATE_END   = date.today().isoformat()   # 2026-06-02

# ── 模态配置 ──────────────────────────────────────────────────────────────
# kind: temporal=时序逐景; static=单景
# bands: 下载并按此顺序输出的波段名; None 表示用 fill 占位(L2缺失波段)
S2_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
# OlmoEarth Landsat 顺序 vs L2 SR 可用资产; None=L2无,填0
LANDSAT_MAP = [
    ("B8",  None),       # 全色 - L2 无
    ("B1",  "coastal"),
    ("B2",  "blue"),
    ("B3",  "green"),
    ("B4",  "red"),
    ("B5",  "nir08"),
    ("B6",  "swir16"),
    ("B7",  "swir22"),
    ("B9",  None),       # 卷云 - L2 无
    ("B10", "lwir11"),
    ("B11", None),       # 热红外2 - L2 无
]
MODALITIES = {
    "s2": dict(kind="temporal", collection="sentinel-2-l2a", res=10, dtype="uint16",
               assets=S2_BANDS, cloud_query=True, max_per_month=2),
    "s1": dict(kind="temporal", collection="sentinel-1-rtc", res=10, dtype="float32",
               assets=["vv","vh"], cloud_query=False, max_per_month=2),
    "landsat": dict(kind="temporal", collection="landsat-c2-l2", res=30, dtype="uint16",
                    assets=[a for _,a in LANDSAT_MAP if a], landsat_map=LANDSAT_MAP,
                    cloud_query=True, max_per_month=2),
    "worldcover": dict(kind="static", collection="esa-worldcover", res=10, dtype="uint16",
                       assets=["map"]),
    "dem": dict(kind="static", collection="cop-dem-glo-30", res=30, dtype="float32",
                assets=["data"]),
}


def utm_to_wgs84(left, bottom, right, top):
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs, ys = t.transform([left, right], [bottom, top])
    return [xs[0], ys[0], xs[1], ys[1]]


def search_with_retry(catalog, collection, bbox_ll, dt, query=None, retries=3):
    for attempt in range(retries):
        try:
            kw = dict(collections=[collection], bbox=bbox_ll)
            if dt:
                kw["datetime"] = dt
            if query:
                kw["query"] = query
            return list(catalog.search(**kw).items())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 + attempt * 5)
            else:
                raise


def load_with_retry(item, assets, geobox, retries=3):
    import odc.stac
    import planetary_computer as pc
    for attempt in range(retries):
        try:
            signed = pc.sign(item)
            ds = odc.stac.load([signed], bands=assets, geobox=geobox, chunks={}).compute()
            arr = np.stack([ds[b].values.squeeze() for b in assets], axis=0).astype(np.float32)
            if not np.any(arr):
                raise ValueError("全零数据")
            return arr
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["403","401","token","expired","unauthorized"]) and attempt < retries - 1:
                time.sleep(2 + attempt * 3)
            else:
                raise


def save_tif(path: Path, arr: np.ndarray, transform, dtype: str):
    c, h, w = arr.shape
    if dtype == "uint16":
        arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, 65535).astype(np.uint16)
    else:
        arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=c,
                       dtype=dtype, crs=CRS, transform=transform, compress="lzw") as dst:
        dst.write(arr)


def build_geobox(left, bottom, right, top, res):
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import CRS as OdcCRS
    from odc.geo import BoundingBox
    return GeoBox.from_bbox(BoundingBox(left, bottom, right, top, crs=CRS), resolution=res)


def month_tag(dt_iso: str) -> tuple[str, str]:
    d = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    return d.strftime("%Y%m%d"), d.strftime("%Y%m")


def download_modality_for_patch(catalog, src: str, cfg: dict, patch: dict):
    pid = patch["patch_id"] if isinstance(patch.get("patch_id"), str) else f"patch_{patch['id']:06d}"
    left, bottom, right, top = patch["bounds"] if "bounds" in patch else patch["utm_bounds"]
    bbox_ll = utm_to_wgs84(left, bottom, right, top)
    geobox = build_geobox(left, bottom, right, top, cfg["res"])
    transform = from_bounds(left, bottom, right, top, geobox.shape[1], geobox.shape[0])
    out_dir = OUT_ROOT / src / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = skip = 0

    if cfg["kind"] == "static":
        dst = out_dir / "static.tif"
        if dst.exists():
            return 0, 0, 1
        try:
            items = search_with_retry(catalog, cfg["collection"], bbox_ll, None)
            if not items:
                return 0, 1, 0
            arr = load_with_retry(items[0], cfg["assets"], geobox)
            save_tif(dst, arr, transform, cfg["dtype"])
            return 1, 0, 0
        except Exception as e:
            print(f"    [WARN] {src} {pid}: {str(e)[:100]}")
            return 0, 1, 0

    # temporal
    query = {"eo:cloud_cover": {"lt": 30}} if cfg.get("cloud_query") else None
    dt = f"{DATE_START}/{DATE_END}"
    try:
        items = search_with_retry(catalog, cfg["collection"], bbox_ll, dt, query)
    except Exception:
        print(f"[{src} {pid}] 搜索失败: {traceback.format_exc()[-150:]}")
        return 0, 0, 0
    # 按月分组限额
    by_month = defaultdict(list)
    for it in items:
        dts = it.properties.get("datetime") or it.properties.get("start_datetime")
        if not dts:
            continue
        day, ym = month_tag(dts)
        by_month[ym].append((day, it))
    selected = []
    for ym, lst in by_month.items():
        lst.sort(key=lambda x: x[0])
        selected.extend(lst[: cfg.get("max_per_month", 2)])

    landsat_map = cfg.get("landsat_map")
    for day, it in selected:
        dst = out_dir / f"{day}.tif"
        if dst.exists():
            skip += 1
            continue
        try:
            arr = load_with_retry(it, cfg["assets"], geobox)  # (n_avail, H, W)
            if landsat_map:
                # 重排为 OlmoEarth 11 波段顺序, 缺失填 0
                avail = {a: i for i, a in enumerate(cfg["assets"])}
                h, w = arr.shape[1], arr.shape[2]
                full = np.zeros((len(landsat_map), h, w), dtype=np.float32)
                for j, (_, a) in enumerate(landsat_map):
                    if a is not None:
                        full[j] = arr[avail[a]]
                arr = full
            save_tif(dst, arr, transform, cfg["dtype"])
            ok += 1
        except Exception as e:
            print(f"    [WARN] {src} {pid} {day}: {str(e)[:100]}")
            fail += 1
    return ok, fail, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(MODALITIES.keys()),
                    choices=list(MODALITIES.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import planetary_computer, pystac_client
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    meta = json.load(open(PATCHES_META))
    patches = meta if isinstance(meta, list) else meta["patches"]
    if args.limit:
        patches = patches[: args.limit]
    print(f"[init] {len(patches)} patches, 模态={args.sources}, 时间 {DATE_START}~{DATE_END}")
    print(f"[init] 输出根目录 {OUT_ROOT}")

    for src in args.sources:
        cfg = MODALITIES[src]
        print(f"\n===== 模态 {src} ({cfg['collection']}, {cfg['kind']}) =====")
        t_ok = t_fail = t_skip = done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download_modality_for_patch, catalog, src, cfg, p): p for p in patches}
            for fut in as_completed(futs):
                ok, fail, skip = fut.result()
                t_ok += ok; t_fail += fail; t_skip += skip; done += 1
                if done % 20 == 0 or done == len(patches):
                    print(f"  [{src} {done}/{len(patches)}] 新下载 {t_ok}, 跳过 {t_skip}, 失败 {t_fail}")
        print(f"  [{src} 完成] 新下载 {t_ok}, 跳过 {t_skip}, 失败 {t_fail}")


if __name__ == "__main__":
    main()
