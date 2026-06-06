#!/usr/bin/env python3
"""下载哈尔滨 424 patch 的 12 波段 Sentinel-2 L2A（OlmoEarth 教师输入用）。

设计目标：与玄女现有 6 波段训练数据 **像素级对齐**，仅把波段补到 12。
做法：复刻现有 cloud_filtered/s2 中每个 (patch, 日期) 场景，
      用现有 6 波段 tif 的 GeoBox（128×128, 精确 transform）下载同网格 12 波段。

波段顺序严格对齐 OlmoEarth SENTINEL2_L2A:
    B02 B03 B04 B08 | B05 B06 B07 B8A B11 B12 | B01 B09   (共 12)
存储：uint16 raw DN（与现有 6 波段一致，OlmoEarth Normalizer 期望 raw DN），lzw 压缩。

借鉴 /workspace/run_download.py 的优化:
  - 每个 item 下载前 planetary_computer.sign + token 过期自动重签重试
  - 搜索失败退避重试
  - 全零 / 波段数校验
  - lzw 压缩

用法:
    conda run -n xuannv python scripts/distill/download_s2_olmoearth.py --limit 2   # 测试
    conda run -n xuannv python scripts/distill/download_s2_olmoearth.py --workers 8 # 全量
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(line_buffering=True)
for v in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import dask
dask.config.set(scheduler="synchronous")

import argparse, json, time, traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from pyproj import Transformer

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

# OlmoEarth SENTINEL2_L2A 波段顺序（PC 资产名）
OLMO_S2_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]

PATCHES_META = "/workspace/xuannv/data_raw/harbin/scenes/patches_meta.json"
EXIST_S2_DIR = Path("/workspace/xuannv/data_raw/harbin/scenes/s2")
OUT_DIR      = Path("/workspace/xuannv/data_raw/phase1_harbin/harbin_olmoearth/s2")
CRS = "EPSG:32652"
RES = 10
MAX_DN = 20000   # S2 L2A 合理 DN 上限（防溢出裁剪）


def _exist_patch_dir(pid: int) -> Path | None:
    for name in (f"patch_{pid:06d}", f"patch_{pid:04d}"):
        d = EXIST_S2_DIR / name
        if d.exists():
            return d
    return None


def utm_to_wgs84_bbox(left, bottom, right, top):
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs, ys = t.transform([left, right], [bottom, top])
    return [xs[0], ys[0], xs[1], ys[1]]


def search_items_with_retry(catalog, bbox_ll, date_filter, retries=3):
    """搜索 STAC items，失败退避重试（借鉴 run_download.py）。"""
    for attempt in range(retries):
        try:
            return list(catalog.search(
                collections=["sentinel-2-l2a"], bbox=bbox_ll, datetime=date_filter,
            ).items())
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 + attempt * 5
                print(f"    [WARN] 搜索失败(尝试{attempt+1}/{retries}): {e}，{wait}s 后重试")
                time.sleep(wait)
            else:
                raise


def load_item_with_retry(item, geobox, retries=3):
    """下载单 item 的 12 波段，自动处理 token 过期并重签名（借鉴 run_download.py）。"""
    import odc.stac
    import planetary_computer as pc
    for attempt in range(retries):
        try:
            signed = pc.sign(item)
            ds = odc.stac.load([signed], bands=OLMO_S2_BANDS, geobox=geobox, chunks={}).compute()
            arr = np.stack([ds[b].values.squeeze() for b in OLMO_S2_BANDS], axis=0)  # (12,H,W)
            if arr.shape[0] != len(OLMO_S2_BANDS):
                raise ValueError(f"波段数不符: 期望 {len(OLMO_S2_BANDS)}, 实得 {arr.shape[0]}")
            if not np.any(arr):
                raise ValueError("全零数据")
            arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, MAX_DN).astype(np.uint16)
            return arr
        except Exception as e:
            err = str(e).lower()
            is_token = any(x in err for x in ["403", "401", "token", "expired", "unauthorized"])
            if is_token and attempt < retries - 1:
                wait = 2 + attempt * 3
                print(f"      [RETRY] token 过期，{wait}s 后重签名重试({attempt+1}/{retries}): {item.id}")
                time.sleep(wait)
            else:
                raise


def save_uint16_lzw(path: Path, arr: np.ndarray, transform, crs: str):
    """保存 (C,H,W) uint16 GeoTIFF，lzw 压缩（借鉴 run_download.py）。"""
    c, h, w = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=c,
        dtype="uint16", crs=crs, transform=transform, compress="lzw",
    ) as dst:
        dst.write(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 patch（测试）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_root = Path(args.out)

    import planetary_computer, pystac_client  # noqa
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import CRS as OdcCRS

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    meta = json.load(open(PATCHES_META))
    patches = meta["patches"]
    if args.limit:
        patches = patches[: args.limit]
    date_filter = f"{meta.get('date_start','2023-01-01')}/{meta.get('date_end','2025-10-31')}"
    print(f"[init] {len(patches)} patches, 时间 {date_filter}, 输出 {out_root}")

    def do_patch(patch: dict):
        pid = patch["id"]
        ex_dir = _exist_patch_dir(pid)
        if ex_dir is None:
            return pid, 0, 0, 0
        ref_tifs = sorted(ex_dir.glob("*.tif"))
        if not ref_tifs:
            return pid, 0, 0, 0
        # 用任一现有 tif 构建精确 GeoBox（同 patch 所有日期 bounds 相同）
        with rasterio.open(ref_tifs[0]) as ds0:
            transform = ds0.transform
            h, w = ds0.height, ds0.width
            geobox = GeoBox((h, w), transform, OdcCRS(CRS))

        want = {t.stem for t in ref_tifs}
        out_dir = out_root / f"patch_{pid:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        need = sorted(d for d in want if not (out_dir / f"{d}.tif").exists())
        skip = len(want) - len(need)
        if not need:
            return pid, skip, 0, 0

        left, bottom, right, top = patch["utm_bounds"]
        ok = fail = 0
        try:
            items = search_items_with_retry(catalog, utm_to_wgs84_bbox(left, bottom, right, top), date_filter)
            by_date = {}
            for it in items:
                dt = it.properties.get("datetime", "")
                if not dt:
                    continue
                tag = datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y%m%d")
                by_date.setdefault(tag, it)   # 同日多景取第一个
            for tag in need:
                it = by_date.get(tag)
                if it is None:
                    fail += 1
                    continue
                try:
                    arr = load_item_with_retry(it, geobox)
                    save_uint16_lzw(out_dir / f"{tag}.tif", arr, transform, CRS)
                    ok += 1
                except Exception as e:
                    print(f"    [WARN] patch_{pid:06d} {tag}: {str(e)[:120]}")
                    fail += 1
        except Exception:
            print(f"[patch {pid}] 搜索失败: {traceback.format_exc()[-200:]}")
        return pid, skip, ok, fail

    total_ok = total_fail = total_skip = done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_patch, p): p for p in patches}
        for fut in as_completed(futs):
            _, skip, ok, fail = fut.result()
            total_ok += ok; total_fail += fail; total_skip += skip; done += 1
            if done % 10 == 0 or done == len(patches):
                print(f"[{done}/{len(patches)}] 累计: 新下载 {total_ok}, 跳过 {total_skip}, 失败 {total_fail}")
    print(f"[done] 新下载 {total_ok}, 跳过 {total_skip}, 失败 {total_fail}")


if __name__ == "__main__":
    main()
