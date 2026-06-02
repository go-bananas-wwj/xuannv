#!/usr/bin/env python3
"""下载哈尔滨 424 patch 的 12 波段 Sentinel-2 L2A（OlmoEarth 教师输入用）。

策略：复刻现有 cloud_filtered/s2 中的 (patch, 日期) 场景，仅把波段从 6 补到 12，
      保证与玄女训练样本时空完全对齐。输出到新目录，不动现有数据。

波段顺序严格对齐 OlmoEarth SENTINEL2_L2A:
    B02 B03 B04 B08 | B05 B06 B07 B8A B11 B12 | B01 B09   (共 12)

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

import argparse, json, traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pyproj import Transformer

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from preprocessing.utils.tiff import write_tif

# OlmoEarth SENTINEL2_L2A 波段顺序（PC 资产名）
OLMO_S2_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]

PATCHES_META = "/workspace/raw/harbin_scenes/patches_meta.json"
EXIST_S2_DIR = Path("/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2")
OUT_DIR      = Path("/workspace/raw/phase1_harbin/harbin_olmoearth/s2")
CRS = "EPSG:32652"
IMG = 128
RES = 10


def utm_to_wgs84(left, bottom, right, top):
    t = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs, ys = t.transform([left, right], [bottom, top])
    return [xs[0], ys[0], xs[1], ys[1]]


def existing_dates(pid: int) -> set[str]:
    d = EXIST_S2_DIR / f"patch_{pid:06d}"
    if not d.exists():
        d = EXIST_S2_DIR / f"patch_{pid:04d}"
    return {f.stem for f in d.glob("*.tif")} if d.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 patch（测试）")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_root = Path(args.out)

    import planetary_computer, pystac_client, odc.stac  # noqa
    from odc.geo.geobox import GeoBox
    from odc.geo import BoundingBox
    from skimage.transform import resize as sk_resize

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    meta = json.load(open(PATCHES_META))
    patches = meta["patches"]
    if args.limit:
        patches = patches[: args.limit]
    tr = meta.get("date_start", "2023-01-01"), meta.get("date_end", "2025-10-31")
    date_filter = f"{tr[0]}/{tr[1]}"
    print(f"[init] {len(patches)} patches, 时间 {date_filter}, 输出 {out_root}")

    def do_patch(patch: dict):
        pid = patch["id"]
        want = existing_dates(pid)
        if not want:
            return pid, 0, 0, 0
        left, bottom, right, top = patch["utm_bounds"]
        out_dir = out_root / f"patch_{pid:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        need = {d for d in want if not (out_dir / f"{d}.tif").exists()}
        if not need:
            return pid, len(want), 0, 0
        geobox = GeoBox.from_bbox(BoundingBox(left, bottom, right, top, crs=CRS), resolution=RES)
        ok = fail = 0
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=utm_to_wgs84(left, bottom, right, top),
                datetime=date_filter,
            )
            items = list(search.items())
            # date_tag -> item（同日多景取第一个）
            by_date = {}
            for it in items:
                dt = it.properties.get("datetime", "")
                if not dt:
                    continue
                tag = datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y%m%d")
                by_date.setdefault(tag, it)
            for tag in sorted(need):
                it = by_date.get(tag)
                if it is None:
                    fail += 1
                    continue
                try:
                    ds = odc.stac.load([it], bands=OLMO_S2_BANDS, geobox=geobox,
                                       chunks={}).compute()
                    arr = np.stack([ds[b].values.squeeze() for b in OLMO_S2_BANDS], axis=0)
                    if arr.shape[1] != IMG or arr.shape[2] != IMG:
                        arr = np.stack([sk_resize(arr[c], (IMG, IMG), anti_aliasing=True,
                                                  preserve_range=True) for c in range(arr.shape[0])], axis=0)
                    write_tif(out_dir / f"{tag}.tif", arr.astype(np.float32), crs=CRS,
                              bounds=[left, bottom, right, top])
                    ok += 1
                except Exception:
                    fail += 1
        except Exception:
            print(f"[patch {pid}] 搜索失败: {traceback.format_exc()[-200:]}")
        return pid, len(want) - len(need), ok, fail

    total_ok = total_fail = total_skip = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_patch, p): p for p in patches}
        for fut in as_completed(futs):
            pid, skip, ok, fail = fut.result()
            total_ok += ok; total_fail += fail; total_skip += skip
            done += 1
            if done % 10 == 0 or done == len(patches):
                print(f"[{done}/{len(patches)}] 累计: 新下载 {total_ok}, 跳过(已存在) {total_skip}, 失败 {total_fail}")
    print(f"[done] 新下载 {total_ok}, 跳过 {total_skip}, 失败 {total_fail}")


if __name__ == "__main__":
    main()
