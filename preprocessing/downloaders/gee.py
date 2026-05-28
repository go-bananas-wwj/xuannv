"""Google Earth Engine 下载器。

依赖:
    pip install earthengine-api
    # 初始化方式二选一：
    #   (A) 服务账号: 设置环境变量 GEE_CREDENTIALS_PATH + GEE_SERVICE_ACCOUNT
    #   (B) 个人认证: earthengine authenticate
"""
from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from preprocessing.downloaders.base import BaseDownloader
from preprocessing.utils.logging import get_logger
from preprocessing.utils.tiff import write_tif

logger = get_logger(__name__)

# GEE collection → 导出配置映射
_COLLECTION_CFG: dict[str, dict] = {
    "COPERNICUS/S2_SR_HARMONIZED": {
        "date_field": "system:time_start",
        "max_cloud": "CLOUDY_PIXEL_PERCENTAGE",
    },
    "COPERNICUS/S1_GRD": {
        "date_field": "system:time_start",
        "filter_mode": "IW",
    },
    "LANDSAT/LC08/C02/T1_L2": {
        "date_field": "system:time_start",
        "max_cloud": "CLOUD_COVER",
    },
    "NASA/NASADEM_HGT/001": {"static": True},
    "ESA/WorldCover/v200": {"static": True},
    "GOOGLE/DYNAMICWORLD/V1": {
        "date_field": "system:time_start",
    },
    "JRC/GSW1_4/MonthlyHistory": {
        "date_field": "system:time_start",
    },
}


class GEEDownloader(BaseDownloader):
    """
    通过 GEE Python API 下载影像并切 patch。

    NOTE: GEE Export API 有配额限制，且任务异步执行。
    本实现使用 getPixels / getMap 方式直接下载小区域，
    适合 patch 级别（≤ 1.28km×1.28km）的数据获取。
    """

    def __init__(self, region_cfg: dict, source_name: str) -> None:
        super().__init__(region_cfg, source_name)
        self._ee_initialized = False

    def _init_ee(self) -> None:
        if self._ee_initialized:
            return
        import ee
        creds = self.region_cfg.get("credentials", {}).get("gee", {})
        cred_path = os.environ.get(creds.get("credentials_path_env", "GEE_CREDENTIALS_PATH"), "")
        svc_acct = os.environ.get(creds.get("service_account_env", "GEE_SERVICE_ACCOUNT"), "")
        if cred_path and svc_acct and Path(cred_path).exists():
            credentials = ee.ServiceAccountCredentials(svc_acct, cred_path)
            ee.Initialize(credentials)
            logger.info("[GEE] 已通过服务账号初始化")
        else:
            ee.Initialize()
            logger.info("[GEE] 已通过个人认证初始化")
        self._ee_initialized = True

    # ------------------------------------------------------------------ #

    def download(self, patches: list[dict], *, workers: int = 4) -> dict[str, int]:
        self._init_ee()
        import ee

        collection_id = self.source_cfg["collection"]
        bands = self.source_cfg["bands"]
        time_range = self.region_cfg["time_range"]
        crs = self.region_cfg.get("crs", "EPSG:32650")
        image_size = self.region_cfg["patch"]["image_size_px"]
        resolution_m = self.source_cfg.get("resolution_m", 10)
        cloud_cfg = self.source_cfg.get("cloud_filter", {})
        col_meta = _COLLECTION_CFG.get(collection_id, {})
        is_static = col_meta.get("static", False)

        stats = {"downloaded": 0, "skipped": 0, "failed": 0}

        def _download_one(patch: dict) -> str:
            pid = patch["id"]
            out_dir = self.patch_out_dir(pid)
            if is_static and len(list(out_dir.glob("*.tif"))) >= 1:
                return "skipped"

            left, bottom, right, top = patch["utm_bounds"]
            geom = _utm_bounds_to_ee_geom(left, bottom, right, top, crs)

            try:
                if is_static:
                    image = ee.Image(collection_id).select(bands)
                    _export_image(image, out_dir / "static.tif",
                                  geom, crs, resolution_m, image_size)
                    return "downloaded"

                # 时序 collection
                col = ee.ImageCollection(collection_id).filterBounds(geom).filterDate(
                    time_range["start"], time_range["end"]
                )

                # 云量过滤
                if cloud_cfg.get("enabled") and "max_cloud" in col_meta:
                    max_cc = cloud_cfg.get("s2_cloud_prob_max",
                                          cloud_cfg.get("max_cloud_cover", 70))
                    col = col.filter(ee.Filter.lt(col_meta["max_cloud"], max_cc))

                # S1 仪器模式过滤
                if col_meta.get("filter_mode") == "IW":
                    col = (col
                           .filter(ee.Filter.eq("instrumentMode", "IW"))
                           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

                images = col.select(bands).toList(col.size())
                n = images.size().getInfo()
                if n == 0:
                    return "skipped"

                for i in range(n):
                    img = ee.Image(images.get(i))
                    ts = img.date().format("YYYYMMdd").getInfo()
                    dst_file = out_dir / f"{ts}.tif"
                    if dst_file.exists():
                        continue
                    _export_image(img, dst_file, geom, crs, resolution_m, image_size)

                return "downloaded"

            except Exception:
                logger.debug(f"  patch_{pid:04d} GEE 失败: {traceback.format_exc()[-300:]}")
                return "failed"

        logger.info(f"[GEE/{self.source_name}] 开始下载 {len(patches)} 个 patch，workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_download_one, p): p for p in patches}
            for fut in as_completed(futs):
                stats[fut.result()] += 1

        logger.info(f"[GEE/{self.source_name}] 完成: {stats}")
        self.save_download_report(stats)
        return stats


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def _utm_bounds_to_ee_geom(
    left: float, bottom: float, right: float, top: float, crs: str
) -> "ee.Geometry":
    """将 UTM bounds 转为 ee.Geometry（先转回 WGS84）。"""
    import ee
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs, ys = to_wgs.transform([left, right], [bottom, top])
    return ee.Geometry.Rectangle([xs[0], ys[0], xs[1], ys[1]])


def _export_image(
    image: "ee.Image",
    dst_file: Path,
    geom: "ee.Geometry",
    crs: str,
    resolution_m: int,
    image_size: int,
) -> None:
    """使用 getPixels 下载单景影像并写入 GeoTIFF。"""
    import io
    import ee
    import numpy as np

    # GEE getPixels 直接下载为 numpy（需要 ee.data.getPixels）
    params = {
        "expression": image,
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": image_size, "height": image_size},
            "affineTransform": _bounds_to_gee_transform(geom, crs, image_size),
            "crsCode": crs,
        },
    }
    response = ee.data.computePixels(params)
    arr_np = np.load(io.BytesIO(response))
    # arr_np shape: structured array → stack to (C, H, W)
    bands = arr_np.dtype.names
    arr = np.stack([arr_np[b] for b in bands], axis=0).astype(np.float32)

    # 从 geom bounds 获取写入坐标
    from pyproj import Transformer
    coords = geom.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = to_utm.transform(lons, lats)
    bounds = [min(xs), min(ys), max(xs), max(ys)]

    write_tif(dst_file, arr, crs=crs, bounds=bounds)


def _bounds_to_gee_transform(geom: "ee.Geometry", crs: str, image_size: int) -> dict:
    """为 getPixels 构造仿射变换参数。"""
    import ee
    from pyproj import Transformer

    coords = geom.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = to_utm.transform(lons, lats)
    left, bottom, right, top = min(xs), min(ys), max(xs), max(ys)
    pixel_size = (right - left) / image_size
    return {
        "scaleX": pixel_size,
        "shearX": 0,
        "translateX": left,
        "shearY": 0,
        "scaleY": -pixel_size,
        "translateY": top,
    }
