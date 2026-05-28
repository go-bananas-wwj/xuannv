"""Microsoft Planetary Computer 下载器。

依赖:
    pip install planetary-computer pystac-client odc-stac rioxarray
"""
from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from preprocessing.downloaders.base import BaseDownloader
from preprocessing.utils.logging import get_logger
from preprocessing.utils.tiff import write_tif

logger = get_logger(__name__)


class PlanetaryComputerDownloader(BaseDownloader):
    """通过 STAC API 从 Planetary Computer 下载影像并切 patch。"""

    # 各 collection 的时间字段名（用于从 STAC item properties 解析日期）
    _DATE_FIELDS = {
        "sentinel-2-l2a": "datetime",
        "sentinel-1-rtc": "datetime",
        "landsat-c2-l2": "datetime",
        "cop-dem-glo-30": None,   # DEM 静态，不分时间
        "esa-worldcover": None,   # 静态
    }

    def __init__(self, region_cfg: dict, source_name: str) -> None:
        super().__init__(region_cfg, source_name)
        self._setup_credentials()

    def _setup_credentials(self) -> None:
        """从环境变量读取 PC SDK 订阅 key。"""
        creds = self.region_cfg.get("credentials", {}).get("planetary_computer", {})
        key_env = creds.get("subscription_key_env", "PC_SDK_SUBSCRIPTION_KEY")
        key = os.environ.get(key_env, "")
        if key:
            os.environ["PC_SDK_SUBSCRIPTION_KEY"] = key
            logger.info("[PC] 订阅 key 已配置")
        else:
            logger.warning("[PC] 未找到订阅 key，将以匿名模式下载（速率受限）")

    # ------------------------------------------------------------------ #

    def download(self, patches: list[dict], *, workers: int = 4) -> dict[str, int]:
        import planetary_computer
        import pystac_client

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        collection = self.source_cfg["collection"]
        bands = self.source_cfg["bands"]
        time_range = self.region_cfg["time_range"]
        crs = self.region_cfg.get("crs", "EPSG:32652")
        image_size = self.region_cfg["patch"]["image_size_px"]
        cloud_cfg = self.source_cfg.get("cloud_filter", {})

        date_filter = f"{time_range['start']}/{time_range['end']}"
        date_field = self._DATE_FIELDS.get(collection, "datetime")
        is_static = date_field is None

        stats = {"downloaded": 0, "skipped": 0, "failed": 0}

        def _download_one(patch: dict) -> tuple[str, str]:
            pid = patch["id"]
            out_dir = self.patch_out_dir(pid)
            existing = len(list(out_dir.glob("*.tif")))

            if is_static and existing >= 1:
                return str(pid), "skipped"

            left, bottom, right, top = patch["utm_bounds"]

            try:
                # STAC 搜索
                query = {
                    "collections": [collection],
                    "bbox": [*_utm_to_wgs84_bbox(left, bottom, right, top, crs)],
                    "datetime": date_filter if not is_static else None,
                }
                if cloud_cfg.get("enabled") and collection == "sentinel-2-l2a":
                    query["query"] = {
                        "eo:cloud_cover": {"lt": cloud_cfg.get("s2_cloud_prob_max", 20)}
                    }

                search = catalog.search(**{k: v for k, v in query.items() if v is not None})
                items = list(search.items())

                if not items:
                    logger.debug(f"  patch_{pid:04d}: 无搜索结果")
                    return str(pid), "skipped"

                # 按时间分组，下载每帧
                import rioxarray  # noqa
                import odc.stac

                for item in items:
                    if date_field:
                        dt_str = item.properties.get(date_field, "")
                        if not dt_str:
                            continue
                        date_tag = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).strftime("%Y%m%d")
                    else:
                        date_tag = "static"

                    dst_file = out_dir / f"{date_tag}.tif"
                    if dst_file.exists():
                        continue

                    # 加载单景
                    ds = odc.stac.load(
                        [item],
                        bands=bands,
                        crs=crs,
                        resolution=self.source_cfg.get("resolution_m", 10),
                        bbox=[left, bottom, right, top],
                        chunks={},
                    ).compute()

                    arr = np.stack([ds[b].values.squeeze() for b in bands], axis=0)  # (C, H, W)
                    # resize 到目标 image_size
                    if arr.shape[1] != image_size or arr.shape[2] != image_size:
                        from skimage.transform import resize as sk_resize
                        arr_r = np.stack([
                            sk_resize(arr[c], (image_size, image_size),
                                      anti_aliasing=True, preserve_range=True)
                            for c in range(arr.shape[0])
                        ], axis=0)
                    else:
                        arr_r = arr

                    write_tif(dst_file, arr_r.astype(np.float32), crs=crs,
                              bounds=[left, bottom, right, top])

                return str(pid), "downloaded"

            except Exception:
                logger.debug(f"  patch_{pid:04d} 失败: {traceback.format_exc()[-300:]}")
                return str(pid), "failed"

        logger.info(f"[PC/{self.source_name}] 开始下载 {len(patches)} 个 patch，workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_download_one, p): p for p in patches}
            for fut in as_completed(futs):
                _, status = fut.result()
                stats[status] += 1

        logger.info(f"[PC/{self.source_name}] 完成: {stats}")
        self.save_download_report(stats)
        return stats


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def _utm_to_wgs84_bbox(left: float, bottom: float, right: float, top: float,
                       crs: str) -> list[float]:
    """将 UTM bbox 转为 WGS84 [west, south, east, north]。"""
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs, ys = to_wgs.transform([left, right], [bottom, top])
    return [xs[0], ys[0], xs[1], ys[1]]
