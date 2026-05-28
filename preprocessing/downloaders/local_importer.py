"""本地 SAR 数据导入器。

处理 /workspace/raw/haidian_sar/ 下的干涉 SAR ZIP 产品，流程：
    1. 扫描所有 ZIP 文件，解析时间戳与极化信息
    2. 解压 → 尝试读取振幅 TIFF（若含 GeoTIFF）或生成振幅图像
    3. 从影像范围确定所属 patch，裁剪到 patch_bounds
    4. 写入到标准目录 output_dir/sar/patch_XXXX/YYYYMMDD.tif

SAR ZIP 文件命名规则（已观测）：
    BC3-SM-SLC-1SVV-20250120T142359-<orbit>-<pass>-<id>.zip
    BC3-SM-ORG-2SVV-20250120T142359-<orbit>-<pass>-<id>.zip

    字段:
        卫星:  BC3 / BC4
        模式:  SM (Stripmap)
        类型:  SLC (Single Look Complex) | ORG (Original)
        极化:  1SVV (单极化VV) | 2SVV (双极化VV+VH)
        时间:  YYYYMMDDTHHMMSS
"""
from __future__ import annotations

import re
import shutil
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from preprocessing.downloaders.base import BaseDownloader
from preprocessing.utils.logging import get_logger
from preprocessing.utils.tiff import write_tif

logger = get_logger(__name__)

# 文件名解析正则
_ZIP_RE = re.compile(
    r"(?P<sat>BC\d+)-"
    r"(?P<mode>[A-Z]+)-"
    r"(?P<ptype>[A-Z]+)-"
    r"(?P<pol>\dS[A-Z]+)-"
    r"(?P<dt>\d{8}T\d{6})-"
    r"(?P<orbit>\d+)-"
    r".*\.zip$",
    re.IGNORECASE,
)


def parse_sar_zip_name(filename: str) -> dict | None:
    """解析 SAR ZIP 文件名，返回元数据字典；匹配失败返回 None。"""
    m = _ZIP_RE.match(Path(filename).name)
    if not m:
        return None
    return {
        "satellite": m.group("sat"),
        "mode": m.group("mode"),
        "product_type": m.group("ptype"),
        "polarization": m.group("pol"),
        "datetime": m.group("dt"),          # YYYYMMDDTHHMMSS
        "date_tag": m.group("dt")[:8],      # YYYYMMDD
        "orbit": m.group("orbit"),
    }


class LocalSARImporter(BaseDownloader):
    """
    导入并预处理本地干涉 SAR ZIP 产品。

    主要完成：
        - ZIP 解压到临时目录
        - SLC → 振幅图（dB）转换 + 保存为 GeoTIFF
        - 空间裁剪到 patch 覆盖范围
        - 写入标准 output_dir/sar/patch_XXXX/YYYYMMDD.tif

    NOTE: 完整的 SLC 干涉处理（配准/相位解缠/几何校正）需要 SNAP 或
    专用 SAR 工具链，这超出本模块范围。本导入器假设 ZIP 中已包含
    可用的振幅 GeoTIFF，或对 SLC 直接取模并地理配准（需 DEM）。
    若 ZIP 内无现成 GeoTIFF，本导入器会记录 skip 并输出警告。
    """

    def __init__(self, region_cfg: dict, source_name: str = "sar") -> None:
        super().__init__(region_cfg, source_name)
        self.local_raw_dir = Path(self.source_cfg["local_raw_dir"])
        self.product_types = set(self.source_cfg.get("product_types", ["SLC", "ORG"]))
        self.polarizations = set(self.source_cfg.get("polarizations", ["VV"]))
        self.satellites = set(self.source_cfg.get("satellites", ["BC3", "BC4"]))
        self._tmp_dir = Path("/tmp/sar_unzip")

    # ------------------------------------------------------------------ #

    def scan_zips(self) -> list[dict]:
        """扫描 local_raw_dir，返回所有符合条件的 ZIP 文件元数据列表。"""
        results: list[dict] = []
        for zip_path in sorted(self.local_raw_dir.rglob("*.zip")):
            meta = parse_sar_zip_name(zip_path.name)
            if meta is None:
                continue
            # 过滤卫星/类型/极化
            if meta["satellite"] not in self.satellites:
                continue
            if meta["product_type"] not in self.product_types:
                continue
            # 极化过滤：极化字段如 "1SVV"，提取最后两位检查
            pol_code = meta["polarization"][-2:].upper()
            if not any(p in pol_code for p in self.polarizations):
                continue
            meta["zip_path"] = str(zip_path)
            results.append(meta)
        logger.info(f"[LocalSAR] 扫描到 {len(results)} 个有效 ZIP")
        return results

    def download(self, patches: list[dict], *, workers: int = 4) -> dict[str, int]:
        """
        「下载」即导入：扫描 local_raw_dir → 解压 → 预处理 → 切 patch。
        """
        zips = self.scan_zips()
        if not zips:
            logger.warning("[LocalSAR] 未扫描到有效 ZIP，跳过导入")
            return {"downloaded": 0, "skipped": 0, "failed": 0}

        # 构建 patch 空间索引（UTM bounds → shapely box）
        patch_index = _build_patch_index(patches)
        crs = self.region_cfg.get("crs", "EPSG:32650")
        image_size = self.region_cfg["patch"]["image_size_px"]

        stats = {"downloaded": 0, "skipped": 0, "failed": 0}

        def _import_one(meta: dict) -> str:
            try:
                return self._import_zip(meta, patch_index, crs, image_size)
            except Exception:
                logger.debug(f"  {meta['zip_path']} 失败: {traceback.format_exc()[-300:]}")
                return "failed"

        logger.info(f"[LocalSAR] 开始导入 {len(zips)} 个 ZIP，workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_import_one, z): z for z in zips}
            for fut in as_completed(futs):
                stats[fut.result()] += 1

        logger.info(f"[LocalSAR] 完成: {stats}")
        self.save_download_report(stats)
        return stats

    def _import_zip(
        self,
        meta: dict,
        patch_index: list[dict],
        crs: str,
        image_size: int,
    ) -> str:
        """处理单个 ZIP：解压 → 找 GeoTIFF → 按 patch 裁剪写出。"""
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject

        zip_path = Path(meta["zip_path"])
        tmp_dir = self._tmp_dir / zip_path.stem
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 解压
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # 找到 GeoTIFF（优先 .tif/.tiff，忽略复数 SLC 的实部/虚部单独文件）
        tif_candidates = sorted(tmp_dir.rglob("*.tif")) + sorted(tmp_dir.rglob("*.tiff"))
        if not tif_candidates:
            # ZIP 内无现成 TIFF（原始 SLC 格式），需外部工具处理，此处跳过
            logger.warning(f"[LocalSAR] {zip_path.name}: ZIP 内无 GeoTIFF，需手动 geocode，跳过")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return "skipped"

        # 取第一个可用 TIFF（振幅图或 ORG 产品）
        src_tif = tif_candidates[0]
        date_tag = meta["date_tag"]

        written = 0
        with rasterio.open(src_tif) as src:
            src_crs = src.crs
            # 若 CRS 与目标不一致，重投影到目标 CRS
            if str(src_crs) != crs:
                # 重投影到目标 CRS（粗略，不做精确地理校正）
                tmp_reproj = tmp_dir / "reproj.tif"
                _reproject_tif(src, tmp_reproj, crs)
                proc_tif = tmp_reproj
            else:
                proc_tif = src_tif

        # 按 patch 裁剪
        with rasterio.open(proc_tif) as src:
            for patch in patch_index:
                pid = patch["id"]
                left, bottom, right, top = patch["utm_bounds"]

                # 检查是否有交叠
                if not _bounds_intersect(
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                    left, bottom, right, top,
                ):
                    continue

                out_file = self.patch_out_dir(pid) / f"{date_tag}.tif"
                if out_file.exists():
                    continue

                # 读取 patch 区域
                from rasterio.windows import from_bounds as win_from_bounds
                win = win_from_bounds(left, bottom, right, top, src.transform)
                arr = src.read(window=win)

                if arr.size == 0:
                    continue

                # SLC 复数 → 振幅 dB
                if np.iscomplexobj(arr):
                    arr = np.abs(arr).astype(np.float32)
                    arr = 20 * np.log10(np.clip(arr, 1e-10, None))
                else:
                    arr = arr.astype(np.float32)
                    # 若数值范围像线性功率（> 0 且 < 1），转 dB
                    if arr.max() <= 1.0 and arr.min() >= 0:
                        arr = 10 * np.log10(np.clip(arr, 1e-10, None))

                # resize
                if arr.shape[1] != image_size or arr.shape[2] != image_size:
                    from skimage.transform import resize as sk_resize
                    arr = np.stack([
                        sk_resize(arr[c], (image_size, image_size),
                                  anti_aliasing=True, preserve_range=True).astype(np.float32)
                        for c in range(arr.shape[0])
                    ], axis=0)

                write_tif(out_file, arr, crs=crs, bounds=[left, bottom, right, top])
                written += 1

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return "downloaded" if written > 0 else "skipped"


# ------------------------------------------------------------------ #
# 辅助函数
# ------------------------------------------------------------------ #

def _build_patch_index(patches: list[dict]) -> list[dict]:
    """构建简单的 patch 空间索引（直接列表，使用 bbox 交叠检测）。"""
    return patches  # 简单实现，数量少时 O(N) 可接受


def _bounds_intersect(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> bool:
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _reproject_tif(src: "rasterio.DatasetReader", dst_path: Path, dst_crs: str) -> None:
    """将已打开的 rasterio 源重投影到 dst_crs，写入 dst_path。"""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    profile = src.profile.copy()
    profile.update(crs=dst_crs, transform=transform, width=width, height=height)
    with rasterio.open(dst_path, "w", **profile) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
