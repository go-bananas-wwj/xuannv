#!/usr/bin/env python3
"""
从 Source Cooperative 直接下载 AEF 2025 嵌入

数据地址: https://source.coop/tge-labs/aef/v1/annual/2025/
索引: https://source.coop/tge-labs/aef/v1/annual/aef_index.csv

用法:
    python download_aef_source_coop.py --region harbin --year 2025
    python download_aef_source_coop.py --region haidian --year 2025
"""
import os, sys, csv, json, argparse, time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

# 需要先安装 rasterio
# pip install rasterio pandas

sys.path.insert(0, "/workspace/xuannv")

REGIONS = {
    "harbin": {
        "bounds": [126.0, 45.0, 128.5, 46.5],
        "patches_meta": "/workspace/xuannv/data_raw/olmoearth_harbin/patches_meta.json",
        "output_dir": "/workspace/xuannv/data_raw/aef_embeddings/harbin_2025_patches",
        "crs": "EPSG:32650",
    },
    "haidian": {
        "bounds": [116.0, 39.5, 116.5, 40.2],
        "patches_meta": "/workspace/xuannv/data_raw/olmoearth_haidian/patches_meta.json",
        "output_dir": "/workspace/xuannv/data_raw/aef_embeddings/haidian_2025_patches",
        "crs": "EPSG:32650",
    },
}

SOURCE_COOP_BASE = "https://data.source.coop/tge-labs/aef"


def dequantize_aef(arr_int8: np.ndarray) -> np.ndarray:
    """AEF 反量化: ((v / 127.5) ** 2) * sign(v).
    
    -128 是 NoData 值.
    """
    # 创建掩码，排除 NoData
    valid_mask = arr_int8 != -128
    result = np.zeros_like(arr_int8, dtype=np.float32)
    
    # 反量化
    v = arr_int8.astype(np.float32)
    result[valid_mask] = ((v[valid_mask] / 127.5) ** 2) * np.sign(v[valid_mask])
    return result, valid_mask


def find_covering_files(index_csv: Path, bounds: list, year: int) -> list:
    """从索引中筛选覆盖目标区域的文件."""
    min_lon, min_lat, max_lon, max_lat = bounds
    files = []
    
    print(f"[筛选] 区域 bounds: {bounds}")
    print(f"[筛选] 年份: {year}")
    
    with open(index_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['year']) != year:
                continue
            # 检查 bbox 重叠
            f_min_lon = float(row['wgs84_west'])
            f_min_lat = float(row['wgs84_south'])
            f_max_lon = float(row['wgs84_east'])
            f_max_lat = float(row['wgs84_north'])
            
            # 简单 bbox 相交测试
            if (f_min_lon <= max_lon and f_max_lon >= min_lon and
                f_min_lat <= max_lat and f_max_lat >= min_lat):
                
                # 将 s3:// 路径转换为 HTTPS URL
                path = row['path']
                if path.startswith('s3://us-west-2.opendata.source.coop/'):
                    url = path.replace('s3://us-west-2.opendata.source.coop/', 
                                       'https://data.source.coop/')
                else:
                    url = path
                
                files.append({
                    'url': url,
                    'crs': row['crs'],
                    'utm_zone': row['utm_zone'],
                    'bounds': [f_min_lon, f_min_lat, f_max_lon, f_max_lat],
                })
    
    print(f"[筛选] 找到 {len(files)} 个覆盖文件")
    for f in files:
        print(f"       {f['utm_zone']}: {f['bounds']}")
    return files


def download_cog(url: str, output_path: Path) -> bool:
    """下载 COG 文件."""
    import subprocess
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用 curl 下载
    cmd = ['curl', '-sL', '-o', str(output_path), url]
    print(f"[下载] {url}")
    print(f"       -> {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] 下载失败: {result.stderr}")
        return False
    
    # 检查文件大小
    size = output_path.stat().st_size
    print(f"       大小: {size / 1024 / 1024:.1f} MB")
    return size > 0


def download_with_vrt(cog_url: str, output_path: Path) -> bool:
    """下载 COG + VRT，使用 VRT 纠正 bottom-up 问题."""
    vrt_url = cog_url.replace('.tiff', '.vrt')
    
    cog_path = output_path
    vrt_path = output_path.with_suffix('.vrt')
    
    if not download_cog(cog_url, cog_path):
        return False
    
    # VRT 可选（如果存在）
    import subprocess
    result = subprocess.run(['curl', '-sI', vrt_url], capture_output=True, text=True)
    if result.returncode == 0 and '200' in result.stdout:
        download_cog(vrt_url, vrt_path)
    
    return True


def merge_and_crop(files: list, region_name: str, year: int, output_dir: Path):
    """合并下载的 COG 文件并裁剪到 patch 级别."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling
    
    region = REGIONS[region_name]
    bounds = region["bounds"]
    patches_meta = region["patches_meta"]
    
    # 读取所有 COG
    sources = []
    print("[合并] 打开 COG 文件...")
    for f in files:
        # 使用 VRT 如果存在，否则用 COG
        cog_path = Path(f['url'].replace('https://data.source.coop/', '/workspace/xuannv/data_raw/aef_embeddings/cache/'))
        vrt_path = cog_path.with_suffix('.vrt')
        
        if vrt_path.exists():
            src_path = str(vrt_path)
        else:
            src_path = str(cog_path)
        
        try:
            src = rasterio.open(src_path)
            sources.append(src)
            print(f"       打开: {src_path} ({src.crs}, {src.shape})")
        except Exception as e:
            print(f"[警告] 无法打开 {src_path}: {e}")
    
    if not sources:
        print("[错误] 没有可读取的源文件")
        return
    
    # 合并
    print("[合并] 正在合并...")
    mosaic, out_transform = merge(sources, bounds=bounds)
    print(f"[合并] 完成: {mosaic.shape}")
    
    # 反量化
    print("[反量化] 处理像素值...")
    mosaic_float, valid_mask = dequantize_aef(mosaic)
    
    # 保存合并后的整体文件（可选）
    merged_path = output_dir.parent / f"aef_{region_name}_{year}_merged.tif"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_profile = sources[0].profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": 64,
        "transform": out_transform,
        "crs": "EPSG:4326",
        "dtype": "float32",
        "compress": "lzw",
    })
    
    with rasterio.open(merged_path, "w", **out_profile) as dst:
        dst.write(mosaic_float)
    print(f"[保存] 合并文件: {merged_path}")
    
    # 裁剪到 patch
    print("[裁剪] 按 patch 裁剪...")
    with open(patches_meta) as f:
        patches = json.load(f)
    
    patch_output_dir = Path(output_dir)
    patch_output_dir.mkdir(parents=True, exist_ok=True)
    
    for patch in patches:
        patch_id = patch["patch_id"]
        pbounds = patch["bounds"]  # [minx, miny, maxx, maxy]
        
        # 计算像素窗口
        min_col, min_row = ~out_transform * (pbounds[0], pbounds[3])
        max_col, max_row = ~out_transform * (pbounds[2], pbounds[1])
        
        min_col, max_col = int(min_col), int(max_col)
        min_row, max_row = int(min_row), int(max_row)
        
        # 边界检查
        min_col = max(0, min_col)
        min_row = max(0, min_row)
        max_col = min(mosaic.shape[2], max_col)
        max_row = min(mosaic.shape[1], max_row)
        
        if max_col <= min_col or max_row <= min_row:
            print(f"[跳过] {patch_id}: 超出范围")
            continue
        
        patch_emb = mosaic_float[:, min_row:max_row, min_col:max_col]
        out_path = patch_output_dir / f"{patch_id}.npy"
        np.save(out_path, patch_emb)
    
    print(f"[完成] 已裁剪 {len(patches)} 个 patches 到 {patch_output_dir}")
    
    # 关闭源文件
    for src in sources:
        src.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["harbin", "haidian"], required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--index-csv", type=Path, default=Path("/workspace/xuannv/data_raw/aef_embeddings/aef_index.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/xuannv/data_raw/aef_embeddings/cache"))
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，使用已有缓存")
    parser.add_argument("--skip-merge", action="store_true", help="跳过合并，只下载")
    args = parser.parse_args()
    
    region = REGIONS[args.region]
    
    print("=" * 60)
    print(f"AEF Source Cooperative 下载: {args.region} / {args.year}")
    print("=" * 60)
    
    # 步骤 1: 筛选文件
    if not args.index_csv.exists():
        print(f"[错误] 索引文件不存在: {args.index_csv}")
        print("[提示] 先运行: curl -sL 'https://data.source.coop/tge-labs/aef/v1/annual/aef_index.csv' -o aef_index.csv")
        return
    
    files = find_covering_files(args.index_csv, region["bounds"], args.year)
    if not files:
        print("[错误] 未找到覆盖该区域的文件")
        return
    
    # 步骤 2: 下载
    if not args.skip_download:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            url = f['url']
            filename = Path(urlparse(url).path).name
            output_path = args.cache_dir / filename
            
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"[跳过] 已存在: {output_path}")
                continue
            
            success = download_cog(url, output_path)
            if not success:
                print(f"[警告] 下载失败，跳过: {url}")
    
    # 步骤 3: 合并裁剪
    if not args.skip_merge:
        merge_and_crop(files, args.region, args.year, region["output_dir"])
    
    print("\n[完成]")


if __name__ == "__main__":
    main()
