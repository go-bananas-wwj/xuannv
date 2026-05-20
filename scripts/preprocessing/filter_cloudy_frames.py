"""
S2 云筛选预处理脚本

策略:
1. 对每个 patch 的 S2 帧计算云量指标 (brightness + NDVI)
2. 按月分组，每月保留 cloud_score 最低 (最 clear) 的帧
3. 如果某月所有帧 cloud_score > threshold，则保留 cloud_score 最低的一张 (fallback)
4. 将筛选后的帧复制到新目录，保持原有结构

输出:
  /workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered/s2/patch_xxxx/YYYYMMDD.tif
"""
import sys
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import rasterio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import shutil
import json

# 云量评分: 亮度越高 + NDVI越低 = 越 cloudy
# cloud_score = brightness / 10000 - ndvi * 2
# 值域大约 [-1, 1]，越高越 cloudy

def compute_cloud_score(data: np.ndarray) -> float:
    """计算云量评分，越低越 clear。"""
    blue, green, red, nir, swir1, swir2 = data.astype(np.float32)
    
    # 自适应：检测0-1范围数据（GEE导出不同scale），先×10000还原
    if data.max() < 2.0:
        blue, green, red, nir, swir1, swir2 = [
            band * 10000.0 for band in [blue, green, red, nir, swir1, swir2]
        ]
    
    brightness = (blue + green + red).mean() / 3.0
    ndvi = np.nanmean((nir - red) / (nir + red + 1e-6))
    # 亮度归一化到 [0,1] (假设最大 ~10000)，NDVI 到 [-1,1]
    cloud_score = brightness / 10000.0 - ndvi
    return cloud_score


def process_patch(args):
    patch_id, src_dir, out_dir, max_per_month, cloud_threshold = args
    patch_src = Path(src_dir) / patch_id
    patch_out = Path(out_dir) / patch_id
    patch_out.mkdir(parents=True, exist_ok=True)
    
    if not patch_src.exists():
        return patch_id, 0, 0, "missing"
    
    tif_files = sorted(patch_src.glob("*.tif"))
    if not tif_files:
        return patch_id, 0, 0, "empty"
    
    # 计算每帧的云量
    frames = []
    for f in tif_files:
        try:
            with rasterio.open(f) as src:
                data = src.read()
                score = compute_cloud_score(data)
                frames.append((f, score))
        except Exception:
            continue
    
    if not frames:
        return patch_id, 0, 0, "read_error"
    
    # 按月分组
    monthly = defaultdict(list)
    for f, score in frames:
        month = f.stem[:6]  # YYYYMM
        monthly[month].append((f, score))
    
    selected = []
    all_cloudy_count = 0
    for month, month_frames in sorted(monthly.items()):
        # 按 cloud_score 排序 (越低越 clear)
        month_frames.sort(key=lambda x: x[1])
        
        # 检查是否全部 cloudy
        if all(score > cloud_threshold for _, score in month_frames):
            all_cloudy_count += 1
            # Fallback: 保留最 clear 的一张
            selected.append(month_frames[0])
        else:
            # 保留最 clear 的 max_per_month 张
            selected.extend(month_frames[:max_per_month])
    
    # 复制筛选后的帧
    n_copied = 0
    for f, score in selected:
        dst = patch_out / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            n_copied += 1
    
    return patch_id, n_copied, all_cloudy_count, "ok"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/workspace/raw/harbin_scenes/harbin_scenes/s2")
    parser.add_argument("--out", default="/workspace/raw/harbin_scenes/harbin_scenes_cloud_filtered/s2")
    parser.add_argument("--max-per-month", type=int, default=1, help="每月最多保留几张")
    parser.add_argument("--cloud-threshold", type=float, default=0.3, help="cloud_score 阈值，高于此认为 cloudy")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="只统计不复制")
    args = parser.parse_args()
    
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    
    patches = sorted([d.name for d in src_dir.iterdir() if d.is_dir() and d.name.startswith("patch_")])
    print(f"[CloudFilter] {len(patches)} patches, max_per_month={args.max_per_month}, threshold={args.cloud_threshold}")
    
    if args.dry_run:
        print("[DRY RUN] 只统计，不复制文件")
    
    # 先分析一个样本 patch 的阈值分布
    sample_patch = patches[0]
    sample_dir = src_dir / sample_patch
    sample_scores = []
    for f in sorted(sample_dir.glob("*.tif")):
        try:
            with rasterio.open(f) as src:
                score = compute_cloud_score(src.read())
                sample_scores.append((f.stem, score))
        except:
            continue
    sample_scores.sort(key=lambda x: x[1])
    print(f"\n样本 patch {sample_patch} 的 cloud_score 分布:")
    print(f"  最 clear: {sample_scores[0][0]} = {sample_scores[0][1]:.3f}")
    print(f"  中位数: {sample_scores[len(sample_scores)//2][0]} = {sample_scores[len(sample_scores)//2][1]:.3f}")
    print(f"  最 cloudy: {sample_scores[-1][0]} = {sample_scores[-1][1]:.3f}")
    
    # 并行处理
    worker_args = [(p, str(src_dir), str(out_dir), args.max_per_month, args.cloud_threshold) for p in patches]
    
    results = []
    total_copied = 0
    total_all_cloudy = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for pid, n_copied, n_all_cloudy, status in executor.map(process_patch, worker_args):
            results.append((pid, n_copied, n_all_cloudy, status))
            total_copied += n_copied
            total_all_cloudy += n_all_cloudy
    
    # 统计
    before_total = sum(len(list((src_dir / p).glob("*.tif"))) for p in patches)
    after_total = sum(len(list((out_dir / p).glob("*.tif"))) for p in patches) if not args.dry_run else 0
    
    print(f"\n{'='*50}")
    print(f"筛选完成:")
    print(f"  原始帧数: {before_total}")
    print(f"  筛选后帧数: {after_total if not args.dry_run else '(dry run)'}")
    print(f"  全 cloudy 月份 (fallback): {total_all_cloudy}")
    print(f"  平均每 patch 保留: {after_total/len(patches):.1f}" if not args.dry_run else "")
    print(f"{'='*50}")
    
    # 保存统计
    stats = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "max_per_month": args.max_per_month,
        "cloud_threshold": args.cloud_threshold,
        "n_patches": len(patches),
        "before_total": before_total,
        "after_total": after_total,
        "all_cloudy_fallbacks": total_all_cloudy,
    }
    stats_file = out_dir / "cloud_filter_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"统计已保存: {stats_file}")


if __name__ == "__main__":
    main()
