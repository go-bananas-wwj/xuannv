"""Fast 256x256 dataset creation using multiprocessing."""
from __future__ import annotations

import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.enums import Resampling


def resize_one(args):
    src_path, dst_path, target_size = args
    try:
        with rasterio.open(src_path) as src:
            if src.width == target_size and src.height == target_size:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                return True
            data = src.read(
                out_shape=(src.count, target_size, target_size),
                resampling=Resampling.bilinear,
            )
            profile = src.profile.copy()
            profile.update({
                "height": target_size,
                "width": target_size,
                "count": src.count,
                "dtype": data.dtype,
            })
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
        return True
    except Exception as e:
        print(f"Error: {src_path}: {e}")
        return False


def main():
    target_size = 256
    max_workers = 8
    
    # Collect all tasks
    tasks = []
    
    # Haidian scenes
    src_root = Path("data_raw/haidian/scenes")
    dst_root = Path("data_raw/haidian/scenes_256")
    for patch_dir in sorted(src_root.iterdir()):
        if not patch_dir.is_dir():
            continue
        for source_dir in patch_dir.iterdir():
            if not source_dir.is_dir():
                continue
            dst_source_dir = dst_root / patch_dir.name / source_dir.name
            for tif_path in source_dir.glob("*.tif"):
                rel = tif_path.relative_to(source_dir)
                dst_path = dst_source_dir / rel
                if not dst_path.exists():
                    tasks.append((tif_path, dst_path, target_size))
    
    # Planet
    src_root = Path("data_raw/beijing/planetscene")
    dst_root = Path("data_raw/beijing/planetscene_256")
    for patch_dir in sorted(src_root.iterdir()):
        if not patch_dir.is_dir():
            continue
        dst_patch_dir = dst_root / patch_dir.name
        for tif_path in patch_dir.glob("*.tif"):
            dst_path = dst_patch_dir / tif_path.name
            if not dst_path.exists():
                tasks.append((tif_path, dst_path, target_size))
    
    print(f"Total files to process: {len(tasks)}")
    if not tasks:
        print("All done!")
        return
    
    success = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(resize_one, tasks)):
            if result:
                success += 1
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(tasks)} ({success} success)")
    
    print(f"Done: {success}/{len(tasks)} files resized to {target_size}x{target_size}")


if __name__ == "__main__":
    main()
