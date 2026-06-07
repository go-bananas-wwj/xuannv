"""Create 256x256 dataset by resizing existing tif files."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.enums import Resampling


def resize_tif(src_path: Path, dst_path: Path, target_size: int = 256) -> bool:
    """Resize a tif file to target_size x target_size."""
    try:
        with rasterio.open(src_path) as src:
            if src.width == target_size and src.height == target_size:
                # Already target size, just copy
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
        print(f"Error resizing {src_path}: {e}")
        return False


def process_source(src_dir: Path, dst_dir: Path, target_size: int = 256) -> int:
    """Process all tif files in a source directory."""
    count = 0
    for tif_path in src_dir.rglob("*.tif"):
        rel = tif_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        if resize_tif(tif_path, dst_path, target_size):
            count += 1
    return count


def main():
    target_size = 256
    
    sources = [
        ("data_raw/haidian/scenes", "data_raw/haidian/scenes_256"),
        ("data_raw/beijing/planetscene", "data_raw/beijing/planetscene_256"),
    ]
    
    for src_root, dst_root in sources:
        src_root = Path(src_root)
        dst_root = Path(dst_root)
        
        if not src_root.exists():
            print(f"Skip {src_root} (not found)")
            continue
        
        print(f"\nProcessing {src_root} -> {dst_root}")
        
        # Collect all patch dirs
        patch_dirs = [d for d in src_root.iterdir() if d.is_dir()]
        print(f"Found {len(patch_dirs)} patches")
        
        total_tifs = 0
        for patch_dir in patch_dirs:
            dst_patch_dir = dst_root / patch_dir.name
            
            # Handle flat structure (planetscene) or nested (scenes)
            if (patch_dir / "s2").exists():
                # Nested: patch/s2/*.tif, patch/tianyi_sar/*.tif, etc.
                for source_dir in patch_dir.iterdir():
                    if not source_dir.is_dir():
                        continue
                    dst_source_dir = dst_patch_dir / source_dir.name
                    count = process_source(source_dir, dst_source_dir, target_size)
                    total_tifs += count
            else:
                # Flat: patch/*.tif (planet)
                count = process_source(patch_dir, dst_patch_dir, target_size)
                total_tifs += count
        
        print(f"Done: {total_tifs} tif files resized to {target_size}x{target_size}")


if __name__ == "__main__":
    main()
