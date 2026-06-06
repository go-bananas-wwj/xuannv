#!/usr/bin/env python3
"""下载小文件: WorldCover, CLCD, TEMPO"""
import os
import requests
from tqdm import tqdm

OUTPUT_BASE = "/workspace/xuannv/data_raw/haidian"

def download_file(url, out_path, desc=""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return
    print(f"  Downloading: {desc}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(out_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=desc) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"  Saved: {out_path}")

print("=" * 50)
print("Downloading reference datasets...")
print("=" * 50)

# 1. ESA WorldCover (Haidian subset via GEE would be better, 
#    but we can download global tile and clip)
# For now, we'll download via direct link if available
print("\n[1/3] ESA WorldCover 2021...")
# WorldCover is available via direct links per tile
# Haidian is in tile N39E116, we'll skip direct download and use GEE later
print("  (Will download via GEE after authentication)")

# 2. CLCD
print("\n[2/3] CLCD China Land Cover...")
# Zenodo links - we'll note these for manual download
print("  URL: https://zenodo.org/records/5205676")
print("  (Please download manually or via zenodo_get)")

# 3. Microsoft TEMPO
print("\n[3/3] Microsoft TEMPO...")
# Available via GitHub release links
print("  URL: https://github.com/microsoft/buildings")
print("  (Please download manually - requires Azure blob access)")

print("\n" + "=" * 50)
print("Note: These datasets require manual download or GEE.")
print("Starting GEE-based downloads after auth...")
print("=" * 50)
