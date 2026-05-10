#!/usr/bin/env python3
"""下载海淀区OSM数据"""
import os

OUTPUT_DIR = "/workspace/raw/haidian/osm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BBOX = (116.05, 39.88, 116.38, 40.15)  # left, bottom, right, top

print("=" * 50)
print("Downloading OSM data for Haidian District")
print(f"BBox: {BBOX}")
print("=" * 50)

try:
    import osmnx as ox
    
    print("\n[1/2] Downloading buildings...")
    buildings = ox.features.features_from_bbox(
        bbox=BBOX,
        tags={"building": True}
    )
    print(f"  Found {len(buildings)} building features")
    out_b = f"{OUTPUT_DIR}/haidian_osm_buildings.gpkg"
    buildings.to_file(out_b, driver="GPKG")
    print(f"  Saved: {out_b}")
    
    print("\n[2/2] Downloading roads...")
    roads = ox.features.features_from_bbox(
        bbox=BBOX,
        tags={"highway": True}
    )
    print(f"  Found {len(roads)} road features")
    out_r = f"{OUTPUT_DIR}/haidian_osm_roads.gpkg"
    roads.to_file(out_r, driver="GPKG")
    print(f"  Saved: {out_r}")
    
    print("\n" + "=" * 50)
    print("OSM download completed successfully!")
    print("=" * 50)
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
