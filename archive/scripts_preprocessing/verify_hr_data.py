#!/usr/bin/env python3
"""验证高分辨率数据预处理的完整性."""
from pathlib import Path
import json

GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
S2_HR_DIR = Path("/workspace/raw/harbin_scenes/s2_hr")
S1_HR_DIR = Path("/workspace/raw/harbin_scenes/s1_hr")

import geopandas as gpd

def main():
    grid = gpd.read_file(str(GRID_PATH))
    pids = sorted(grid["patch_id"].unique().tolist())

    expected_s2 = ["20250409", "20250622", "20250804", "20250901", "20251001"]
    expected_s1 = ["20250627", "20250810", "20250912", "20251005"]

    s2_ok = 0
    s1_ok = 0
    s2_issues = []
    s1_issues = []

    for pid in pids:
        s2_dir = S2_HR_DIR / pid
        if s2_dir.exists():
            files = sorted([f.stem for f in s2_dir.glob("*.tif")])
            if files == expected_s2:
                s2_ok += 1
            else:
                s2_issues.append({"patch_id": pid, "expected": expected_s2, "actual": files})
        else:
            s2_issues.append({"patch_id": pid, "error": "dir_missing"})

        s1_dir = S1_HR_DIR / pid
        if s1_dir.exists():
            files = sorted([f.stem for f in s1_dir.glob("*.tif")])
            if files == expected_s1:
                s1_ok += 1
            else:
                s1_issues.append({"patch_id": pid, "expected": expected_s1, "actual": files})
        else:
            s1_issues.append({"patch_id": pid, "error": "dir_missing"})

    print(f"S2_HR: {s2_ok}/{len(pids)} patches complete")
    print(f"S1_HR: {s1_ok}/{len(pids)} patches complete")
    if s2_issues:
        print(f"S2_HR issues: {len(s2_issues)}")
        for issue in s2_issues[:5]:
            print(f"  {issue}")
    if s1_issues:
        print(f"S1_HR issues: {len(s1_issues)}")
        for issue in s1_issues[:5]:
            print(f"  {issue}")

    report = {
        "total_patches": len(pids),
        "s2_hr_complete": s2_ok,
        "s1_hr_complete": s1_ok,
        "s2_hr_issues": len(s2_issues),
        "s1_hr_issues": len(s1_issues),
        "s2_hr_issue_details": s2_issues,
        "s1_hr_issue_details": s1_issues,
    }
    out_path = Path("/workspace/raw/harbin_scenes/hr_verification_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
