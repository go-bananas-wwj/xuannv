#!/usr/bin/env python3
"""
替换海淀区 S1 数据：将 PC 下载的 RTC 数据替换原有的 GEE S1 数据。
同时保留原有目录结构，方便训练代码直接读取。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC_DIR = Path("/workspace/xuannv/data_raw/haidian/s1_pc_download")
DST_DIR = Path("/workspace/xuannv/data_raw/haidian/scenes")


def main():
    patches = sorted([p.name for p in SRC_DIR.iterdir() if p.is_dir()])
    print(f"找到 {len(patches)} 个 patch 待替换")

    replaced = 0
    skipped = 0
    for patch_id in patches:
        src_s1 = SRC_DIR / patch_id / "s1"
        dst_s1 = DST_DIR / patch_id / "s1"

        if not src_s1.exists():
            print(f"  [SKIP] {patch_id}: 源目录不存在")
            skipped += 1
            continue

        # 删除原有 S1 数据
        if dst_s1.exists():
            n_old = len(list(dst_s1.glob("*.tif")))
            shutil.rmtree(dst_s1)
            print(f"  {patch_id}: 删除 {n_old} 个旧文件")

        # 复制新数据
        shutil.copytree(src_s1, dst_s1)
        n_new = len(list(dst_s1.glob("*.tif")))
        print(f"  {patch_id}: 复制 {n_new} 个新文件")
        replaced += 1

    print(f"\n完成: 替换 {replaced}, 跳过 {skipped}")


if __name__ == "__main__":
    main()
