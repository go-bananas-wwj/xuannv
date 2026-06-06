#!/usr/bin/env python3
"""
海淀区SAR数据上传脚本 - 上传到 ModelScope
4个子目录依次上传，每个完成后记录，支持断点续传
"""
from __future__ import annotations
import os, sys, time, logging
from datetime import datetime
from modelscope.hub.api import HubApi

TOKEN    = "ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977"
REPO_ID  = "WeijieWu/haidian_sar_2025"
SRC_ROOT = "/workspace/xuannv/data_raw/haidian_sar"
LOG_FILE = f"{SRC_ROOT}/upload.log"
WORKERS  = 8

SUBDIRS = [
    "北京朝阳角反",
    "北京市门头沟区大台-干涉",
    "中国北京市点位1_干涉",
    "中国北京市点位2_干涉",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("upload")


def human_size(path: str) -> str:
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(path) for f in fs
    )
    return f"{total/1024**3:.2f} GB"


def count_zips(path: str) -> int:
    return sum(1 for r, _, fs in os.walk(path) for f in fs if f.endswith(".zip"))


def main():
    api = HubApi()
    api.login(TOKEN)
    log.info("=" * 60)
    log.info(f"开始上传: {REPO_ID}")
    log.info(f"并发: {WORKERS}  断点续传: 开启")
    log.info("=" * 60)

    total_start = time.time()
    for subdir in SUBDIRS:
        folder_path = os.path.join(SRC_ROOT, subdir)
        if not os.path.isdir(folder_path):
            log.warning(f"目录不存在，跳过: {subdir}")
            continue

        n_zip  = count_zips(folder_path)
        size   = human_size(folder_path)
        log.info(f"▶  开始上传: {subdir}  ({n_zip} 个ZIP, {size})")
        t0 = time.time()
        try:
            api.upload_folder(
                repo_id=REPO_ID,
                folder_path=folder_path,
                path_in_repo=subdir,        # 保留子目录结构
                repo_type="dataset",
                commit_message=f"upload {subdir}",
                max_workers=WORKERS,
                use_cache=True,             # 断点续传：已上传的跳过
            )
            elapsed = time.time() - t0
            log.info(f"✅ 完成: {subdir}  耗时 {elapsed/60:.1f} 分钟")
        except Exception as e:
            log.error(f"❌ 失败: {subdir}  {type(e).__name__}: {e}")
            log.error("继续下一个目录...")

    total_elapsed = time.time() - total_start
    log.info("=" * 60)
    log.info(f"🎉 全部上传完毕！总耗时 {total_elapsed/3600:.1f} 小时")
    log.info(f"   仓库地址: https://modelscope.cn/datasets/{REPO_ID}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
