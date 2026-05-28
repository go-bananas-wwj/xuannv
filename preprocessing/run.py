#!/usr/bin/env python3
"""
preprocessing/run.py — 数据预处理统一入口

用法:
    # 全量流程（哈尔滨）
    python preprocessing/run.py \
        --config preprocessing/configs/harbin.json \
        --steps patchify,download,cloud_filter,process,statistics

    # 仅云筛选 + 统计
    python preprocessing/run.py \
        --config preprocessing/configs/harbin.json \
        --steps cloud_filter,statistics

    # 海淀 SAR 导入
    python preprocessing/run.py \
        --config preprocessing/configs/haidian.json \
        --steps patchify,import,process,statistics \
        --workers 4

    # 仅重跑统计（哈尔滨已有数据）
    python preprocessing/run.py \
        --config preprocessing/configs/harbin.json \
        --steps statistics

可用步骤（按自然顺序）:
    patchify     : 生成 patches_meta.json（从 bbox 切 patch 网格）
    download     : 从远端（PC / GEE）下载所有启用的遥感数据源
    import       : 导入本地数据（如海淀干涉 SAR zip）
    cloud_filter : S2 云筛选，输出到 cloud_filtered_dir
    process      : 后处理：S1 dB 转换 / Landsat SR 转换 / 参考数据格式化
    statistics   : 计算各源通道 mean/std，写入 statistics_dir

注意:
    1. GEE 需提前设置环境变量 GEE_CREDENTIALS_PATH + GEE_SERVICE_ACCOUNT
       或运行 earthengine authenticate
    2. Planetary Computer 需设置 PC_SDK_SUBSCRIPTION_KEY（可免费申请）
    3. 本脚本只修改 /workspace/raw/ 和 /workspace/statistics/，不影响训练代码
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 确保 /workspace/xuannv 在 Python path 中（用于 import src.data.transforms）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_region_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path) as f:
        cfg = json.load(f)
    # 简单验证必需字段
    required = ["region_name", "bbox", "time_range", "patch", "output_dir",
                "statistics_dir", "sources"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"配置缺少必需字段: {missing}")
    return cfg


def parse_steps(steps_str: str) -> list[str]:
    return [s.strip() for s in steps_str.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="xuannv 数据预处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="区域配置 JSON 文件路径，如 preprocessing/configs/harbin.json",
    )
    parser.add_argument(
        "--steps", "-s", default="patchify,download,cloud_filter,process,statistics",
        help="逗号分隔的步骤列表（默认全量）",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=8,
        help="并发 worker 数（默认 8）",
    )
    parser.add_argument(
        "--max-patches", type=int, default=None,
        help="可选: 限制 patchify 的最大 patch 数（用于快速调试）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的步骤，不实际执行",
    )
    args = parser.parse_args()

    # 加载配置
    region_cfg = load_region_config(args.config)
    steps = parse_steps(args.steps)

    print(f"\n{'='*60}")
    print(f"区域:   {region_cfg['display_name']} ({region_cfg['region_name']})")
    print(f"步骤:   {steps}")
    print(f"Workers: {args.workers}")
    print(f"输出目录: {region_cfg['output_dir']}")
    print(f"统计目录: {region_cfg['statistics_dir']}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("[DRY RUN] 不执行实际操作")
        return

    # 执行流水线
    from preprocessing.pipelines.orchestrator import Orchestrator

    # 将 max_patches 写入配置（orchestrator 的 patchify 会读取）
    if args.max_patches:
        region_cfg["_max_patches"] = args.max_patches

    orch = Orchestrator(region_cfg, workers=args.workers)

    # 为 patchify 传递 max_patches
    if "patchify" in steps and args.max_patches:
        from preprocessing.pipelines.patchify import run_patchify
        patches = run_patchify(region_cfg, max_patches=args.max_patches)
        steps = [s for s in steps if s != "patchify"]
        if steps:
            orch.run(steps)
    else:
        orch.run(steps)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 多区域 manifest 生成入口（独立命令）
# ---------------------------------------------------------------------------

def cmd_manifest(args: "argparse.Namespace" = None) -> None:
    """
    从多个区域配置生成合并 manifest。

    用法:
        python preprocessing/run.py manifest \
            --configs preprocessing/configs/harbin.json,preprocessing/configs/haidian.json \
            --output configs/multi_region_manifest.json
    """
    import argparse
    parser = argparse.ArgumentParser(description="生成多区域训练 manifest")
    parser.add_argument("--configs", "-c", required=True,
                        help="逗号分隔的区域配置文件列表")
    parser.add_argument("--output", "-o",
                        default="/workspace/xuannv/configs/multi_region_manifest.json")
    args = parser.parse_args(None if args is None else [])

    region_configs = [load_region_config(p.strip())
                      for p in args.configs.split(",")]

    from preprocessing.pipelines.generate_manifest import generate_manifest
    manifest = generate_manifest(region_configs, output_path=args.output)
    print(f"\n生成完成: {len(manifest)} 个区域 → {args.output}")
    for r, info in manifest.items():
        print(f"  {r}: {len(info['patches'])} patches, 源: {info['sources']}")
