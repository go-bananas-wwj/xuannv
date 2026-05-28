"""流水线编排器。

按步骤顺序调度下载、处理、切 patch、统计等任务。

可执行的步骤:
    patchify    : 生成 patches_meta.json
    download    : 从配置指定的数据源下载所有启用的源
    import      : 导入本地数据（如海淀 SAR）
    cloud_filter: S2 云筛选
    process     : 后处理（S1 dB 转换、Landsat SR 转换、参考数据规范化）
    statistics  : 计算通道统计量

典型工作流:
    全量: patchify → download → cloud_filter → process → statistics
    仅统计: statistics
    仅导入 SAR: import → process → statistics
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from preprocessing.utils.logging import get_logger

logger = get_logger(__name__)

ALL_STEPS = [
    "patchify",
    "download",
    "import",
    "cloud_filter",
    "process",
    "statistics",
    "manifest",     # 生成多区域训练 manifest（需要所有区域都已完成 patchify）
]


class Orchestrator:
    """流水线编排器。"""

    def __init__(self, region_cfg: dict, workers: int = 8) -> None:
        self.region_cfg = region_cfg
        self.workers = workers
        self.region_name = region_cfg["region_name"]

    def run(self, steps: list[str]) -> dict:
        """
        按顺序执行指定步骤。

        Args:
            steps: 步骤名列表，按 ALL_STEPS 顺序执行

        Returns:
            各步骤结果的汇总字典
        """
        # 验证步骤名
        unknown = [s for s in steps if s not in ALL_STEPS]
        if unknown:
            raise ValueError(f"未知步骤: {unknown}. 可用步骤: {ALL_STEPS}")

        # 按规范顺序排序
        ordered = [s for s in ALL_STEPS if s in steps]
        logger.info(
            f"[Orchestrator/{self.region_name}] 开始执行步骤: {ordered}"
        )

        reports: dict[str, dict] = {}
        patches: list[dict] = []

        for step in ordered:
            t0 = time.time()
            logger.info(f"\n{'='*50}\n[Step: {step}]\n{'='*50}")

            if step == "patchify":
                from preprocessing.pipelines.patchify import run_patchify
                patches = run_patchify(self.region_cfg)
                reports[step] = {"n_patches": len(patches)}

            elif step in ("download", "import"):
                if not patches:
                    from preprocessing.pipelines.patchify import load_patches
                    patches = load_patches(self.region_cfg)
                step_stats = self._run_download(patches, import_only=(step == "import"))
                reports[step] = step_stats

            elif step == "cloud_filter":
                from preprocessing.processors.s2 import S2CloudFilter
                s2_cfg = self.region_cfg["sources"].get("s2", {})
                if s2_cfg.get("enabled") and s2_cfg.get("cloud_filter", {}).get("enabled"):
                    cf = S2CloudFilter(self.region_cfg)
                    reports[step] = cf.run(workers=self.workers)
                else:
                    logger.info("  S2 云筛选未启用，跳过")
                    reports[step] = {"skipped": True}

            elif step == "process":
                reports[step] = self._run_process()

            elif step == "statistics":
                from preprocessing.pipelines.statistics import run_statistics
                reports[step] = run_statistics(self.region_cfg, workers=self.workers)

            elif step == "manifest":
                from preprocessing.pipelines.generate_manifest import generate_manifest
                # manifest 步骤需要在所有区域均已完成 patchify 后执行
                # 传入当前区域配置（单区域模式），多区域需从 run.py 调用
                result = generate_manifest([self.region_cfg])
                reports[step] = {"regions": list(result.keys()), "ok": True}

            elapsed = time.time() - t0
            logger.info(f"[Step: {step}] 耗时 {elapsed:.1f}s")

        # 保存流水线执行报告
        report_path = Path(self.region_cfg["output_dir"]) / "pipeline_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=str))
        logger.info(f"\n[Orchestrator/{self.region_name}] 完成 | 报告: {report_path}")
        return reports

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _run_download(self, patches: list[dict], import_only: bool = False) -> dict:
        """下载所有启用且 download_source 匹配的数据源。"""
        from preprocessing.downloaders import get_downloader

        all_stats: dict[str, dict] = {}
        for src_name, src_cfg in self.region_cfg["sources"].items():
            if not src_cfg.get("enabled", False):
                continue
            dl_src = src_cfg.get("download_source", "")
            # import 步骤只处理 local 源，download 步骤只处理远端源
            if import_only and dl_src != "local":
                continue
            if not import_only and dl_src == "local":
                continue

            try:
                downloader = get_downloader(self.region_cfg, src_name)
                stats = downloader.download(patches, workers=self.workers)
                all_stats[src_name] = stats
            except Exception as e:
                logger.error(f"  [{src_name}] 下载失败: {e}")
                all_stats[src_name] = {"error": str(e)}

        return all_stats

    def _run_process(self) -> dict:
        """对所有启用源执行后处理。"""
        from preprocessing.processors import get_processor

        all_stats: dict[str, dict] = {}
        for src_name, src_cfg in self.region_cfg["sources"].items():
            if not src_cfg.get("enabled", False):
                continue
            try:
                processor = get_processor(self.region_cfg, src_name)
                stats = processor.run(workers=self.workers)
                all_stats[src_name] = stats
            except ValueError:
                pass  # 该源无对应处理器（如 sar 已由 local_importer 完成）
            except Exception as e:
                logger.error(f"  [{src_name}] 处理失败: {e}")
                all_stats[src_name] = {"error": str(e)}

        return all_stats
