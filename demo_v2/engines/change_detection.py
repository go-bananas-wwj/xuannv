"""变化检测引擎 — 支持预计算 embedding 和实时推理."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from demo_v2.cache_manager import cache
from demo_v2.engines.model_engine import ModelEngine
from demo_v2.engines.task_head_engine import TaskHeadEngine


# 全局 ModelEngine 缓存
_MODEL_ENGINE_CACHE: dict[str, ModelEngine] = {}


class ChangeDetectionEngine:
    """变化检测引擎."""

    def __init__(self, version: str, device: str = "cuda:0"):
        self.version = version
        self._device = device

    def _get_model_engine(self) -> ModelEngine:
        if self.version not in _MODEL_ENGINE_CACHE:
            _MODEL_ENGINE_CACHE[self.version] = ModelEngine(self.version, self._device)
        return _MODEL_ENGINE_CACHE[self.version]

    def get_embedding(
        self,
        patch_id: str,
        window_start_ms: float,
        window_end_ms: float,
        use_precomputed: bool = True,
    ) -> Optional[np.ndarray]:
        """获取 embedding map，优先使用预计算，否则实时推理."""
        # 检查是否有预计算的 embedding maps
        # FIX: 预计算 embedding 只对应数据集默认时间窗口，
        # 对于自定义窗口必须实时推理，否则 before/after 会拿到同一个 embedding
        if use_precomputed and self.version in cache.embedding_maps:
            emb = cache.get_embedding_map(self.version, patch_id)
            if emb is not None:
                # 只有当请求的是默认数据集时间窗口时才使用预计算
                # 目前预计算默认窗口是 2023-01-01 ~ 2026-03-01
                # 这里简化处理：如果窗口偏离默认值超过 1 天，强制实时推理
                default_start = 1672531200000.0  # 2023-01-01
                default_end = 1709251200000.0    # 2024-03-01
                if abs(window_start_ms - default_start) < 86400000 and abs(window_end_ms - default_end) < 86400000:
                    return emb
        # 实时推理
        return self._get_model_engine().extract_embedding(patch_id, window_start_ms, window_end_ms)

    def compute_change_score(
        self,
        patch_id: str,
        before_window: Tuple[float, float],
        after_window: Tuple[float, float],
        use_precomputed: bool = True,
        use_task_head: bool = True,
    ) -> Optional[np.ndarray]:
        """计算单 patch 的变化强度图 [H, W]，值域 [0, 1]."""
        emb_before = self.get_embedding(patch_id, before_window[0], before_window[1], use_precomputed)
        emb_after = self.get_embedding(patch_id, after_window[0], after_window[1], use_precomputed)
        if emb_before is None or emb_after is None:
            return None

        # 优先使用训练好的 task head
        if use_task_head:
            task_engine = TaskHeadEngine.get_instance(self._device)
            if task_engine.has_cd_head:
                prob = task_engine.predict_change(emb_before, emb_after)
                if prob is not None:
                    return prob

        # 回退到原始 cosine distance
        D, H, W = emb_before.shape
        fb = emb_before.reshape(D, -1)
        fa = emb_after.reshape(D, -1)

        # L2 normalize
        nb = np.linalg.norm(fb, axis=0, keepdims=True)
        na = np.linalg.norm(fa, axis=0, keepdims=True)
        fb = fb / np.maximum(nb, 1e-8)
        fa = fa / np.maximum(na, 1e-8)

        cos_sim = np.sum(fb * fa, axis=0)
        change_score = ((1.0 - cos_sim) / 2.0).reshape(H, W)
        return change_score

    def _compute_batch_with_head(
        self,
        records: list,
        before_window: Tuple[float, float],
        after_window: Tuple[float, float],
        x_to_col: dict,
        y_to_row: dict,
        H: int,
        W: int,
        batch_size: int = 16,
    ) -> Tuple[Optional[np.ndarray], list[float], float]:
        """使用 TaskHead 批量 GPU 推理全区域变化图."""
        import time
        import torch
        task_engine = TaskHeadEngine.get_instance(self._device)
        if not task_engine.has_cd_head:
            return None, [], 0.0

        canvas = np.zeros((len(y_to_row) * H, len(x_to_col) * W), dtype=np.float32)
        all_dists = []
        start = time.time()

        # 分批收集 embedding
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            eb_list, ea_list, coords = [], [], []
            for bounds, pid, _ in batch:
                emb_before = self.get_embedding(pid, before_window[0], before_window[1])
                emb_after = self.get_embedding(pid, after_window[0], after_window[1])
                if emb_before is None or emb_after is None:
                    continue
                eb_list.append(emb_before)
                ea_list.append(emb_after)
                col = x_to_col.get(round(bounds[0]))
                row = y_to_row.get(round(bounds[1]))
                coords.append((row, col))

            if not eb_list:
                continue

            eb = torch.from_numpy(np.stack(eb_list, axis=0)).float().to(self._device)
            ea = torch.from_numpy(np.stack(ea_list, axis=0)).float().to(self._device)
            with torch.no_grad():
                probs = torch.sigmoid(task_engine._cd_head(eb, ea)).squeeze(1).cpu().numpy()

            for (row, col), score in zip(coords, probs):
                if row is None or col is None:
                    continue
                r0, c0 = row * H, col * W
                canvas[r0:r0 + H, c0:c0 + W] = score
                all_dists.append(score.mean())

        elapsed = time.time() - start
        return canvas, all_dists, elapsed

    def compute_global_change_map(
        self,
        before_window: Tuple[float, float],
        after_window: Tuple[float, float],
        max_patches: Optional[int] = None,
        use_precomputed: bool = True,
    ) -> Tuple[Optional[np.ndarray], str]:
        """计算全区域变化强度拼接图 [nrows*H, ncols*W] 和统计信息."""
        ids = cache.embedding_map_patch_ids.get(self.version, cache.patch_ids)
        if not ids:
            return None, "❌ 无可用 patch 数据"

        # 计算网格布局
        records = []
        for i, pid in enumerate(ids):
            meta = cache.get_meta(pid)
            if meta is None:
                continue
            records.append((meta.bounds, pid, i))

        if not records:
            return None, "❌ 无法计算网格布局"

        all_x = sorted({round(b[0]) for b, _, _ in records})
        all_y = sorted({round(b[1]) for b, _, _ in records}, reverse=True)
        x_to_col = {x: c for c, x in enumerate(all_x)}
        y_to_row = {y: r for r, y in enumerate(all_y)}
        nrows = len(all_y)
        ncols = len(all_x)

        # 获取单 patch 尺寸
        sample_emb = cache.get_embedding_map(self.version, records[0][1])
        if sample_emb is None:
            sample_emb = self.get_embedding(records[0][1], before_window[0], before_window[1], use_precomputed)
        if sample_emb is None:
            return None, "❌ 无法获取 embedding 尺寸"
        H, W = sample_emb.shape[1], sample_emb.shape[2]

        n_limit = max_patches if max_patches else len(records)
        records = records[:n_limit]

        # 尝试批量 TaskHead 推理（当有预计算 embedding 且 head 可用时最快）
        if use_precomputed and self.version in cache.embedding_maps:
            canvas, all_dists, elapsed = self._compute_batch_with_head(
                records, before_window, after_window, x_to_col, y_to_row, H, W, batch_size=16
            )
            if canvas is not None and all_dists:
                msg = (
                    f"✅ 计算完成 ({elapsed:.1f}s) [TaskHead 批量 GPU]\n\n"
                    f"| 指标 | 值 |\n|------|-----|\n"
                    f"| 计算 patch 数 | {len(all_dists)} |\n"
                    f"| 平均变化强度 | {np.mean(all_dists):.4f} |\n"
                    f"| 最大 patch 均值 | {np.max(all_dists):.4f} |\n"
                    f"| 最小 patch 均值 | {np.min(all_dists):.4f} |\n\n"
                    f"🔴 暖色 = 高变化 | ⚫ 冷色 = 低变化"
                )
                return canvas, msg

        # 回退到串行计算
        canvas = np.zeros((nrows * H, ncols * W), dtype=np.float32)
        all_dists = []
        import time
        start = time.time()

        for bounds, pid, _ in records:
            col = x_to_col.get(round(bounds[0]))
            row = y_to_row.get(round(bounds[1]))
            if col is None or row is None:
                continue

            score = self.compute_change_score(pid, before_window, after_window, use_precomputed)
            if score is None:
                continue

            r0, c0 = row * H, col * W
            canvas[r0:r0 + H, c0:c0 + W] = score
            all_dists.append(score.mean())

        elapsed = time.time() - start
        if not all_dists:
            return None, "❌ 所有 patch 计算失败"

        msg = (
            f"✅ 计算完成 ({elapsed:.1f}s)\n\n"
            f"| 指标 | 值 |\n|------|-----|\n"
            f"| 计算 patch 数 | {len(all_dists)} |\n"
            f"| 平均变化强度 | {np.mean(all_dists):.4f} |\n"
            f"| 最大 patch 均值 | {np.max(all_dists):.4f} |\n"
            f"| 最小 patch 均值 | {np.min(all_dists):.4f} |\n\n"
            f"🔴 暖色 = 高变化 | ⚫ 冷色 = 低变化"
        )
        return canvas, msg
