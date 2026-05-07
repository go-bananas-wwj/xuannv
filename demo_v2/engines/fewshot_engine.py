"""Few-Shot 变化检测引擎 — 哈尔滨新区 2025."""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from demo_v2.cache_manager import cache
from demo_v2.engines.change_detection import ChangeDetectionEngine
from demo_v2.engines.task_head_engine import TaskHeadEngine
from demo_v2.utils.harbin_annotations_v2 import (
    get_annotated_patches,
    get_period_for_patch,
    rasterize_patch_changes,
    PERIODS,
)
from demo_v2.utils.constants import RAW_DIR


class FewShotEngine:
    """哈尔滨新区 few-shot 变化检测引擎."""

    def __init__(self, version: str = "v2", device: str = "npu:0"):
        self.version = version
        self.cd_engine = ChangeDetectionEngine(version, device)
        self._annotated_patches = get_annotated_patches()
        self._task_engine = TaskHeadEngine.get_instance(device)

    def get_annotated_patches(self) -> list[str]:
        return self._annotated_patches

    def _load_s2_rgb_for_period(self, patch_id: str, start_ms: float, end_ms: float) -> np.ndarray | None:
        """读取某 patch 在指定时间窗口内的 S2 中位值 RGB 影像 [H, W, 3] uint8."""
        import rasterio
        from pathlib import Path
        from datetime import datetime

        s2_dir = RAW_DIR / "s2" / patch_id
        if not s2_dir.exists():
            return None

        files = sorted(s2_dir.glob("*.tif"))
        valid_files = []
        for f in files:
            stem = f.stem
            if len(stem) >= 8 and stem.isdigit():
                ts = float(datetime.strptime(stem[:8], "%Y%m%d").timestamp() * 1000)
                if start_ms <= ts <= end_ms:
                    valid_files.append(f)

        if not valid_files:
            candidates = []
            for f in files:
                stem = f.stem
                if len(stem) >= 8 and stem.isdigit():
                    ts = float(datetime.strptime(stem[:8], "%Y%m%d").timestamp() * 1000)
                    candidates.append((abs(ts - (start_ms + end_ms) / 2), f))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                valid_files = [candidates[0][1]]
            else:
                return None

        frames = []
        for f in valid_files[:5]:
            try:
                with rasterio.open(str(f)) as ds:
                    data = ds.read()
                if data.shape[0] >= 3:
                    rgb = data[[2, 1, 0]].astype(np.float32)
                    frames.append(rgb)
            except Exception:
                pass

        if not frames:
            return None

        img = np.median(frames, axis=0)
        valid = img[img > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                img = (img - p2) / (p98 - p2)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img.transpose(1, 2, 0)

    def _sklearn_predict(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        shot_count: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
        """运行传统 sklearn few-shot 分类."""
        rng = np.random.RandomState(42)
        classes = np.unique(labels)
        train_samples = []
        for c in classes:
            c_idx = np.where(labels == c)[0]
            n_sample = min(shot_count, len(c_idx))
            if n_sample > 0:
                sample_idx = rng.choice(c_idx, n_sample, replace=False)
                train_samples.extend(sample_idx.tolist())

        if len(train_samples) < 2 or len(np.unique(labels[train_samples])) < 2:
            return None, None, None

        X_train = features[train_samples]
        y_train = labels[train_samples]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_all_s = scaler.transform(features)

        best_clf = None
        best_auc = -1
        classifiers = [
            ("Linear", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ]
        if shot_count >= 10:
            classifiers.append(("kNN-3", KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train) // 10)))))
        else:
            classifiers.append(("kNN-1", KNeighborsClassifier(n_neighbors=1)))

        for clf_name, clf in classifiers:
            try:
                clf.fit(X_train_s, y_train)
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_all_s)[:, 1]
                else:
                    probs = clf.predict(X_all_s).astype(float)
                if len(np.unique(y_train)) >= 2:
                    train_probs = clf.predict_proba(X_train_s)[:, 1]
                    auc = roc_auc_score(y_train, train_probs)
                    if auc > best_auc:
                        best_auc = auc
                        best_clf = clf
            except Exception:
                pass

        if best_clf is None:
            return None, None, None

        if hasattr(best_clf, "predict_proba"):
            probs = best_clf.predict_proba(X_all_s)[:, 1]
        else:
            probs = best_clf.predict(X_all_s).astype(float)

        # eval metrics (same few-shot sampling)
        from sklearn.model_selection import train_test_split
        metrics = None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
            scaler_eval = StandardScaler()
            eval_samples = []
            for c in np.unique(y_tr):
                c_idx = np.where(y_tr == c)[0]
                n_sample = min(shot_count, len(c_idx))
                if n_sample > 0:
                    eval_samples.extend(rng.choice(c_idx, n_sample, replace=False).tolist())
            if len(eval_samples) >= 2 and len(np.unique(y_tr[eval_samples])) >= 2:
                X_tr_s = scaler_eval.fit_transform(X_tr[eval_samples])
                X_te_s = scaler_eval.transform(X_te)
                clf_eval = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
                clf_eval.fit(X_tr_s, y_tr[eval_samples])
                y_prob = clf_eval.predict_proba(X_te_s)[:, 1]
                y_pred = clf_eval.predict(X_te_s)
                metrics = {
                    "auc": float(roc_auc_score(y_te, y_prob)),
                    "ba": float(balanced_accuracy_score(y_te, y_pred)),
                    "f1": float(f1_score(y_te, y_pred, zero_division=0)),
                }
        except Exception:
            pass

        return probs, labels, metrics

    def detect_single_patch(
        self,
        patch_id: str,
        shot_count: int,
    ) -> dict:
        """对单个 patch 运行 few-shot 变化检测."""
        period = get_period_for_patch(patch_id)
        if period is None or period not in PERIODS:
            return {"error": f"No valid period for {patch_id}"}

        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0

        before_rgb = self._load_s2_rgb_for_period(patch_id, bs, mid)
        after_rgb = self._load_s2_rgb_for_period(patch_id, mid, be)

        emb_before = self.cd_engine.get_embedding(patch_id, bs, mid, use_precomputed=True)
        emb_after = self.cd_engine.get_embedding(patch_id, mid, be, use_precomputed=True)
        if emb_before is None or emb_after is None:
            return {"error": f"Embedding extraction failed for {patch_id}"}

        D, H, W = emb_before.shape
        gt_mask, _ = rasterize_patch_changes(patch_id, grid_size=H)

        # ── 策略：高 shot 直接用 TaskHead CD（性能更好）；低 shot 保留 sklearn few-shot 能力 ──
        if shot_count >= 50 and self._task_engine.has_cd_head:
            prob_map = self._task_engine.predict_change(emb_before, emb_after)
            if prob_map is not None:
                # 仍尝试计算 metrics（用 train/test split 模拟）
                metrics = None
                try:
                    labels_flat = gt_mask.flatten()
                    from sklearn.model_selection import train_test_split
                    idx = np.arange(len(labels_flat))
                    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=labels_flat)
                    tr_labels = labels_flat[tr_idx]
                    te_labels = labels_flat[te_idx]
                    te_probs = prob_map.flatten()[te_idx]
                    te_preds = (te_probs > 0.5).astype(int)
                    metrics = {
                        "auc": float(roc_auc_score(te_labels, te_probs)),
                        "ba": float(balanced_accuracy_score(te_labels, te_preds)),
                        "f1": float(f1_score(te_labels, te_preds, zero_division=0)),
                    }
                except Exception:
                    pass
                return {
                    "prob_map": prob_map,
                    "gt_mask": gt_mask,
                    "metrics": metrics,
                    "period": period,
                    "shot_count": shot_count,
                    "before_rgb": before_rgb,
                    "after_rgb": after_rgb,
                }

        # 低 shot：传统 sklearn
        features = np.zeros((H * W, D * 2), dtype=np.float32)
        for px in range(H):
            for py in range(W):
                idx = px * W + py
                features[idx, :D] = emb_before[:, px, py]
                features[idx, D:] = emb_after[:, px, py]

        labels = gt_mask.flatten()
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        if len(pos_idx) == 0:
            return {"error": f"No positive labels for {patch_id}"}

        n_neg_sample = min(len(pos_idx) * 10, len(neg_idx))
        if n_neg_sample == 0:
            return {"error": f"No negative samples for {patch_id}"}

        neg_sample = np.random.choice(neg_idx, n_neg_sample, replace=False)
        sampled_idx = np.concatenate([pos_idx, neg_sample])
        np.random.shuffle(sampled_idx)

        X = features[sampled_idx]
        y = labels[sampled_idx]

        probs, _, metrics = self._sklearn_predict(X, y, shot_count)
        if probs is None:
            return {"error": f"Insufficient training samples (shot={shot_count})"}

        prob_map = probs.reshape(H, W)
        return {
            "prob_map": prob_map,
            "gt_mask": gt_mask,
            "metrics": metrics,
            "period": period,
            "shot_count": shot_count,
            "before_rgb": before_rgb,
            "after_rgb": after_rgb,
        }

    def evaluate_benchmark(self, patch_id: str) -> dict:
        """对单个 patch 运行 1/10/50/100/500-shot 全套评估."""
        period = get_period_for_patch(patch_id)
        if period is None or period not in PERIODS:
            return {"error": f"No valid period for {patch_id}"}

        bs, be = PERIODS[period]
        mid = (bs + be) / 2.0
        before_rgb = self._load_s2_rgb_for_period(patch_id, bs, mid)
        after_rgb = self._load_s2_rgb_for_period(patch_id, mid, be)

        shots = [1, 10, 50, 100, 500]
        results = {}
        prob_maps = {}
        for shot in shots:
            res = self.detect_single_patch(patch_id, shot)
            if "error" in res:
                continue
            prob_maps[shot] = res["prob_map"]
            results[shot] = {
                "metrics": res.get("metrics"),
                "period": res.get("period"),
            }

        gt_mask = None
        if 500 in prob_maps:
            gt_mask = self.detect_single_patch(patch_id, 500).get("gt_mask")
        elif prob_maps:
            first_shot = list(prob_maps.keys())[0]
            gt_mask = self.detect_single_patch(patch_id, first_shot).get("gt_mask")

        return {
            "prob_maps": prob_maps,
            "results": results,
            "gt_mask": gt_mask,
            "before_rgb": before_rgb,
            "after_rgb": after_rgb,
        }
