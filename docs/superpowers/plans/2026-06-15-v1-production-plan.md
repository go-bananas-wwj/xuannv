# v1 Haidian 生产包 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将表现最好的 `exp_multires_v1_0612_backup/epoch_80.pt` 封装为独立的 `production/v1_haidian/` 目录，提供海淀 6 任务、WorldCover kNN、哈尔滨双时相变化检测三类生产推理能力。

**Architecture:** 在 `production/v1_haidian/xuannv_v1/` 下新建一层轻量 Python 包装模块，屏蔽原始 `src/` 复杂接口；`scripts/` 提供一键运行脚本；`tests/` 提供基于 CPU 的冒烟测试。`model/` 存放配置与 checkpoint（大文件不入 git，由 `copy_model.sh` 生成）。

**Tech Stack:** Python 3.11, PyTorch 2.1, torch_npu 2.1, sklearn (LogisticRegression / MLPClassifier / KNeighborsClassifier), rasterio, geopandas, Pillow.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `production/v1_haidian/.gitignore` | 排除 `model/epoch_80.pt` 与 `outputs/` |
| `production/v1_haidian/README.md` | 使用说明与快速开始 |
| `production/v1_haidian/model/config_multires_v1.yaml` | 复制 `configs/config_multires_v1.yaml` |
| `production/v1_haidian/model/epoch_80.pt` | 复制 checkpoint（由 `copy_model.sh` 生成，不入 git） |
| `production/v1_haidian/model/CHECKPOINT_SOURCE` | 记录权重来源（由 `copy_model.sh` 生成） |
| `production/v1_haidian/xuannv_v1/__init__.py` | 包入口，导出公共 API |
| `production/v1_haidian/xuannv_v1/backbone.py` | 模型加载 + embedding 提取 |
| `production/v1_haidian/xuannv_v1/haidian_tasks.py` | 海淀 6 任务训练/推理 |
| `production/v1_haidian/xuannv_v1/worldcover_knn.py` | WorldCover kNN 语义分割 |
| `production/v1_haidian/xuannv_v1/changedetection.py` | 双时相变化检测 |
| `production/v1_haidian/scripts/copy_model.sh` | 复制 config + checkpoint |
| `production/v1_haidian/scripts/run_haidian.sh` | 一键跑海淀 6 任务 |
| `production/v1_haidian/scripts/run_worldcover.sh` | 一键跑 WorldCover |
| `production/v1_haidian/scripts/run_changedetection.sh` | 一键跑变化检测 |
| `production/v1_haidian/tests/test_backbone.py` | backbone 冒烟测试 |
| `production/v1_haidian/tests/test_haidian_tasks.py` | 海淀任务冒烟测试 |
| `production/v1_haidian/tests/test_changedetection.py` | 变化检测冒烟测试 |

---

### Task 1: Bootstrap production directory, `.gitignore`, `README.md`, and `copy_model.sh`

**Files:**
- Create: `production/v1_haidian/.gitignore`
- Create: `production/v1_haidian/README.md`
- Create: `production/v1_haidian/scripts/copy_model.sh`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
model/epoch_80.pt
outputs/
__pycache__/
*.pyc
*.egg-info/
```

- [ ] **Step 2: Create `README.md`**

```markdown
# xuannv v1 Haidian 生产包

本目录封装在海淀标注下游任务上表现最好的 checkpoint `exp_multires_v1_0612_backup/epoch_80.pt`，
提供三类推理能力：

1. 海淀 6 个标注任务（单时相 / 双时相，Linear / MLP）
2. WorldCover kNN 语义分割
3. 哈尔滨双时相变化检测（cosine distance）

## 目录结构

```
production/v1_haidian/
├── README.md
├── model/
│   ├── config_multires_v1.yaml
│   ├── epoch_80.pt          # 由 scripts/copy_model.sh 生成，不入 git
│   └── CHECKPOINT_SOURCE    # 由 scripts/copy_model.sh 生成
├── xuannv_v1/
│   ├── __init__.py
│   ├── backbone.py
│   ├── haidian_tasks.py
│   ├── worldcover_knn.py
│   └── changedetection.py
├── scripts/
│   ├── copy_model.sh
│   ├── run_haidian.sh
│   ├── run_worldcover.sh
│   └── run_changedetection.sh
├── outputs/                  # 推理结果（gitignored）
└── tests/
    ├── test_backbone.py
    ├── test_haidian_tasks.py
    └── test_changedetection.py
```

## 快速开始

1. 复制权重：
   ```bash
   bash production/v1_haidian/scripts/copy_model.sh
   ```

2. 运行海淀 6 任务：
   ```bash
   bash production/v1_haidian/scripts/run_haidian.sh
   ```

3. 运行 WorldCover kNN：
   ```bash
   bash production/v1_haidian/scripts/run_worldcover.sh
   ```

4. 运行哈尔滨变化检测：
   ```bash
   bash production/v1_haidian/scripts/run_changedetection.sh
   ```

## 说明

- `model/epoch_80.pt` 约 1.4GB，默认不入 git。
- 脚本默认设备为 `npu:0`；如无 NPU，请在脚本中改为 `--device cpu`。
- 所有输出写入 `production/v1_haidian/outputs/`。
```

- [ ] **Step 3: Create `scripts/copy_model.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$PROD_DIR")")"

SRC_CKPT="/workspace/xuannv/outputs/exp_multires_v1_0612_backup/epoch_80.pt"
SRC_CFG="/workspace/xuannv/configs/config_multires_v1.yaml"
DEST_DIR="$PROD_DIR/model"

mkdir -p "$DEST_DIR"

if [ ! -f "$SRC_CKPT" ]; then
    echo "ERROR: 源 checkpoint 不存在: $SRC_CKPT" >&2
    exit 1
fi

cp -v "$SRC_CFG" "$DEST_DIR/config_multires_v1.yaml"
cp -v "$SRC_CKPT" "$DEST_DIR/epoch_80.pt"

{
    echo "# Checkpoint 来源记录"
    echo "source_checkpoint=$SRC_CKPT"
    echo "source_config=$SRC_CFG"
    echo "copied_at=$(date -Iseconds)"
} > "$DEST_DIR/CHECKPOINT_SOURCE"

echo "生产模型已复制到 $DEST_DIR"
```

- [ ] **Step 4: Run `copy_model.sh`**

Run:

```bash
bash production/v1_haidian/scripts/copy_model.sh
```

Expected output:

```
'configs/config_multires_v1.yaml' -> 'production/v1_haidian/model/config_multires_v1.yaml'
'outputs/exp_multires_v1_0612_backup/epoch_80.pt' -> 'production/v1_haidian/model/epoch_80.pt'
生产模型已复制到 production/v1_haidian/model
```

- [ ] **Step 5: Verify `CHECKPOINT_SOURCE`**

Run:

```bash
cat production/v1_haidian/model/CHECKPOINT_SOURCE
```

Expected: file contains `source_checkpoint=...`, `source_config=...`, `copied_at=...`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: bootstrap production/v1_haidian directory, README, copy_model.sh" && git push origin v12-clean-dynamic
```

---

### Task 2: Create `xuannv_v1/backbone.py`

**Files:**
- Create: `production/v1_haidian/xuannv_v1/__init__.py`
- Create: `production/v1_haidian/xuannv_v1/backbone.py`

- [ ] **Step 1: Create `__init__.py`**

```python
from __future__ import annotations

from xuannv_v1.backbone import (
    load_production_model,
    extract_embedding_for_month,
    extract_embeddings_for_patches,
)
from xuannv_v1.haidian_tasks import run_task, run_all_tasks
from xuannv_v1.worldcover_knn import run_worldcover_knn
from xuannv_v1.changedetection import run_change_detection

__all__ = [
    "load_production_model",
    "extract_embedding_for_month",
    "extract_embeddings_for_patches",
    "run_task",
    "run_all_tasks",
    "run_worldcover_knn",
    "run_change_detection",
]
```

- [ ] **Step 2: Create `backbone.py`**

```python
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT: Path | None = None


def _project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    return _PROJECT_ROOT


def _ensure_src_on_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_src_on_path()

from src.config import Config
from src.data.dataset import HarbinPatchDataset
from src.inference.engine import (
    load_backbone as _engine_load_backbone,
    extract_embedding_for_month as _engine_extract_embedding_for_month,
)
from src.models.model import AEFModel
from src.utils.device import get_device


def _resolve_model_dir(model_dir: str | Path) -> Path:
    md = Path(model_dir)
    if md.is_absolute():
        if not md.exists():
            raise FileNotFoundError(f"model 目录不存在: {md}")
        return md.resolve()
    if md.exists():
        return md.resolve()
    fallback = _project_root() / md
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"找不到 model 目录: {model_dir}（也尝试了 {fallback}）。"
        "请先运行 scripts/copy_model.sh。"
    )


def load_production_model(
    model_dir: str | Path = "production/v1_haidian/model",
    device: str = "npu:0",
) -> tuple[AEFModel, HarbinPatchDataset, Config]:
    md = _resolve_model_dir(model_dir)
    cfg_path = md / "config_multires_v1.yaml"
    ckpt_path = md / "epoch_80.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"缺少 checkpoint: {ckpt_path}。请运行 scripts/copy_model.sh。"
        )

    try:
        get_device(device_str=device)
    except Exception as exc:
        raise RuntimeError(
            f"设备 {device} 不可用。请设置 ASCEND_RT_VISIBLE_DEVICES 或改用 --device cpu。"
        ) from exc

    model, dataset, cfg = _engine_load_backbone(
        config_path=cfg_path,
        checkpoint_path=ckpt_path,
        device=device,
        eval_mode=True,
    )
    return model, dataset, cfg


def extract_embedding_for_month(
    model: AEFModel,
    dataset: HarbinPatchDataset,
    patch_id: str,
    year: int,
    month: int,
    device: str,
) -> np.ndarray:
    return _engine_extract_embedding_for_month(
        model=model,
        dataset=dataset,
        patch_id=patch_id,
        year=year,
        month=month,
        device=device,
        normalize=True,
        use_pre_norm=False,
    )


def extract_embeddings_for_patches(
    model: AEFModel,
    dataset: HarbinPatchDataset,
    patch_ids: list[str],
    year: int,
    month: int,
    device: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for pid in patch_ids:
        try:
            emb = _engine_extract_embedding_for_month(
                model=model,
                dataset=dataset,
                patch_id=pid,
                year=year,
                month=month,
                device=device,
                normalize=True,
                use_pre_norm=False,
            )
            out[pid] = emb
        except Exception as exc:
            warnings.warn(f"[backbone] {pid} {year}-{month:02d} 提取失败，跳过: {exc}")
    return out
```

- [ ] **Step 3: Smoke test backbone loading on CPU**

Run:

```bash
cd production/v1_haidian && python - <<'PY'
import sys
sys.path.insert(0, ".")
from xuannv_v1 import backbone
model, dataset, cfg = backbone.load_production_model(device="cpu")
emb = backbone.extract_embedding_for_month(model, dataset, "patch_000000", 2026, 4, "cpu")
print("embedding shape:", emb.shape)
PY
```

Expected:

```
embedding shape: (64, 64, 64)
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: add production v1 backbone wrapper and package init" && git push origin v12-clean-dynamic
```

---

### Task 3: Create `xuannv_v1/haidian_tasks.py` and smoke test

**Files:**
- Create: `production/v1_haidian/xuannv_v1/haidian_tasks.py`
- Create: `production/v1_haidian/tests/test_haidian_tasks.py`

- [ ] **Step 1: Create `haidian_tasks.py`**

```python
from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from . import backbone

warnings.filterwarnings("ignore")

LABEL_NORMALIZE = {
    "gongdi": "gongdi",
    "jiazhudongdi": "jianzhudongdi",
    "jianzhudongdi": "jianzhudongdi",
    "weijian": "weijian",
    "nongyongdi": "nongyongdi",
    "chachu": "chaichu",
    "chaichu": "chaichu",
    "daolubianhuo": "daolubianhua",
    "daolubianhua": "daolubianhua",
}

CLASS_NAMES = [
    "gongdi",
    "jianzhudongdi",
    "weijian",
    "nongyongdi",
    "chaichu",
    "daolubianhua",
]

CLASS_NAMES_CN = {
    "gongdi": "施工工地",
    "jianzhudongdi": "建筑用地",
    "weijian": "疑似违建",
    "nongyongdi": "农用地变化",
    "chaichu": "建筑消失",
    "daolubianhua": "施工道路",
}


def load_label_json(
    json_path: Path, image_size: tuple[int, int] = (427, 427)
) -> dict[str, np.ndarray]:
    with open(json_path) as f:
        data = json.load(f)

    h, w = image_size
    masks: dict[str, np.ndarray] = {
        name: np.zeros((h, w), dtype=np.uint8) for name in CLASS_NAMES
    }

    for shape in data.get("shapes", []):
        raw_label = shape.get("label", "").strip().lower()
        norm_label = LABEL_NORMALIZE.get(raw_label)
        if norm_label is None or norm_label not in masks:
            continue
        pts = [(int(p[0]), int(p[1])) for p in shape["points"]]
        if len(pts) < 3:
            continue
        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
        masks[norm_label] |= np.array(img, dtype=np.uint8)

    return masks


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def discover_labeled_patches(label_dir: Path) -> list[str]:
    pids: set[str] = set()
    for f in label_dir.glob("*.json"):
        m = re.search(r"(patch_\d+)", f.name)
        if m:
            pids.add(m.group(1))
    return sorted(pids)


def _extract_embeddings(
    model: Any,
    dataset: Any,
    patch_ids: list[str],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    emb_dec = backbone.extract_embeddings_for_patches(
        model, dataset, patch_ids, 2025, 12, device
    )
    emb_apr = backbone.extract_embeddings_for_patches(
        model, dataset, patch_ids, 2026, 4, device
    )
    return emb_dec, emb_apr


def run_task(
    task_name: str,
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    mode: str = "bitemporal",
    classifier: str = "linear",
    seed: int = 42,
    patch_ids: list[str] | None = None,
) -> dict[str, Any]:
    if task_name not in CLASS_NAMES:
        raise ValueError(f"未知任务: {task_name}，可选: {CLASS_NAMES}")
    if mode not in ("single", "bitemporal"):
        raise ValueError(f"未知 mode: {mode}")
    if classifier not in ("linear", "mlp"):
        raise ValueError(f"未知 classifier: {classifier}")

    out_dir = Path(output_dir) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, dataset, cfg = backbone.load_production_model(model_dir, device=device)
    label_dir = Path(label_dir)

    candidate_pids = patch_ids if patch_ids else discover_labeled_patches(label_dir)
    valid_pids = [
        p
        for p in candidate_pids
        if (label_dir / f"{p}_20260430_rgb_uint8.json").exists()
    ]

    if len(valid_pids) < 2:
        result = {"skipped": True, "reason": "带标注的 patch 不足 2 个"}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    rng = np.random.RandomState(seed)
    rng.shuffle(valid_pids)
    n_train = max(1, int(len(valid_pids) * 0.8))
    train_pids = set(valid_pids[:n_train])
    test_pids = valid_pids[n_train:]

    emb_dec, emb_apr = _extract_embeddings(model, dataset, valid_pids, device)

    X_train, y_train, X_test, y_test = [], [], [], []
    test_patch_ids: list[str] = []
    test_label_maps: list[np.ndarray] = []
    H = W = None

    for pid in valid_pids:
        if pid not in emb_apr:
            continue
        emb = emb_apr[pid]
        if mode == "bitemporal":
            if pid not in emb_dec:
                continue
            emb = np.concatenate([emb, emb_dec[pid]], axis=0)

        D, H, W = emb.shape
        json_path = label_dir / f"{pid}_20260430_rgb_uint8.json"
        masks = load_label_json(json_path, image_size=(427, 427))
        label_mask = resize_mask(masks[task_name], H)

        emb_flat = emb.reshape(D, -1).T
        label_flat = label_mask.flatten()

        if pid in train_pids:
            X_train.append(emb_flat)
            y_train.append(label_flat)
        else:
            X_test.append(emb_flat)
            y_test.append(label_flat)
            test_patch_ids.append(pid)
            test_label_maps.append(label_mask)

    if not X_train or not X_test:
        result = {"skipped": True, "reason": "训练集或测试集为空"}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    X_train = np.concatenate(X_train, 0)
    y_train = np.concatenate(y_train, 0)
    X_test = np.concatenate(X_test, 0)
    y_test = np.concatenate(y_test, 0)

    pos_ratio = y_train.mean()
    if pos_ratio < 1e-6:
        result = {"skipped": True, "reason": "训练集无正例"}
        (out_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result

    if classifier == "linear":
        clf = LogisticRegression(
            max_iter=500, n_jobs=4, class_weight="balanced", random_state=seed
        )
        clf.fit(X_train, y_train)
    else:
        sample_weight = np.where(
            y_train == 1, 1.0 / max(pos_ratio, 1e-6), 1.0 / (1 - pos_ratio)
        )
        sample_weight = sample_weight / sample_weight.mean()
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=200,
            random_state=seed,
            early_stopping=False,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    metrics: dict[str, Any] = {
        "task": task_name,
        "mode": mode,
        "classifier": classifier,
        "n_train_patches": len(train_pids),
        "n_test_patches": len(test_pids),
        "pos_ratio": float(pos_ratio),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "iou": float(jaccard_score(y_test, y_pred, zero_division=0)),
    }
    if prob is not None and len(np.unique(y_test)) == 2:
        metrics["auc"] = float(roc_auc_score(y_test, prob))
    else:
        metrics["auc"] = 0.0

    if prob is not None and H is not None and W is not None:
        prob_map = np.zeros((len(test_patch_ids), H, W), dtype=np.float32)
        label_map = np.zeros((len(test_patch_ids), H, W), dtype=np.uint8)
        offset = 0
        for idx in range(len(test_patch_ids)):
            n_pix = H * W
            prob_map[idx] = prob[offset : offset + n_pix].reshape(H, W)
            label_map[idx] = y_test[offset : offset + n_pix].reshape(H, W)
            offset += n_pix
        np.savez_compressed(
            out_dir / "pred.npz",
            patch_ids=np.array(test_patch_ids),
            prob_map=prob_map,
            label_map=label_map,
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    return metrics


def run_all_tasks(
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    mode: str = "bitemporal",
    classifier: str = "linear",
    patch_ids: list[str] | None = None,
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for task in CLASS_NAMES:
        print(f"\n[run_all_tasks] 开始任务: {task} ({CLASS_NAMES_CN[task]})")
        summary[task] = run_task(
            task,
            model_dir,
            label_dir,
            output_dir,
            device,
            mode,
            classifier,
            patch_ids=patch_ids,
        )
    (Path(output_dir) / "metrics_all.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="海淀 6 任务推理")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    parser.add_argument("--output-dir", default="outputs/haidian")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", default="bitemporal", choices=["single", "bitemporal"])
    parser.add_argument("--classifier", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_tasks(
        model_dir=args.model_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        device=args.device,
        mode=args.mode,
        classifier=args.classifier,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `tests/test_haidian_tasks.py`**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import haidian_tasks


def test_gongdi_bitemporal_linear():
    label_dir = Path("/workspace/xuannv/haidian_label/labeljson")
    pids = haidian_tasks.discover_labeled_patches(label_dir)[:5]

    metrics = haidian_tasks.run_task(
        task_name="gongdi",
        model_dir=str(PROD_DIR / "model"),
        label_dir=str(label_dir),
        output_dir=str(PROD_DIR / "outputs" / "test_haidian_tasks"),
        device="cpu",
        mode="bitemporal",
        classifier="linear",
        patch_ids=pids,
    )

    assert "auc" in metrics, f"metrics missing auc: {metrics}"
    out_path = PROD_DIR / "outputs" / "test_haidian_tasks" / "gongdi" / "metrics.json"
    assert out_path.exists(), f"metrics file not found: {out_path}"
    saved = json.loads(out_path.read_text())
    assert saved["task"] == "gongdi"
```

- [ ] **Step 3: Run the smoke test**

Run:

```bash
cd /workspace/xuannv && python -m pytest production/v1_haidian/tests/test_haidian_tasks.py -v
```

Expected: `test_gongdi_bitemporal_linear` PASSED.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: add haidian 6-task wrapper and smoke test" && git push origin v12-clean-dynamic
```

---

### Task 4: Create `xuannv_v1/worldcover_knn.py` and `run_worldcover.sh`

**Files:**
- Create: `production/v1_haidian/xuannv_v1/worldcover_knn.py`
- Create: `production/v1_haidian/scripts/run_worldcover.sh`

- [ ] **Step 1: Create `worldcover_knn.py`**

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from . import backbone

LABEL_MAPPING = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6, 90: 7}
NUM_CLASSES = len(LABEL_MAPPING)
ESA_CODES = list(LABEL_MAPPING.keys())


def _load_worldcover_label(
    patch_id: str, label_dir: Path, H: int, W: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    patch_label_dir = label_dir / patch_id / "worldcover"
    if not patch_label_dir.exists():
        return None, None
    tifs = sorted([f for f in patch_label_dir.iterdir() if f.suffix.lower() == ".tif"])
    if not tifs:
        return None, None

    with rasterio.open(tifs[0]) as src:
        label = src.read(1)
        nodata = src.nodata

    if label.shape != (H, W):
        t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
        label = (
            F.interpolate(t, size=(H, W), mode="nearest")
            .squeeze()
            .numpy()
            .astype(label.dtype)
        )

    mapped = np.full_like(label, -1, dtype=np.int64)
    for code, idx in LABEL_MAPPING.items():
        mapped[label == code] = idx
    mask = mapped >= 0
    if nodata is not None:
        mask &= label != nodata
    return mapped, mask


def run_worldcover_knn(
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    k: int = 5,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, dataset, cfg = backbone.load_production_model(model_dir, device=device)
    patch_ids = [p for p in dataset.patches]
    if not patch_ids:
        raise ValueError("数据集中没有 patch")

    embeddings = backbone.extract_embeddings_for_patches(
        model, dataset, patch_ids, 2025, 6, device
    )
    valid_pids = [p for p in patch_ids if p in embeddings]
    if not valid_pids:
        raise ValueError("没有成功提取任何 embedding")

    emb0 = embeddings[valid_pids[0]]
    D, H, W = emb0.shape

    label_dir = Path(label_dir)
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_pidx: list[np.ndarray] = []

    for pidx, pid in enumerate(valid_pids):
        if not label_dir.exists():
            break
        label, mask = _load_worldcover_label(pid, label_dir, H, W)
        if label is None or mask.sum() == 0:
            continue
        emb = embeddings[pid]
        all_X.append(emb[:, mask].T)
        all_y.append(label[mask])
        all_pidx.append(np.full(mask.sum(), pidx))

    if not all_X:
        pred_classes = -np.ones((len(valid_pids), H, W), dtype=np.int64)
        np.savez_compressed(
            out_dir / "pred_worldcover.npz",
            patch_ids=np.array(valid_pids),
            pred_classes=pred_classes,
        )
        return {
            "task": "worldcover",
            "k": k,
            "note": "未提供有效 WorldCover 标签，仅输出占位预测图。",
        }

    all_X = np.concatenate(all_X, 0)
    all_y = np.concatenate(all_y, 0)
    all_pidx = np.concatenate(all_pidx, 0)

    rng = np.random.RandomState(seed)
    n_test = max(1, int(len(valid_pids) * split_ratio))
    test_set = set(rng.choice(len(valid_pids), n_test, replace=False).tolist())
    train_mask = ~np.isin(all_pidx, list(test_set))

    X_tr, y_tr = all_X[train_mask], all_y[train_mask]
    X_te, y_te = all_X[~train_mask], all_y[~train_mask]

    if len(X_tr) == 0 or len(X_te) == 0:
        raise ValueError("训练集或测试集为空")

    clf = KNeighborsClassifier(
        n_neighbors=min(k, len(X_tr)), metric="euclidean", n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(NUM_CLASSES)))

    per_class: dict[str, dict] = {}
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        per_class[f"class_{c}"] = {
            "esa_code": ESA_CODES[c],
            "iou": float(tp / (tp + fp + fn + 1e-8)),
            "support": int(cm[c, :].sum()),
        }

    valid_ious = [v["iou"] for v in per_class.values() if v["support"] > 0]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0

    metrics = {
        "task": "worldcover",
        "k": k,
        "accuracy": float(acc),
        "mean_iou": miou,
        "num_train_pixels": int(len(X_tr)),
        "num_test_pixels": int(len(X_te)),
        "per_class": per_class,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )

    pred_maps: dict[str, np.ndarray] = {}
    for pidx in sorted(test_set):
        pid = valid_pids[pidx]
        emb = embeddings[pid]
        flat = emb.reshape(D, -1).T
        pred_flat = clf.predict(flat)
        pred_maps[pid] = pred_flat.reshape(H, W)

    if pred_maps:
        np.savez_compressed(
            out_dir / "pred_worldcover.npz",
            patch_ids=np.array(list(pred_maps.keys())),
            pred_classes=np.stack(list(pred_maps.values()), axis=0),
        )

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldCover kNN 推理")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument(
        "--label-dir",
        default="/workspace/xuannv/data_raw/haidian/scenes",
        help="包含 patch_id/worldcover/static.tif 的数据根目录；为空则仅输出占位图",
    )
    parser.add_argument("--output-dir", default="outputs/worldcover")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_worldcover_knn(
        model_dir=args.model_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        device=args.device,
        k=args.k,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `scripts/run_worldcover.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python -m xuannv_v1.worldcover_knn \
    --model-dir "$PROD_DIR/model" \
    --label-dir "/workspace/xuannv/data_raw/haidian/scenes" \
    --output-dir "$PROD_DIR/outputs/worldcover" \
    --device npu:0 \
    --k 5
```

- [ ] **Step 3: Smoke test with empty label dir**

Run:

```bash
cd production/v1_haidian && PYTHONPATH=. python -m xuannv_v1.worldcover_knn \
    --model-dir model \
    --label-dir /nonexistent \
    --output-dir outputs/test_worldcover_empty \
    --device cpu
```

Expected: creates `outputs/test_worldcover_empty/pred_worldcover.npz` and prints note about placeholder.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: add WorldCover kNN wrapper and run script" && git push origin v12-clean-dynamic
```

---

### Task 5: Create `xuannv_v1/changedetection.py` and `run_changedetection.sh`

**Files:**
- Create: `production/v1_haidian/xuannv_v1/changedetection.py`
- Create: `production/v1_haidian/scripts/run_changedetection.sh`
- Create: `production/v1_haidian/tests/test_changedetection.py`

- [ ] **Step 1: Create `changedetection.py`**

```python
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point, box
from sklearn.metrics import roc_auc_score

from . import backbone

warnings.filterwarnings("ignore")

PERIODS = {
    "june": {"before": (2025, 4), "after": (2025, 6)},
    "aug": {"before": (2025, 6), "after": (2025, 8)},
    "September": {"before": (2025, 8), "after": (2025, 9)},
    "October": {"before": (2025, 9), "after": (2025, 10)},
}


def _harbin_cfg(cfg: Any) -> Any:
    cfg.data.manifest_path = "/workspace/xuannv/data_raw/harbin/scenes"
    cfg.data.stats_dir = "/workspace/xuannv/statistics/harbin"
    cfg.data.num_samples = 424
    cfg.data.preload = True
    return cfg


def _load_grid(grid_path: Path) -> dict[str, tuple]:
    with open(grid_path) as f:
        data = json.load(f)
    bounds: dict[str, tuple] = {}
    for feat in data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bounds[pid] = (min(xs), min(ys), max(xs), max(ys))
    return bounds


def _load_changes(annot_dir: Path) -> dict[str, list]:
    period_changes: dict[str, list] = {p: [] for p in PERIODS}
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        period = shp_name.replace(".shp", "")
        path = annot_dir / shp_name
        if not path.exists():
            continue
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            warnings.warn(f"无法读取 {path}: {exc}")
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        period_changes[period] = list(gdf.geometry)
    return period_changes


def run_change_detection(
    model_dir: str,
    output_dir: str,
    device: str = "npu:0",
    periods: list[str] | None = None,
    annot_dir: str | None = None,
    grid_path: str | None = None,
    patch_limit: int | None = None,
) -> dict[str, Any]:
    if periods is None:
        periods = list(PERIODS.keys())

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _, cfg = backbone.load_production_model(model_dir, device=device)
    cfg = _harbin_cfg(cfg)

    sys.path.insert(0, str(backbone._project_root()))
    from src.data.dataset import HarbinPatchDataset

    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    patch_ids = [p for p in dataset.patches]
    if patch_limit is not None:
        patch_ids = patch_ids[:patch_limit]

    period_results: dict[str, Any] = {}
    all_scores: list[float] = []
    all_labels: list[int] = []
    all_ch_means: list[float] = []
    all_unch_means: list[float] = []

    evaluate = annot_dir is not None and grid_path is not None
    if evaluate:
        annot_dir = Path(annot_dir)
        grid_path = Path(grid_path)
        patch_bounds = _load_grid(grid_path)
        period_changes = _load_changes(annot_dir)

    for period in periods:
        if period not in PERIODS:
            raise ValueError(f"未知 period: {period}")
        pinfo = PERIODS[period]
        before_y, before_m = pinfo["before"]
        after_y, after_m = pinfo["after"]

        emb_before = backbone.extract_embeddings_for_patches(
            model, dataset, patch_ids, before_y, before_m, device
        )
        emb_after = backbone.extract_embeddings_for_patches(
            model, dataset, patch_ids, after_y, after_m, device
        )
        common_pids = [p for p in patch_ids if p in emb_before and p in emb_after]
        if not common_pids:
            continue

        scores = {
            p: 1.0 - np.sum(emb_before[p] * emb_after[p], axis=0)
            for p in common_pids
        }
        score_arr = np.stack([scores[p] for p in common_pids], axis=0)
        np.savez_compressed(
            out_dir / f"change_score_{period}.npz",
            patch_ids=np.array(common_pids),
            scores=score_arr,
        )

        if not evaluate:
            continue

        changes = period_changes.get(period, [])
        if not changes:
            continue

        annotated_pids = {
            pid
            for geom in changes
            for pid, bounds in patch_bounds.items()
            if box(*bounds).intersects(geom) and pid in common_pids
        }

        p_scores: list[float] = []
        p_labels: list[int] = []
        p_ch: list[float] = []
        p_unch: list[float] = []

        for local_pid in sorted(annotated_pids):
            if local_pid not in common_pids:
                continue
            bounds = patch_bounds[local_pid]
            minx, miny, maxx, maxy = bounds
            eb = emb_before[local_pid]
            ea = emb_after[local_pid]
            H, W = eb.shape[1], eb.shape[2]
            changed_mask = np.zeros((H, W), dtype=bool)

            for geom in changes:
                if not box(minx, miny, maxx, maxy).intersects(geom):
                    continue
                for y in range(H):
                    for x in range(W):
                        px = minx + (x + 0.5) / W * (maxx - minx)
                        py = maxy - (y + 0.5) / H * (maxy - miny)
                        if geom.buffer(1.0).contains(Point(px, py)):
                            changed_mask[y, x] = True

            dist_map = 1.0 - np.sum(eb * ea, axis=0)
            lflat = changed_mask.flatten()
            sflat = dist_map.flatten()
            if lflat.sum() == 0 or lflat.sum() == len(lflat):
                continue

            p_scores.extend(sflat.tolist())
            p_labels.extend(lflat.tolist())
            p_ch.append(float(dist_map[changed_mask].mean()))
            p_unch.append(float(dist_map[~changed_mask].mean()))

            all_scores.extend(sflat.tolist())
            all_labels.extend(lflat.tolist())
            all_ch_means.append(p_ch[-1])
            all_unch_means.append(p_unch[-1])

        if p_labels and 0 < sum(p_labels) < len(p_labels):
            period_results[period] = {
                "auc": float(roc_auc_score(p_labels, p_scores)),
                "changed_mean": float(np.mean(p_ch)),
                "unchanged_mean": float(np.mean(p_unch)),
                "separation": float(np.mean(p_ch) - np.mean(p_unch)),
                "n_samples": len(p_labels),
                "n_positive": int(sum(p_labels)),
            }

    result: dict[str, Any] = {"periods": period_results}

    if evaluate and all_labels:
        result["global"] = {
            "auc": float(roc_auc_score(all_labels, all_scores)),
            "changed_mean": float(np.mean(all_ch_means)),
            "unchanged_mean": float(np.mean(all_unch_means)),
            "separation": float(np.mean(all_ch_means) - np.mean(all_unch_means)),
        }

    (out_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="双时相变化检测")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--output-dir", default="outputs/changedetection")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--periods",
        default="june,aug,September,October",
        help="逗号分隔的 period 列表",
    )
    parser.add_argument(
        "--annot-dir",
        default="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件",
    )
    parser.add_argument(
        "--grid",
        default="/workspace/index/harbin/grid/harbin_grid.geojson",
    )
    args = parser.parse_args()

    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    run_change_detection(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        periods=periods,
        annot_dir=args.annot_dir,
        grid_path=args.grid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `scripts/run_changedetection.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python -m xuannv_v1.changedetection \
    --model-dir "$PROD_DIR/model" \
    --output-dir "$PROD_DIR/outputs/changedetection" \
    --device npu:0 \
    --periods june,aug,September,October \
    --annot-dir "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件" \
    --grid "/workspace/index/harbin/grid/harbin_grid.geojson"
```

- [ ] **Step 3: Create `tests/test_changedetection.py`**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_DIR))

from xuannv_v1 import changedetection


def test_june_change_score_shape():
    metrics = changedetection.run_change_detection(
        model_dir=str(PROD_DIR / "model"),
        output_dir=str(PROD_DIR / "outputs" / "test_changedetection"),
        device="cpu",
        periods=["june"],
        annot_dir="/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件",
        grid_path="/workspace/index/harbin/grid/harbin_grid.geojson",
        patch_limit=3,
    )

    out_path = PROD_DIR / "outputs" / "test_changedetection" / "change_score_june.npz"
    assert out_path.exists(), f"change score file not found: {out_path}"

    data = np.load(out_path)
    assert "patch_ids" in data
    assert "scores" in data
    assert data["scores"].ndim == 3

    metrics_path = PROD_DIR / "outputs" / "test_changedetection" / "metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert "june" in saved["periods"]
```

- [ ] **Step 4: Run the smoke test**

Run:

```bash
cd /workspace/xuannv && python -m pytest production/v1_haidian/tests/test_changedetection.py -v
```

Expected: `test_june_change_score_shape` PASSED.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add change detection wrapper, script, and smoke test" && git push origin v12-clean-dynamic
```

---

### Task 6: Create `scripts/run_haidian.sh` and final integration check

**Files:**
- Create: `production/v1_haidian/scripts/run_haidian.sh`

- [ ] **Step 1: Create `scripts/run_haidian.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROD_DIR"
export PYTHONPATH="$PROD_DIR:${PYTHONPATH:-}"

python -m xuannv_v1.haidian_tasks \
    --model-dir "$PROD_DIR/model" \
    --label-dir "/workspace/xuannv/haidian_label/labeljson" \
    --output-dir "$PROD_DIR/outputs/haidian" \
    --device npu:0 \
    --mode bitemporal \
    --classifier linear
```

- [ ] **Step 2: Syntax-check all shell scripts**

Run:

```bash
bash -n production/v1_haidian/scripts/copy_model.sh
bash -n production/v1_haidian/scripts/run_haidian.sh
bash -n production/v1_haidian/scripts/run_worldcover.sh
bash -n production/v1_haidian/scripts/run_changedetection.sh
```

Expected: no output (syntax OK).

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add run_haidian.sh and syntax-check all production scripts" && git push origin v12-clean-dynamic
```

---

### Task 7: Final smoke-test suite and wrap-up

- [ ] **Step 1: Run all production smoke tests**

Run:

```bash
cd /workspace/xuannv && python -m pytest production/v1_haidian/tests/ -v
```

Expected:

```
tests/test_backbone.py::test_load_and_extract PASSED
tests/test_haidian_tasks.py::test_gongdi_bitemporal_linear PASSED
tests/test_changedetection.py::test_june_change_score_shape PASSED
```

- [ ] **Step 2: Confirm git status only ignores outputs and checkpoint**

Run:

```bash
git status --short production/v1_haidian/
```

Expected: tracked files listed as staged/committed; `model/epoch_80.pt` and `outputs/` not listed.

- [ ] **Step 3: Final commit if any changes**

```bash
git add -A && git commit -m "test: add production v1 smoke tests and finalize" && git push origin v12-clean-dynamic
```

---

## Spec Coverage Self-Review

| Spec Requirement | Implementing Task |
|------------------|-------------------|
| `production/v1_haidian/` directory with listed structure | Task 1–5 |
| `backbone.py` with `load_production_model`, `extract_embedding_for_month`, `extract_embeddings_for_patches` | Task 2 |
| `haidian_tasks.py` with `run_task` / `run_all_tasks`, sklearn classifiers | Task 3 |
| `worldcover_knn.py` with sklearn kNN | Task 4 |
| `changedetection.py` with cosine distance + optional AUC | Task 5 |
| `scripts/` one-click runners | Tasks 1, 4, 5, 6 |
| `tests/` smoke tests | Tasks 2, 3, 5 |
| `.gitignore` for checkpoint and outputs | Task 1 |
| git commit commands | All tasks |
