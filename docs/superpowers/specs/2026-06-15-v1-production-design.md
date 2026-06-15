# v1 生产版封装设计文档

> **目标**：把当前在海淀标注下游任务上表现最好的 `exp_multires_v1_0612_backup/epoch_80.pt` 封装成一个可独立运行、结构清晰的生产目录 `production/v1_haidian/`，支持海淀 6 个标注任务、WorldCover kNN 语义分割、双时相变化检测三类下游推理。

---

## 1. 目录结构

```
production/v1_haidian/
├── README.md                         # 使用说明与快速开始
├── model/
│   ├── config_multires_v1.yaml       # 复制 configs/config_multires_v1.yaml
│   ├── epoch_80.pt                   # 复制 outputs/exp_multires_v1_0612_backup/epoch_80.pt（1.4GB，不入 git）
│   └── CHECKPOINT_SOURCE             # 纯文本，记录权重来源路径与复制时间
├── xuannv_v1/                        # Python 包装模块
│   ├── __init__.py
│   ├── backbone.py                   # 模型加载 + embedding 提取
│   ├── haidian_tasks.py              # 海淀 6 个标注任务推理
│   ├── worldcover_knn.py             # WorldCover kNN 语义分割
│   └── changedetection.py            # 双时相变化检测
├── scripts/                          # 一键运行脚本
│   ├── run_haidian.sh                # 跑海淀 6 任务（单时相 + 双时相）
│   ├── run_worldcover.sh             # 跑 WorldCover kNN
│   └── run_changedetection.sh        # 跑哈尔滨变化检测 AUC / 输出变化图
├── outputs/                          # 推理结果输出（gitignored）
│   ├── haidian/
│   ├── worldcover/
│   └── changedetection/
└── tests/                            # 冒烟测试
    ├── test_backbone.py
    ├── test_haidian_tasks.py
    └── test_changedetection.py
```

---

## 2. 组件设计

### 2.1 `xuannv_v1/backbone.py`

职责：屏蔽原始 `src/` 的复杂接口，提供生产友好的加载与 embedding 提取函数。

对外接口：

```python
def load_production_model(
    model_dir: str | Path = "production/v1_haidian/model",
    device: str = "npu:0",
) -> tuple[AEFModel, HarbinPatchDataset, Config]:
    """自动加载 production/v1_haidian/model 下的 config + checkpoint."""

def extract_embedding_for_month(
    model, dataset, patch_id: str, year: int, month: int, device: str
) -> np.ndarray:
    """为单个 patch 的指定年月提取 embedding map [D, H, W]."""

def extract_embeddings_for_patches(
    model, dataset, patch_ids: list[str], year: int, month: int, device: str
) -> dict[str, np.ndarray]:
    """批量提取多个 patch 的 embedding，返回 {patch_id: emb}."""
```

实现要点：
- 通过 `sys.path.insert(0, project_root)` 引入原始 `src` 模块。
- 统一使用 `src.inference.engine.load_backbone` 和 `extract_embedding_for_month`。
- 设备自动降级：如果 `npu:0` 不可用，显式报错提示用户设置 `ASCEND_RT_VISIBLE_DEVICES` 或改用 `cpu`。

### 2.2 `xuannv_v1/haidian_tasks.py`

职责：封装海淀 6 个标注任务的训练和推理。

对外接口：

```python
def run_task(
    task_name: str,                       # gongdi / jianzhudongdi / weijian / nongyongdi / chaichu / daolubianhua
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    mode: str = "bitemporal",             # "single" 用 2026-04；"bitemporal" 用 2025-12 + 2026-04
    classifier: str = "linear",           # "linear" 或 "mlp"
    seed: int = 42,
) -> dict:
    """训练并推理单个任务，返回 metrics dict."""

def run_all_tasks(
    model_dir: str,
    label_dir: str,
    output_dir: str,
    device: str = "npu:0",
    mode: str = "bitemporal",
    classifier: str = "linear",
) -> dict[str, dict]:
    """顺序跑 6 个任务，汇总 metrics."""
```

实现要点：
- 读取 `labeljson/*.json`，将 polygon 栅格化为与 embedding 对齐的 64×64 mask。
- 按 patch 分层划分 train/test（80/20），避免数据泄漏。
- 默认使用 `Linear`（LogisticRegression），因为 MLP 在极不平衡数据上不稳定。
- 输出每个任务的 `metrics.json` 和 `pred_<task>.npz`（含 prob_map、label_map、patch_ids）。

### 2.3 `xuannv_v1/worldcover_knn.py`

职责：基于已有 embedding 和 WorldCover 标签做 kNN 像素级分类。

对外接口：

```python
def run_worldcover_knn(
    model_dir: str,
    label_dir: str,                       # WorldCover tiff 目录
    output_dir: str,
    device: str = "npu:0",
    k: int = 5,
    split_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """提取 embedding，按 patch 划分 train/test，kNN 分类并返回 mIoU/OA."""
```

实现要点：
- 复用 `scripts/eval/knn_eval.py` 的核心逻辑，但封装成函数。
- 支持 `--label-dir` 为空：此时只输出预测图，不计算 metrics。
- 标签为 ESA WorldCover 编码，自动映射到 0-based。

### 2.4 `xuannv_v1/changedetection.py`

职责：双时相变化检测推理与评估。

对外接口：

```python
def run_change_detection(
    model_dir: str,
    output_dir: str,
    device: str = "npu:0",
    periods: list[str] | None = None,     # ["june", "aug", "September", "October"]
    annot_dir: str | None = None,         # 哈尔滨变化检测 shapefile 目录
    grid_path: str | None = None,         # harbin_grid.geojson
) -> dict:
    """对哈尔滨 patch 提取 before/after embedding，计算 cosine distance 变化图，可选评估 AUC."""
```

实现要点：
- 使用 `extract_embedding_map()` 按时间窗口提取 embedding。
- 变化分数 = `1 - cosine_similarity(emb_before, emb_after)` per pixel。
- 如果提供 `annot_dir` 和 `grid_path`，则按 shapefile 计算 AUC；否则只输出变化图。
- 输出 `change_score_<period>.npz` 和 `metrics.json`。

---

## 3. 数据流

1. 用户执行 `bash scripts/run_haidian.sh`。
2. `run_haidian.sh` 调用 `python -m xuannv_v1.haidian_tasks run_all_tasks ...`。
3. `haidian_tasks.py` 调用 `backbone.load_production_model()` 加载 config + checkpoint。
4. 批量提取 2025-12 和 2026-04 两期 embedding。
5. 对每个任务，按 patch 划分 train/test，训练 sklearn Linear/MLP。
6. 在 test patches 上推理，保存概率图和 metrics。

WorldCover 与变化检测流程类似，只是下游头不同。

---

## 4. 错误处理

| 场景 | 处理 |
|------|------|
| `epoch_80.pt` 不存在 | 抛出 `FileNotFoundError`，提示运行复制脚本或检查 `model/` 目录 |
| NPU 不可用 | 提示设置 `ASCEND_RT_VISIBLE_DEVICES` 或改用 `--device cpu` |
| 某 patch 某月份数据缺失 | 记录 warning，跳过该 patch，继续处理其余 |
| 某任务正例数为 0 | 跳过该任务，metrics 中标记 `skipped=True` |
| 输出目录不存在 | 自动创建 |

---

## 5. 测试策略

每个组件至少一个冒烟测试：

- `tests/test_backbone.py`：加载模型，对 `patch_000000` 提取 2026-04 embedding，断言 shape 为 `(64, 64, 64)`。
- `tests/test_haidian_tasks.py`：在 5 个 patch 上跑 `gongdi` 任务，断言输出 `metrics.json` 存在且含 `auc` 字段。
- `tests/test_changedetection.py`：对 `june` 窗口跑变化检测，断言输出变化图 shape 正确。

所有测试默认在 `cpu` 上运行，避免占用 NPU。

---

## 6. 提交与版本管理

- `model/epoch_80.pt` 加入 `.gitignore`，不提交大文件。
- `model/CHECKPOINT_SOURCE` 和 `model/config_multires_v1.yaml` 提交到 git。
- 复制权重的操作在 README 中说明，并提供复制脚本 `scripts/copy_model.sh`。

---

## 7. 后续可扩展（本次不做）

- REST API / CLI SDK 形态。
- CD Head 训练与加载（目前用 cosine distance）。
- Docker 镜像打包。
