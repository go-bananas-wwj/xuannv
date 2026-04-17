# xuannv 项目标准化与代码质量提升计划

> **核心约束**：
> 1. **不改 `src` 包名**，保持所有 `from src.xxx` import 不变，确保 PyTorch state_dict 兼容性。
> 2. **只改根目录名**：`/workspace/xuannv` → `/workspace/xuannv`。
> 3. **重点提升代码质量、模块化、可维护性**，方便后续升级。

---

## Phase 0: 根目录重命名（低风险）

### Step 0.1: 创建完整备份
```bash
cd /workspace && zip -r xuannv_standardization_backup_$(date +%Y%m%d_%H%M%S).zip xuannv_embdding/
```

### Step 0.2: 全局替换路径引用
在项目内将所有硬编码的 `/workspace/xuannv` 替换为 `/workspace/xuannv`：
- `.py` 文件中的 `sys.path.insert(...)`
- `.sh` 文件中的 `cd /workspace/xuannv`
- `.md` 文件中的路径说明
- `.kimi/skills/aef-qwen/SKILL.md` 中的示例路径
- `demo_v2/utils/constants.py` 中的 `PROJECT_ROOT`
- `configs/*.yaml` 中的注释路径

### Step 0.3: 重命名文件夹
```bash
cd /workspace
mv xuannv_embdding xuannv
```

### Step 0.4: 清理缓存
```bash
cd /workspace/xuannv
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

---

## Phase 1: 消除硬编码与重复逻辑（高优先级）

当前问题：大量脚本里重复写 `load_model()`, `load_head()`, `CUDA_VISIBLE_DEVICES=6` 等。

### Step 1.1: 新建 `src/utils/logging.py`
统一 Python logger，所有脚本和 demo 共用同一格式。
```python
import logging
def get_logger(name: str) -> logging.Logger: ...
```

### Step 1.2: 新建 `src/utils/checkpoint.py`
统一 checkpoint 加载/保存逻辑，支持 `map_location`、`weights_only` 安全选项。
```python
def load_checkpoint(path: Path, device: str = "cpu", keys: tuple = ("model_state_dict",)) -> dict: ...
def save_checkpoint(path: Path, epoch: int, model_state: dict, optimizer_state: dict | None = None, metrics: dict | None = None) -> None: ...
```

### Step 1.3: 新建 `src/utils/device.py`
```python
def setup_device(gpu_idx: int | None = None) -> torch.device: ...
def set_cuda_visible(gpu_idx: int | str | None = None) -> None: ...
```

### Step 1.4: 新建 `src/inference/engine.py`
把散落在 `scripts/` 和 `demo_v2/` 里的**公共推理逻辑**收进来：
- `load_backbone(config_path: str, checkpoint_path: str, device: torch.device) -> AEFModel`
- `load_cd_head(head_path: str, device: torch.device) -> ChangeDetectionHeadV3`
- `extract_monthly_embedding(model, patch_id, month, ...) -> np.ndarray`
- `run_change_detection(head, emb_before, emb_after) -> np.ndarray`

这样 `scripts/inference/`, `scripts/visualize/`, `demo_v2/engines/` 都能复用，避免重复实现。

### Step 1.5: 统一配置加载
所有脚本应通过 `src.config.load_config()` 加载配置，而不是：
```python
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"  # 坏实践
```
改为：
```python
parser.add_argument("--config", default="configs/qwen_v1_scenes.yaml")
cfg = load_config(args.config)
```
路径用相对路径，保证项目可移植。

---

## Phase 2: 重构脚本层（中优先级）

### Step 2.1: 将 `validate_v2.py` 移入 `scripts/eval/`
它本质上是一个评估脚本，不应该留在根目录。移动后更新内部 import 路径（`sys.path` 相对路径）。

### Step 2.2: 为 `scripts/` 中的脚本提取公共 CLI 参数
很多脚本都有 `--gpu_idx` / `--device` / `--config` 等重复参数。可以新建 `src/utils/cli.py`：
```python
def add_common_args(parser: argparse.ArgumentParser) -> None: ...
```
或者至少统一 argparse 风格。

### Step 2.3: 删除脚本中的 `os.environ["CUDA_VISIBLE_DEVICES"] = "6"` 硬编码
这是反模式，会全局污染 GPU 可见性。改为：
- 启动脚本时用 `CUDA_VISIBLE_DEVICES=6 python xxx.py`
- 或在脚本内部只通过 `torch.cuda.set_device(local_rank)` 选择逻辑设备

### Step 2.4: 整理 `demo_v2/engines/`
目前 `demo_v2/engines/model_engine.py`、`change_detection.py`、`patch_image_loader.py` 里有大量和 `scripts/` 重复的逻辑。重构为：
- `demo_v2/engines/` 只做 **Gradio 相关的包装和缓存管理**
- 核心模型加载、推理全部委托给 `src/inference/engine.py`

---

## Phase 3: 模型层代码规范化（中优先级）

### Step 3.1: 为 `src/models/*.py` 补全类型提示
当前 `AEFModel`, `STPBlock`, `VMFBottleneck`, `ChangeDetectionHeadV3` 等已有部分类型提示，但不够完整。统一补全：
- 所有 `__init__` 参数
- 所有 `forward` 返回值 `-> torch.Tensor`
- 所有模块级常量和函数

### Step 3.2: 为公共 API 补全 docstring
按照 Google Style 或 NumPy Style 为以下文件补全 docstring：
- `src/models/model.py` 中的 `AEFModel.encode_frames()`, `AEFModel.forward()`
- `src/models/heads.py` 中的 `ChangeDetectionHeadV3`
- `src/training/trainer.py` 中的 `DDPTrainer` 公共方法

### Step 3.3: 拆分过长的 `trainer.py`
`src/training/trainer.py` 目前可能超过 400 行。考虑拆分为：
- `src/training/trainer.py`：仅保留 `DDPTrainer` 主类
- `src/training/loops.py`：`train_one_epoch()`, `validate_epoch()` 等纯函数
- `src/training/optimizer.py`：`build_optimizer()`, `build_scheduler()`

---

## Phase 4: 数据层模块化（中优先级）

### Step 4.1: 拆分 `src/data/dataset.py`
当前 `dataset.py` 超过 600 行，包含 `HarbinPatchDataset` + 大量预处理工具函数。拆分为：
- `src/data/dataset.py`：仅保留 `HarbinPatchDataset` 类
- `src/data/transforms.py`：所有预处理函数（log transform, sigma clip, class mapping 等）
- `src/data/builder.py`：`build_dataloader()` 工厂函数

### Step 4.2: 统一 `DataModule` 风格（可选）
新建 `src/data/datamodule.py`，封装 dataset + dataloader 的创建逻辑，让 `train_ddp.py` 更简洁：
```python
class HarbinDataModule:
    def __init__(self, cfg): ...
    def train_dataloader(self): ...
    def val_dataloader(self): ...
```

---

## Phase 5: 工程化配置（低优先级，但长期收益高）

### Step 5.1: 完善 `pyproject.toml`
添加开发工具链配置：
```toml
[tool.ruff]
line-length = 120
select = ["E", "F", "I", "W", "UP"]

[tool.black]
line-length = 120

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

### Step 5.2: 添加 `.gitignore`（如果缺失）
```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
outputs/
logs/*.log
archive/
```

### Step 5.3: 统一代码格式化
用 `ruff format`（或 `black`）格式化整个 `src/` 和 `scripts/`。由于只是格式变动，不影响逻辑和模型效果。

---

## Phase 6: 验证与测试（必须）

### Step 6.1: 基础 import 测试
```bash
cd /workspace/xuannv
python -c "from src.inference.engine import load_backbone, load_cd_head; print('OK')"
python -c "from src.data.datamodule import HarbinDataModule; print('OK')"
```

### Step 6.2: 核心脚本 dry-run 测试
- `python scripts/train/train_ddp.py --help` 能正常解析参数
- `python scripts/eval/validate_v2.py` 能正确加载 checkpoint 并运行（不报错）
- `python scripts/inference/extract_monthly_embeddings_all_patches.py --help` 正常
- `bash start_train.sh` 命令语法无误（不实际启动训练）

### Step 6.3: checkpoint 兼容性验证
```python
from src.inference.engine import load_backbone, load_cd_head
# 加载旧 checkpoint
model = load_backbone("configs/qwen_v1_scenes.yaml", "/workspace/outputs/xuannv_embdding_v2/epoch_499.pt", "cpu")
head = load_cd_head("/workspace/outputs/xuannv_embdding_v2/monthly_cd_head/best_cv_fold0_v3_ohem_head.pt", "cpu")
print("Checkpoint compatible!")
```

---

## 执行顺序与优先级

```
Phase 0 (重命名) → Phase 1 (消除硬编码) → Phase 2 (脚本重构) → Phase 3 (模型规范) → Phase 4 (数据拆分) → Phase 5 (工程化) → Phase 6 (验证)
```

**建议分批次执行**：
- **第一批**：Phase 0 + Phase 1 + Phase 2（路径正确 + 消除重复逻辑 = 最大收益）
- **第二批**：Phase 3 + Phase 4（纯代码结构优化，不影响模型）
- **第三批**：Phase 5（格式化 + 配置，低风险）

---

## 回滚方案

如果出现 import 错误或其他异常：
```bash
cd /workspace
rm -rf xuannv
# 从最新备份恢复
unzip xuannv_standardization_backup_YYYYMMDD_HHMMSS.zip
mv xuannv_embdding xuannv  # 或保持原样
```

---

**请审阅此计划，确认后回复：**
- **"执行第一批"**（重命名 + 消除硬编码 + 脚本重构）
- **"执行全部"**（一次性做完所有 Phase）
- 或提出修改意见
