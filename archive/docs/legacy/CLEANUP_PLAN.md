# xuannv_embdding 项目目录整理计划

> **备份已创建**: `/workspace/xuannv_backup_20260417_192921.zip` (396KB，不含 logs/precomputed_cd)

---

## 一、现状问题

1. **根目录文件过多**: 5 个 `check_*.py`、4 个 `validate_*.py`、2 个 `.sh`、7 个 `.md` 报告、若干工具脚本全部散落在根目录
2. **`demo/` 与 `demo_v2/` 并存**: `demo/` 是旧版 Gradio，已被 `demo_v2/` 完全替代
3. **`scripts/` 目录臃肿**: 38 个脚本混杂训练、推理、评估、可视化、预处理、数据下载、一次性实验等
4. **报告文件散落**: 7 个 `.md` 报告全部在根目录
5. **`__pycache__` 和备份目录未清理**

---

## 二、目标目录结构

```
xuannv_embdding/
├── README.md                    # 项目说明
├── AGENTS.md                    # Agent 上下文
├── pyproject.toml               # 依赖配置
├── start_train.sh               # 训练启动脚本
├── start_gradio.sh              # Demo 启动脚本
├── validate_v2.py               # 当前活跃的变化检测验证（或移入 scripts/eval/）
│
├── configs/                     # 训练配置（保持现状）
│   └── qwen_v*.yaml
│
├── src/                         # 核心代码（保持现状）
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── demo_v2/                     # 当前活跃 Demo（精简后）
│   ├── app.py
│   ├── cache_manager.py
│   ├── config.py
│   ├── precompute_cd.py
│   ├── components/
│   ├── engines/
│   ├── utils/
│   └── precomputed_cd/          # 保留已有预计算结果
│
├── scripts/                     # 按功能分 4 个子目录
│   ├── train/
│   │   ├── train_ddp.py                 # DDP 主训练入口
│   │   ├── train_monthly_cd_head.py     # CD Head 训练
│   │   └── crossval_monthly_cd_head.py  # 交叉验证
│   ├── inference/
│   │   ├── extract_monthly_embeddings_2025.py        # 月度 embedding 提取（标注 patch）
│   │   └── extract_monthly_embeddings_all_patches.py # 全量 embedding 提取
│   ├── eval/
│   │   ├── benchmark_monthly_cd_head.py      # 基准测试
│   │   ├── evaluate_alphaearth_cd_auc.py     # AlphaEarth 对比
│   │   └── verify_annotation_alignment.py    # 标注对齐验证
│   └── visualize/
│       ├── generate_all_patch_predictions.py  # 全量预测
│       ├── stitch_regional_heatmaps.py        # 区域热力图拼接
│       └── visualize_monthly_cd_results.py    # 月度 CD 可视化
│
├── docs/                        # 报告/文档归档
│   ├── alphaearth_auc_report.md
│   ├── executive_report_failure_analysis_20250415.md
│   ├── experiment_summary_20250415.md
│   ├── optimization_plan_20250415.md
│   ├── taskhead_integration_summary_20250415.md
│   └── 项目工作总结.md
│
├── archive/                     # 废弃/实验脚本归档（不再维护，但保留备查）
│   ├── demo_legacy/             # 旧版 demo 完整目录
│   ├── scripts_experimental/    # 实验性训练/微调脚本
│   ├── scripts_preprocessing/   # 数据预处理（已完成）
│   ├── scripts_deprecated/      # 旧版验证/检查脚本
│   └── root_tools/              # 根目录旧工具脚本
│
└── logs/                        # 保持现状（运行日志）
```

---

## 三、详细迁移清单

### 3.1 删除/清理

| 路径 | 操作 | 理由 |
|------|------|------|
| `demo/` | 移入 `archive/demo_legacy/` | 旧版 Gradio，已被 demo_v2 完全替代 |
| `src_backup_20260415_003610/` | **删除** | 已有 zip 备份，且 src/ 当前为活跃代码 |
| 所有 `__pycache__/` | **删除** | Python 缓存，可随时重建 |
| `demo_v2/app.py.bak2` | **删除** | 备份文件 |
| `demo_v2/app_test.py` | 移入 `archive/` | 测试脚本，非生产代码 |
| `demo_v2/app_min.py` | 移入 `archive/` | 最小化测试版本 |
| `demo_v2/app_benchmark.py` | 移入 `archive/` | 基准测试专用版本 |
| `demo_v2/quick_precompute_v3.py` | 移入 `archive/` | 一次性快速预计算 |
| `demo_v2/quick_precompute_v3_fast.py` | 移入 `archive/` | 同上 |
| `demo_v2/precompute_v2_only.py` | 移入 `archive/` | 旧版预计算 |

### 3.2 根目录脚本 → `archive/root_tools/`

| 文件 | 说明 |
|------|------|
| `check_hr_monthly.py` | HR 数据月度检查（一次性） |
| `check_monthly_distances.py` | 月度距离计算（已验证） |
| `check_monthly_distances_10.py` | 同上变体 |
| `check_monthly_distances_fast.py` | 同上变体 |
| `check_monthly_distances_v2.py` | 同上变体 |
| `validate_change_detection.py` | 旧版变化检测验证 |
| `validate_comparison.py` | 旧版对比验证 |
| `validate_hr_only.py` | HR-only 模型验证 |
| `visualize_hr_changes.py` | HR-only 可视化 |
| `evaluate_quality.py` | Embedding 质量评估（早期实验） |
| `monitor.py` | 训练监控（早期实验） |
| `monitor_and_finetune.sh` | 监控脚本（早期实验） |

> **`validate_v2.py`** 建议保留在根目录或移入 `scripts/eval/`，它是当前活跃的变化检测验证入口。

### 3.3 scripts/ 精简 — 保留 11 个活跃脚本，归档 27 个

**保留在 `scripts/`（按功能子目录）：**

| 新路径 | 说明 | 活跃状态 |
|--------|------|----------|
| `scripts/train/train_ddp.py` | DDP 主训练入口 | ✅ 核心 |
| `scripts/train/train_monthly_cd_head.py` | CD Head 训练 | ✅ 核心 |
| `scripts/train/crossval_monthly_cd_head.py` | 交叉验证 | ✅ 核心 |
| `scripts/inference/extract_monthly_embeddings_2025.py` | 月度 embedding（标注 patch） | ✅ 常用 |
| `scripts/inference/extract_monthly_embeddings_all_patches.py` | 全量 embedding | ✅ 刚使用过 |
| `scripts/eval/benchmark_monthly_cd_head.py` | 基准测试 | ✅ 常用 |
| `scripts/eval/evaluate_alphaearth_cd_auc.py` | AlphaEarth 对比 | ✅ 常用 |
| `scripts/eval/verify_annotation_alignment.py` | 标注对齐验证 | ✅ 常用 |
| `scripts/visualize/generate_all_patch_predictions.py` | 全量预测 | ✅ 刚使用过 |
| `scripts/visualize/stitch_regional_heatmaps.py` | 区域热力图 | ✅ 刚使用过 |
| `scripts/visualize/visualize_monthly_cd_results.py` | 月度 CD 可视化 | ✅ 常用 |

**归档到 `archive/scripts_experimental/`（实验性训练脚本）：**

- `finetune_backbone_hinge.py`
- `finetune_pixel_level.py`
- `finetune_supervised.py`
- `finetune_supervised_contrastive.py`
- `finetune_with_backbone.py`
- `train_best_cv_fold.py`
- `train_mlp_downstream.py`
- `train_prototype_head.py`
- `train_task_heads.py`
- `train_task_heads_v2.py`
- `crossval_ensemble_monthly_cd.py`
- `train_monthly_cd_head_baseline.py`
- `run_cv_experiments.py`

**归档到 `archive/scripts_preprocessing/`（数据预处理，已完成）：**

- `preprocess_hr_optical.py`
- `preprocess_hr_sar.py`
- `download_alphaearth_embedding.py`
- `resume_alphaearth_2024.py`
- `precompute_alphaearth_global.py`

**归档到 `archive/scripts_deprecated/`（旧版/废弃脚本）：**

- `benchmark_all_annotated.py`
- `downstream_fewshot.py`
- `downstream_fewshot_harbin_2025.py`
- `eval_task_heads.py`
- `export_embeddings.py`
- `generate_shp_result_maps.py`
- `regenerate_mlp_maps_chinese_font.py`
- `test_task_head_integration.py`

### 3.4 根目录报告 → `docs/`

| 文件 | 说明 |
|------|------|
| `alphaearth_auc_report.md` | AlphaEarth 对比报告 |
| `executive_report_failure_analysis_20250415.md` | 失败分析 |
| `experiment_summary_20250415.md` | 实验总结 |
| `optimization_plan_20250415.md` | 优化计划 |
| `taskhead_integration_summary_20250415.md` | TaskHead 集成总结 |
| `项目工作总结.md` | 中文工作总结 |

---

## 四、预期效果

| 指标 | 整理前 | 整理后 |
|------|--------|--------|
| 根目录 `.py` 文件数 | ~15 | **1** (`validate_v2.py`) |
| 根目录 `.md` 文件数 | 7 | **2** (README + AGENTS) |
| scripts/ 脚本数 | 38 | **11** (按功能分目录) |
| 根目录整洁度 | 混乱 | **清晰** |
| 活跃代码可发现性 | 低 | **高** |

---

## 五、执行步骤（待审批后执行）

1. `mkdir -p scripts/{train,inference,eval,visualize} docs archive/{demo_legacy,scripts_experimental,scripts_preprocessing,scripts_deprecated,root_tools}`
2. 移动活跃脚本到 `scripts/` 子目录
3. 移动废弃脚本到 `archive/` 子目录
4. 移动报告到 `docs/`
5. 移动旧 demo 到 `archive/demo_legacy/`
6. 删除 `src_backup_*` 和 `__pycache__`
7. 清理 `demo_v2/` 中的备份/测试文件
8. 更新 `README.md` 和 `AGENTS.md` 中的文件路径引用

---

**请审阅此计划，确认后我将按步骤执行。**
