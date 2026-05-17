---
name: aef-evaluation-pipeline
description: >
  AEF (AlphaEarth Foundations) 下游评估自动化 Pipeline。用于多实验并行 embedding 提取、
  KNN/MLP 分类评估、变化检测 AUC 计算，以及结果汇总报告生成。
  触发条件：用户提到"评估"、"下游"、"embedding 提取"、"变化检测"、"AUC"、
  "对比实验"、"调参"、"KNN"、"MLP"、"Pipeline"，或需要对多个训练实验做系统性对比时。
---

# AEF 下游评估自动化 Pipeline

## 概述

本 Pipeline 自动化执行以下流程：
1. **Embedding 提取**: 7 卡并行提取所有 patch × 12 个月的 embedding，支持断点续传
2. **KNN 评估**: 3 任务（WorldCover 11类、DynamicWorld 9类、JRC Water 2类）
3. **MLP 评估**: PixelMLPHead (hidden=256, 50 epochs)
4. **变化检测 AUC**: Cosine + LogisticRegression（4 个时期加权平均）
5. **报告生成**: Markdown 汇总表格 + 结论分析

## 核心步骤

### Step 1: 确认实验列表

检查 `/workspace/outputs/xuannv_round2/` 下的实验目录，确认每个实验有：
- `epoch_best_epoch{N}.pt` checkpoint
- `configs/round4_8gpu/{exp_name}.yaml` 配置文件

### Step 2: 并行提取 Embedding

使用 `scripts/eval/launch_all_round4_eval.sh` 启动 7 卡并行提取：

```bash
cd /workspace/xuannv
bash scripts/eval/launch_all_round4_eval.sh
```

**关键设计**:
- 每个实验独立 tmux session，分配独立 NPU
- `--device npu:0` + `ASCEND_RT_VISIBLE_DEVICES=X` 做设备映射
- 每 500 样本保存断点，崩溃后自动恢复
- 预计 25 分钟/实验

**监控**:
```bash
tmux list-sessions | grep eval_round4
tmux capture-pane -t eval_round4_full_vicreg_baseline -p | tail -10
```

### Step 3: 运行下游评估

提取完成后，使用 `scripts/eval/launch_downstream_v2.sh` 启动 KNN + MLP + CD：

```bash
cd /workspace/xuannv
bash scripts/eval/launch_downstream_v2.sh
```

**评估顺序**（每个实验串行）:
1. KNN (~5 分钟): `evaluate_knn_v2.py --k 5 --month 6`
2. MLP (~20 分钟): `evaluate_mlp_v2.py --epochs 50 --month 6`
3. CD (~5 分钟): `evaluate_cd_v2.py`

**标签映射**（必须在评估脚本中处理）:
- WorldCover: {10→0, 30→1, 40→2, 50→3, 60→4, 80→5, 90→6}
- DynamicWorld: 0-7（原生 0-based）
- JRC Water: 0-1（必须过滤掉 >1 的值，nodata=-32768）

### Step 4: 汇总结果

收集所有实验的 `knn_summary.json`、`mlp_summary.json`、`cd_v2.log`，生成 Markdown 报告：

```bash
python3 scripts/eval/generate_report.py  # 见 references/report-template.md
```

**输出位置**: `/workspace/outputs/round4_full_evaluation_report.md`

## 关键脚本路径

| 脚本 | 功能 |
|------|------|
| `scripts/eval/extract_embeddings_v2.py` | 提取 embedding（断点续传） |
| `scripts/eval/run_extraction_resilient.py` | 崩溃自动重启调度器 |
| `scripts/eval/launch_all_round4_eval.sh` | 7 卡并行提取启动器 |
| `scripts/eval/evaluate_knn_v2.py` | KNN 三任务评估 |
| `scripts/eval/evaluate_mlp_v2.py` | MLP 下游头训练 |
| `scripts/eval/evaluate_cd_v2.py` | 变化检测 AUC |
| `scripts/eval/launch_downstream_v2.sh` | 下游评估批量启动 |

## 常见问题

### 1. NPU 设备映射错误
**症状**: `open device X failed, runtime result = 107001`  
**解决**: `ASCEND_RT_VISIBLE_DEVICES=X` 时，PyTorch 内设备名必须为 `npu:0`（不是 `npu:X`）

### 2. KNN 超时
**症状**: `evaluate_knn_v2.py` 运行数分钟无输出  
**根因**: `torch.unique` 在 NPU 上 350K 次调用极慢  
**解决**: 使用 `scipy.stats.mode` 在 CPU 上批量计算众数

### 3. WorldCover 标签过滤失败
**症状**: KNN/MLP 报 `ValueError: At least one label specified must be in y_true`  
**根因**: WorldCover 标签值是 ESA 编码（10,30,40...），不是 0-based  
**解决**: 使用 `LABEL_MAPPINGS` 映射到 0-based

### 4. JRC Water MLP 结果异常低
**症状**: JRC Water MLP Acc ~6-7%  
**根因**: `prepare_data` 未过滤 `label < num_classes`，保留了 0-98 全部值  
**解决**: mask 添加 `(label < num_classes)`，传入正确的 `num_classes`

## 评估成功标准

| 指标 | 及格线 | 良好 | 优秀 |
|------|--------|------|------|
| CD AUC (LR) | > 0.55 | > 0.65 | > 0.75 |
| WC KNN Acc | > 30% | > 35% | > 40% |
| DW KNN Acc | > 40% | > 45% | > 50% |
| JW KNN Acc | > 15% | > 20% | > 25% |

## 参考文档

- **详细脚本参数**: 见 `references/evaluation-scripts.md`
- **报告模板**: 见 `references/report-template.md`
