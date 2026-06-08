# AEF 海淀 5 源蒸馏训练计划

> **适用范围**：`aef_reference/` 工作空间  
> **目标**：在海淀区 320 个 patch 的多源遥感数据上，训练一个 64 维 embedding 模型，并通过 AEF 蒸馏对齐提升表征质量。  
> **硬件**：8 × Huawei Ascend 910B4 NPU，hccl 后端  
> **预计训练时长**：~30 小时（100000 step @ 8 卡同步）  

---

## 一、项目目标

### 1.1 核心目标
1. 训练一个输出 **64D embedding** 的深度学习模型（`AlphaEarthFoundations`）。
2. 通过 **5 源异构输入**（S1/S2/天仪SAR/Landsat/Planet）的重建任务，迫使模型学习高质量遥感语义表征。
3. 通过 **AEF 官方预训练 embedding 蒸馏**，让 Student embedding 空间与经过大规模验证的 AEF 表征对齐。
4. 最终 64D embedding 可直接用于下游任务：变化检测、地物分类（kNN/微调）等。

### 1.2 量化目标

| 指标 | 目标值 | 验收方式 |
|------|--------|----------|
| Embedding 维度 | 64D 空间向量 | 模型输出验证 |
| Student-AEF PCA RGB 结构相似度 | 目视接近 | 每 500 step 可视化图对比 |
| 重建能力 | 5 源输入 + DEM/WorldCover/DynamicWorld/JRC Water 均可重建 | 可视化图 + 损失收敛 |
| 训练步数 | 100000 step | 日志记录 |
| 下游可用性 | 64D embedding 可用于 kNN 评估 | 训练完成后运行 `knn_eval.py` |

---

## 二、模型架构

### 2.1 输入源（5 源）

| 源名 | 通道数 | 空间分辨率 | 数据路径 |
|------|--------|-----------|----------|
| `s1` (Sentinel-1) | 2 (VH+VV) | ~10m | `data_raw/haidian/scenes/{patch_id}/s1/` |
| `s2` | 6 | 10m | `data_raw/haidian/scenes/{patch_id}/s2/` |
| `tianyi_sar` (天仪SAR) | 1 | ~3m | `data_raw/haidian/scenes/{patch_id}/tianyi_sar/` |
| `landsat` | 6 | 30m → 上采样到 10m | `data_raw/haidian/scenes/{patch_id}/landsat/` |
| `planet` | 4 | ~3-5m | `data_raw/beijing/planetscene/{patch_id}/` |

### 2.2 重建目标（解码源）

| 目标 | 输入数据 | 模型预测输出 | 损失类型 | 权重 |
|------|---------|-------------|---------|------|
| `s1` | (T, H, W, 2) | (B, H, W, 2) | L1 | 1.0 |
| `s2` | (T, H, W, 6) | (B, H, W, 6) | L1 | 1.0 |
| `tianyi_sar` | (T, H, W, 1) | (B, H, W, 1) | L1 | 1.0 |
| `landsat` | (T, H, W, 6) | (B, H, W, 6) | L1 | 1.0 |
| `planet` | (T, H, W, 4) | (B, H, W, 4) | L1 | 1.0 |
| `dem` | 1 通道（高程值） | 1 通道（回归值） | L1 | 0.05 |
| `worldcover` | **1 通道（ESA 类别索引）** | **11 通道（类别 logits）** | CrossEntropy | 0.5 |
| `dynamic_world` | **1 通道（类别索引）** | **9 通道（类别 logits）** | CrossEntropy | 0.5 |
| `jrc_water` | 1 通道（水体概率/掩码） | 1 通道（回归值） | L1 | 0.3 |

### 2.3 模型配置

```python
input_sources = {
    "s1": 2, "s2": 6, "tianyi_sar": 1,
    "landsat": 6, "planet": 4,
}
decode_sources = {
    "s1": 2, "s2": 6, "tianyi_sar": 1,
    "landsat": 6, "planet": 4,
    "dem": 1, "worldcover": 11,
    "dynamic_world": 9, "jrc_water": 1,
}
per_source_latent = 32
model_size = "small"
```

---

## 三、数据准备

### 3.1 数据路径（已就绪）

```
/workspace/xuannv/data_raw/haidian/scenes/{patch_id}/
    ├── s2/
    ├── s1/
    ├── tianyi_sar/
    ├── landsat/
    ├── dem/
    ├── worldcover/
    ├── dynamic_world/
    └── jrc_water/

/workspace/xuannv/data_raw/beijing/planetscene/{patch_id}/

/workspace/xuannv/statistics/haidian/{source}_stats.json
/workspace/xuannv/data_raw/haidian/aef_embeddings/haidian_2025_patches/{patch_id}.npy
```

### 3.2 统计量

各源 mean/std 已预计算，位于 `statistics/haidian/`。注意：**WorldCover 和 Dynamic World 作为分类目标，不做 mean/std 归一化**，保持原始类别索引。

### 3.3 AEF 官方 Embedding

每个 patch 对应一个 `patch_id.npy`，shape `(64, 128, 128)`。缺失时返回 `None`，蒸馏损失自动跳过。

---

## 四、训练环境

### 4.1 硬件检查（训练前必做）

```bash
npu-smi info
```

- 确认 8 卡全部空闲（`No running processes found`）
- 确认 Health = OK，Temperature < 50°C

### 4.2 软件环境

```bash
cd /workspace/xuannv
conda activate xuannv
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
```

### 4.3 关键环境变量

`train.py` 已自动设置：
- `ASCEND_LAUNCH_BLOCKING=1`（强制 NPU op 同步，规避 SDMA 竞态）
- `ASCEND_CACHE_PATH=/tmp/ascend_cache_{local_rank}`（避免 8 卡编译缓存冲突）

---

## 五、训练启动

### 5.1 启动命令

```bash
cd /workspace/xuannv
conda activate xuannv
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=29500 \
  aef_reference/train.py \
  --batch-size 2 \
  --max-steps 100000 \
  --save-every 500 \
  --eval-every 500 \
  --distill-warmup-steps 1000 \
  --grad-accum-steps 2 \
  --log-every 50 \
  --seed 42
```

### 5.2 tmux 启动（禁止 nohup）

```bash
# 创建 session
tmux new-session -d -s aef_train -c /workspace/xuannv
tmux send-keys -t aef_train 'conda activate xuannv' Enter
tmux send-keys -t aef_train 'ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 aef_reference/train.py --batch-size 2 --max-steps 100000 --save-every 500 --eval-every 500 --distill-warmup-steps 1000 --grad-accum-steps 2 --log-every 50 --seed 42' Enter

# 查看日志
tmux capture-pane -t aef_train -p | tail -30

# 终止
tmux kill-session -t aef_train
```

### 5.3 输出目录

```
aef_reference/outputs/aef_distill_seed{seed}/
├── step_{N:06d}_seed{seed}.pt    # checkpoint
├── visualizations/                # 每 500 step 生成
│   └── viz_step_{N:06d}_{patch_id}_seed{seed}.png
└── train.log                      # 训练日志（若已配置）
```

---

## 六、两阶段训练策略

| 阶段 | 条件 | Distill 权重 | Recon 权重 | 目的 |
|------|------|-------------|-----------|------|
| **Stage 1: distill_align** | step ≤ 1000 | 5.0 | 0.1 | 先对齐 AEF 表征空间 |
| **Stage 2: normal** | step > 1000 | 0.2 | 1.0 | 再学习重建细节 |

**切换逻辑**：由 `AEFLoss.__init__()` 根据 `distill_warmup_steps` 自动切换，Trainer 每 step 调用 `loss_fn.set_stage(step)`。

---

## 七、训练过程监控

### 7.1 关键指标与正常范围

| 指标 | 正常范围 | 异常信号 | 应对措施 |
|------|----------|---------|---------|
| `recon` | < 0.3（稳定后） | warmup 后 > 0.5 | 检查数据路径、统计量 |
| `uniform` | -4.0 ~ -1.0 | **> -0.5 持续 5 step** | embedding 坍缩，检查 skip_l2_norm、uniformity_weight |
| `distill` | 逐步下降 | 突然跳变 > 10x | 检查 AEF embedding 路径 |
| `consistency` | 0.01 ~ 0.05 | > 0.1 | Teacher/Student 差异过大 |
| `active_dims` | 64/64 | < 50 | 维度坍缩，检查 uniformity |
| `lr` | 1e-4 → 0（余弦退火） | 不衰减 | 检查 scheduler step 频率 |

### 7.2 日志查看

```bash
# 实时查看
tmux capture-pane -t aef_train -p | tail -50

# 或找到输出文件
tail -f aef_reference/outputs/aef_distill_seed42/train.log
```

### 7.3 可视化检查

每 `save_every=500` step 自动生成：
- **Row 0**: 5 个输入源的中点帧（S1/S2/天仪SAR/Landsat/Planet）
- **Row 1**: Student PCA RGB | AEF PCA RGB | |Student - AEF| 差异热图

**验收标准**：
- Student PCA RGB 的空间结构应与 AEF PCA RGB 目视接近
- 差异热图不应全红（全红 = 完全不对齐）
- 随着训练进行，差异图应逐渐变淡

---

## 八、验收标准

### 8.1 训练完成标准

| 检查项 | 标准 | 验证方式 |
|--------|------|----------|
| 训练步数 | ≥ 100000 step | checkpoint 文件名 |
| 重建损失 | < 0.3 | 日志中 recon 值 |
| uniformity | -4.0 ~ -1.0 | 日志中 uniform 值 |
| 可视化对齐 | Student/AEF PCA 结构相似 | 目视检查 viz png |
| 无 NaN/Inf | 训练全程未出现 | 日志搜索 "nan" / "inf" |
| EMA 保存 | checkpoint 包含 ema_state_dict | torch.load 检查 key |

### 8.2 模型质量验收

训练完成后，运行下游评估脚本验证 embedding 质量：

```bash
# 1. 提取 embedding（示例）
python scripts/eval/extract_embeddings.py \
  --config aef_reference/configs/config.yaml \
  --checkpoint aef_reference/outputs/aef_distill_seed42/step_100000_seed42.pt

# 2. KNN 评估
python scripts/eval/knn_eval.py \
  --embedding-file outputs/eval/patch_embeddings.npz \
  --device npu:0
```

**目标**：
- WorldCover kNN 准确率 > 60%（基础）/ > 70%（良好）
- JRC Water kNN AUC > 0.75
- Dynamic World kNN 准确率 > 55%

### 8.3 Checkpoint 质量检查

```bash
python3 -c "
import torch
ckpt = torch.load('aef_reference/outputs/aef_distill_seed42/step_100000_seed42.pt', map_location='cpu')
print('step:', ckpt['step'])
print('has ema:', 'ema_state_dict' in ckpt)
print('model keys:', len(ckpt['model_state_dict']))
"
```

---

## 九、已知风险与应对

| 风险 | 现象 | 应对 |
|------|------|------|
| **NPU SDMA 竞态** | `fftsplus sdma error` | 已设置 `ASCEND_LAUNCH_BLOCKING=1` |
| **可视化 detach 崩溃** | `Can't call numpy() on Tensor that requires grad` | 已修复：所有 tensor 显式 `.detach()` |
| **分类目标归一化错误** | CE loss 报 `target out of bounds` | 已修复：分类目标跳过 normalize |
| **EMA 覆盖 embeddings** | reconstruction/embedding 来自不同参数 | 已修复：不再覆盖 `out["embeddings"]` |
| **Teacher 梯度切断** | encoder 参数不更新 | 已修复：移除 `with torch.no_grad()` |
| **3源→5源 resume 失败** | `Missing key` 报错 | 已修复：`strict=False` + 打印 missing keys |
| **Planet 填充帧** | 可视化显示全黑图 | 已修复：跳过 `abs.max() < 0.001` 的帧 |
| **hash 随机化** | 不同 run 时间戳不同 | 已修复：改用 `hashlib.md5` |
| **梯度 NaN** | loss 突然变 NaN | 检查 loss weight；不要删除 checkpoint；执行 dummy backward |

---

## 十、恢复训练

若训练中断，使用最近 checkpoint 恢复：

```bash
torchrun --nproc_per_node=8 aef_reference/train.py \
  --resume aef_reference/outputs/aef_distill_seed42/step_005000_seed42.pt \
  --batch-size 2 --max-steps 100000 --save-every 500 --eval-every 500 \
  --distill-warmup-steps 1000 --grad-accum-steps 2 --log-every 50 --seed 42
```

**注意**：
- 3 源 checkpoint（s1/s2/landsat）**不能 resume** 到 5 源模型
- resume 时 `strict=False`，缺失的 key（tianyi_sar/planet stem）会随机初始化
- EMA 状态会自动从 checkpoint 加载

---

## 十一、检查清单（训练前必做）

- [ ] `npu-smi info` 确认 8 卡空闲
- [ ] `conda activate xuannv` 环境正确
- [ ] 数据路径存在：`data_raw/haidian/scenes/`, `data_raw/beijing/planetscene/`
- [ ] 统计量存在：`statistics/haidian/`
- [ ] AEF embedding 存在：`data_raw/haidian/aef_embeddings/haidian_2025_patches/`
- [ ] 输出目录可写：`aef_reference/outputs/`
- [ ] 使用 tmux（禁止 nohup）
- [ ] 确认 `v12-clean-dynamic` 分支最新代码已 push

---

## 十二、任务分解（GOAL 模式用）

### Phase 1: 启动训练
1. 检查 NPU 占用（`npu-smi info`）
2. 激活 conda 环境
3. 使用 tmux 启动 8 卡 DDP 训练
4. 确认日志正常输出（前 100 step 无报错）

### Phase 2: 监控与调试
1. 每 500 step 检查 checkpoint 和可视化图
2. 监控 recon/uniform/distill 损失趋势
3. 若出现 NaN/Inf，立即检查损失权重，不删除 checkpoint
4. 若 AUC/对齐度不达标，调整 distill_weight 或 uniformity_weight

### Phase 3: 完成与验收
1. 训练达到 100000 step
2. 验证最终 checkpoint 包含 EMA 状态
3. 运行下游 KNN 评估
4. 生成最终报告（损失曲线、可视化图、KNN 指标）
5. 提交最终 checkpoint 和报告

---

*本文档由 AI 编码代理根据项目实际情况整理，供 GOAL 模式自动执行训练任务使用。*
