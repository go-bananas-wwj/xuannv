# V12 训练自动监控与改进指令书

> 本文件是 Kimi CLI 自动监控任务的核心指令文档。
> 每次调用时，Kimi 应读取本文件，然后按步骤执行。

## 0. 项目上下文

- **项目路径**: `/workspace/xuannv`
- **训练脚本**: `scripts/train/train_ddp_v12.py`
- **配置**: `configs/xuannv_v12_clean.yaml`
- **输出目录**: `/workspace/outputs/xuannv_v12_clean`
- **训练日志**: `/workspace/outputs/xuannv_v12_clean/train.log`
- **状态快照**: `/workspace/xuannv/monitor_status.txt`
- **分支**: `v12-clean-dynamic`
- **GitHub**: `go-bananas-wwj/xuannv`

### 核心设计
- **模型**: AEFModel，128-dim embedding，8 STP blocks，VMF bottleneck
- **输入**: S2(6ch) + S1(2ch) + Landsat(6ch)
- **目标**: 只重建3个动态源（S2/S1/Landsat），移除所有静态目标
- **损失**: Recon(0.5) + BatchUniformity(0.3) + Consistency(0.2)
- **Batch**: 3 per GPU × 8 GPUs × accum 4 = effective batch 96
- **Memory Bank**: 512 容量，扩大 uniformity 有效 batch 到 536

### 硬件
- 8 × Huawei Ascend 910B4 NPU
- 分布式后端: hccl
- PyTorch NPU: torch 2.1.0 + torch_npu 2.1.0.post18

---

## 1. 监控任务

### 步骤 1.1: 检查训练进程是否存活

```bash
pgrep -f 'train_ddp_v12.py' | wc -l
```

- **正常**: 返回值 >= 8（8个子进程 + 1个torchrun）
- **异常**: 返回值 < 8 或返回 0

### 步骤 1.2: 读取最新训练指标

```bash
tail -20 /workspace/outputs/xuannv_v12_clean/train.log
tmux capture-pane -t v12_train -p 2>/dev/null | grep -E 'Epoch|Step' | tail -5
```

### 步骤 1.3: 检查 NPU 状态

```bash
npu-smi info
```

- **正常**: AICore > 90%，HBM < 95%
- **异常**: AICore = 0（卡死），HBM > 95%（OOM风险）

---

## 2. 判断标准

### 2.1 训练正常的标志

| 指标 | 正常范围 | 说明 |
|------|---------|------|
| `recon` | < 1.0 (warmup期), < 0.3 (稳定期) | 重建损失在下降 |
| `consist` | < 0.5 (warmup期), < 0.1 (稳定期) | student-teacher 对齐 |
| `uniform` | < 0.6 且持续下降 | **最关键指标**，0=分散好，1=坍缩坏 |
| `lr` | 在 warmup 期递增，之后 cosine 衰减 | 学习率调度正常 |
| 进程数 | >= 8 | DDP 8 卡都在运行 |
| AICore | > 90% | NPU 满负荷 |

### 2.2 训练异常的标志

| 异常类型 | 判断条件 | 严重程度 |
|---------|---------|---------|
| **进程退出** | `pgrep` 返回值 < 8 | 🔴 严重 |
| **Uniformity 坍缩** | `uniform > 0.95` 且持续上升 | 🔴 严重 |
| **Uniformity 恶化** | `uniform > 0.8` 且连续 3 个 epoch 上升 | 🟠 中度 |
| **Reconstruction 上升** | `recon` 连续 3 个 epoch 上升 | 🟠 中度 |
| **NPU OOM** | HBM > 95% 或报错 "NPU out of memory" | 🔴 严重 |
| **NaN/Inf** | 日志中出现 NaN 或 Inf | 🟠 中度 |
| **Uniformity 停滞** | `uniform > 0.6` 且 10 个 epoch 不下降 | 🟡 轻度 |

### 2.3 Uniformity 状态判断（核心）

`batch_uniformity_loss_l2` 值域 `[0, 1]`：
- **0 = 完美分散**（所有 embedding 互相正交，理想状态）
- **1 = 完全坍缩**（所有 embedding 同向，训练失败）

| Uniform 值 | 状态 | emoji | 说明 |
|-----------|------|-------|------|
| < 0.3 | 优秀 | 🟢 | embedding 充分分散，变化检测 AUC 会高 |
| 0.3~0.6 | 及格 | 🟡 | 轻度坍缩，AUC 可能 0.6~0.7 |
| 0.6~0.8 | 轻度坍缩 | 🟠 | AUC 可能 0.55~0.6 |
| 0.8~0.95 | 中度坍缩 | 🔴 | AUC 接近 0.5（随机） |
| > 0.95 | 严重坍缩 | 🚨 | 完全失败，embedding 没有任何区分度 |

**目标**: uniform < 0.3 且稳定

---

## 3. 正常时的操作

如果训练正常：
1. **不要做任何改动**
2. 汇报当前指标和趋势
3. 等待下一次检查（10分钟后）

---

## 4. 异常时的操作

### 4.1 通用流程

```
1. 读取错误日志，定位问题
2. 搜索改进方案（使用网络搜索）
3. 修改代码/配置
4. git commit + push
5. 找到最新 checkpoint
6. resume 训练
7. 验证改进效果（运行几个 epoch 观察指标）
```

### 4.2 异常处理决策树

#### 情况 A: 进程退出 / RuntimeError

1. 读取 tmux 输出最后 100 行，找到错误信息
2. 搜索该错误的解决方案
3. 修复代码
4. 用最新 checkpoint resume

#### 情况 B: Uniformity 严重坍缩 (uniform > 0.95)

**可能原因**: Memory Bank 失效 / uniformity weight 不够 / batch 仍然太小

**改进方案**:
1. 检查 Memory Bank 是否在工作（queue size > 0？）
2. 如果无效，尝试以下方案：
   - **方案 B1**: 提高 `batch_uniformity_weight` (0.3 → 0.5)
   - **方案 B2**: 降低 `reconstruction_weight` (0.5 → 0.3)
   - **方案 B3**: 实现 Hard Negative Mining（只惩罚 cos_sim > 0.3 的对）
   - **方案 B4**: 换用 `raw_uniformity_loss`（欧氏空间，不 L2 归一化）
3. 选择最可能有效的方案，修改配置
4. resume 训练

#### 情况 C: NPU OOM

1. 降低 `batch_size` (3 → 2)
2. 增加 `gradient_accumulation_steps` (4 → 6)
3. resume 训练

#### 情况 D: NaN/Inf

1. 检查 loss weights 是否过高
2. 降低 `batch_uniformity_weight`
3. 检查 `GradScaler` 是否溢出
4. resume 训练

---

## 5. 改进方案的参考文档

- **Uniformity 理论**: Wang & Isola 2020, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"
- **AEF 论文**: Brown et al. 2025, AlphaEarth Foundations (S2.2.4 Batch Uniformity)
- **Memory Bank**: He et al. 2020, Momentum Contrast (MoCo)
- **Hard Negative Mining**: 参考 SimCLR / MoCo 的 hard negative 策略

搜索关键词建议:
- "contrastive learning uniformity loss collapse"
- "embedding collapse prevention small batch"
- "memory bank contrastive learning"
- "von Mises Fisher embedding uniformity"

---

## 6. 验证清单

每次改进后，必须验证：

- [ ] 训练能正常启动（无 crash）
- [ ] Recon 继续下降
- [ ] Uniform 开始下降（或不再上升）
- [ ] Consist < 0.3 且稳定
- [ ] NPU 利用率 > 90%
- [ ] 无 NaN/Inf
- [ ] git commit + push

如果验证通过 → 继续监控
如果验证不通过 → 回滚到上次的 checkpoint，尝试其他改进方案

---

## 7. 汇报格式

每次检查后，汇报格式：

```
========== 检查时间: YYYY-MM-DD HH:MM:SS ==========
[训练状态] 正常/异常
[进程数] X/8
[最新指标] Epoch=N | recon=X.XXXX | consist=X.XXXX | uniform=X.XXXX | lr=X.XXXXXX
[Uniform 状态] 🟢/🟡/🟠/🔴/🚨 (状态说明)
[NPU 状态] AICore=X%, HMB=X%
[趋势] recon: ↑/↓/→ | uniform: ↑/↓/→
[操作] 继续监控 / 已改进并恢复 / 需要人工介入
[改进详情] (如果有)
```

---

## 8. 免责声明

- 本监控脚本**不能替代**人工检查
- 复杂 bug 仍需要人工介入
- 每次自动改进后，应通知用户查看结果
- 自动改进有试错成本，保留最近 3 个 checkpoint 以便回滚
