# 回到 V4 路径：重建 + CD Head 详细计划

## 背景

用户核心诉求：**不要花里胡哨的技巧，直接回到已被验证的 V4 路径**。

V4 的关键结果（来自项目文档）：
- Backbone Bare AUC: 0.586（backbone 本身一般）
- **CD Head AUC: 0.896**（下游头大幅提升！）

这证明了：**变化检测能力来自下游 CD Head，不是 backbone 本身**。

---

## 当前资源状态

| 资源 | 状态 |
|------|------|
| V4 checkpoint | ❌ 已不存在（`/workspace/outputs/aef_qwen_v4_official/` 为空） |
| V7 phase1 v2 checkpoint | ✅ `epoch_best_99.pt`（E99，纯重建，64-dim/6-block） |
| V7 temporal 训练 | 🔄 正在运行（tmux `v7_temporal`，E16+） |
| 标注数据 | ✅ 105 个 polygon shapefile |
| CD Head 脚本 | ⚠️ 在 archive 中，需适配 |

---

## 计划概览

```
Step 1: 停止 V7 temporal 训练（保留权重）
Step 2: 用 V7 phase1 v2 epoch_best_99.pt 提取 before/after embedding
Step 3: 冻结 backbone，训练轻量 CD Head（只改 head，不动 backbone）
Step 4: 5-fold CV 验证 AUC
Step 5: 根据结果判断方向
```

---

## Step 1: 停止当前训练（保留权重）

```bash
# 停止 tmux session
tmux kill-session -t v7_temporal

# 保留的权重
/workspace/outputs/xuannv_backbone_v7_phase1_v2/epoch_best_79.pt
/workspace/outputs/xuannv_backbone_v7_phase1_v2/epoch_best_99.pt
/workspace/outputs/xuannv_backbone_v7_temporal/  (新训练的所有 checkpoint)
```

**风险**：无。权重文件不动，只停止进程。

---

## Step 2: 提取 Embedding（Before/After）

### 2.1 选择 Backbone

使用 **V7 phase1 v2 `epoch_best_99.pt`**：
- 纯重建训练，无 temporal loss
- 64-dim embedding，6 blocks
- 与 V4 思路一致（重建驱动）

### 2.2 提取逻辑

复用 `validate_v7_level1_bare.py` 的 embedding 提取逻辑：

```python
# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)   # 2024Q3-Q4

# 对每个 patch:
emb_before = model.forward(..., valid_start=BEFORE_WINDOW[0], valid_end=BEFORE_WINDOW[1]).embedding_map
emb_after = model.forward(..., valid_start=AFTER_WINDOW[0], valid_end=AFTER_WINDOW[1]).embedding_map

# 保存为 .npy
np.save(f"{pid}_before.npy", emb_before)  # [D, H, W]
np.save(f"{pid}_after.npy", emb_after)
```

### 2.3 输出

```
/workspace/outputs/xuannv_backbone_v7_phase1_v2/cd_head_data/
├── patch_000001_before.npy   # [64, 128, 128]
├── patch_000001_after.npy
├── patch_000002_before.npy
├── patch_000002_after.npy
├── masks/
│   ├── patch_000001_mask.npy  # [128, 128], 0/1
│   └── ...
└── metadata.json
```

**预估时间**：424 patches × 2 forward ≈ 15-20 分钟（NPU）

---

## Step 3: 训练 CD Head（冻结 Backbone）

### 3.1 架构

使用 `src/models/heads.py` 中的 `ChangeDetectionHead`：

```python
class ChangeDetectionHead(nn.Module):
    """输入 before/after embedding maps [B, D, H, W]
       输出变化概率 [B, 1, H, W]
    """
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64):
        in_dim = embedding_dim * 4  # |diff|, mul, e1, e2
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.out = nn.Conv2d(hidden_dim, 1, 1)
```

**适配**：V7 是 64-dim，V4 脚本默认 128-dim，需改为 `ChangeDetectionHead(64, 64)`。

### 3.2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Backbone | 冻结 | 不训练 |
| Head 参数量 | ~50K | 轻量 |
| Batch size | 8 | 显存友好 |
| LR | 1e-3 | head 收敛快 |
| Epochs | 200 | 早停 patience=30 |
| Loss | BCE + Dice | 标准变化检测损失 |
| 数据 | 105 标注 polygon | 光栅化为 mask |
| Split | 80/20 | train/val |

### 3.3 损失函数

```python
loss = BCE(pred, mask) + Dice(pred, mask)
```

**无需 focal loss、无需 boundary aware、无需 OHEM** —— 保持简单。

### 3.4 训练脚本

写一个**自包含的、无外部依赖**的脚本：
- 不依赖 `archive/` 中的旧脚本
- 不依赖 `demo_v2/` 的复杂引擎
- 直接读取 .npy embedding + shapefile 标注

---

## Step 4: 验证（5-fold CV）

```python
# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(...)):
    # 训练 CD Head
    # 验证 AUC
    
# 输出 Mean AUC ± Std
```

### 预期结果

| 情况 | Mean AUC | 结论 |
|------|---------|------|
| AUC > 0.75 | 优秀 | 用户观点正确，专注优化 CD Head |
| AUC 0.60-0.75 | 可用 | 有提升空间，继续调 head |
| AUC < 0.60 | 失败 | backbone 时间盲，需重新训练 backbone |

**参考基线**：V4 CD Head AUC = 0.896

---

## Step 5: 根据结果判断方向

### 情况 A：AUC > 0.75（大概率）

**结论**：变化检测能力确实来自下游 CD Head，不需要 backbone 显式学习时间敏感性。

**后续**：
1. 用全部数据训练最终 CD Head
2. 集成到 demo 中
3. 停止 backbone 训练的花式改进

### 情况 B：AUC < 0.60（小概率）

**结论**：V7 phase1 v2 的 backbone 时间盲程度比 V4 严重，CD Head 无法拯救。

**后续**：
1. 分析原因（embedding 是否完全相同？）
2. 考虑重新训练 backbone（类似 V4 配置）
3. 或接受当前数据集的天花板

---

## 执行命令（批准后执行）

```bash
# Step 1: 停止当前训练
tmux kill-session -t v7_temporal

# Step 2: 提取 embedding
python scripts/eval/extract_embeddings_for_cd_head.py \
    --checkpoint /workspace/outputs/xuannv_backbone_v7_phase1_v2/epoch_best_99.pt \
    --config configs/xuannv_v7.yaml \
    --output /workspace/outputs/xuannv_backbone_v7_phase1_v2/cd_head_data

# Step 3: 训练 CD Head
python scripts/train/train_cd_head_simple.py \
    --embedding-dir /workspace/outputs/xuannv_backbone_v7_phase1_v2/cd_head_data \
    --output /workspace/outputs/cd_head_v7p1v2

# Step 4: 验证
python scripts/eval/validate_cd_head.py \
    --checkpoint /workspace/outputs/cd_head_v7p1v2/best.pt \
    --embedding-dir /workspace/outputs/xuannv_backbone_v7_phase1_v2/cd_head_data
```

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| V7 phase1 v2 backbone 时间盲 | 中 | 先跑验证，AUC 低再分析 |
| 标注数据太少（105个） | 低 | CD Head 只有 50K 参数，105 个样本够 |
| Embedding 提取耗时 | 低 | 约 20 分钟，可接受 |

---

## 时间预估

| 步骤 | 耗时 |
|------|------|
| 停止训练 + 环境准备 | 5 min |
| 提取 embedding | 20 min |
| 训练 CD Head | 30 min (200 epochs, 早停) |
| 5-fold CV 验证 | 10 min |
| **总计** | **~1 小时** |

---

*计划生成时间: 2026-05-09*
*基于: V4 评估报告、archive 中的 CD Head 脚本、当前可用 checkpoint*
