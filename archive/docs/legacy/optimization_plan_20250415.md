# xuannv_embdding 优化计划书

> 日期：2025-04-15  
> 范围：Gradio 前端展示界面优化 + 任务头微调（Task Head Finetuning）  
> 约束：**不再训练 backbone**，只新增/微调任务头

---

## 执行摘要

当前系统面临两个核心问题：
1. **前端展示层**：Gradio Demo V2 拥有 8 个 Tab，信息架构臃肿，用户认知负荷高；变化检测结果以静态热力图为主，缺少时间轴、对比视图和全局鹰眼图等交互元素。
2. **模型效果**：Backbone 训练已达瓶颈（temporal loss 的 soft collapse 问题），且训练成本极高（57M 参数、~200s/epoch）。遥感领域的大量研究表明，在预训练 backbone 固定后，通过**轻量级任务头微调**即可在下游任务（变化检测、语义分割）上获得显著提升，同时避免破坏通用表征。

本计划提出：
- **前端**：精简为 3 大核心视图，引入 KPI 卡片、交互式时间轴、分屏对比和全局鹰眼缩略图。
- **模型**：冻结全部 backbone，新增 **Change Detection Head**（像素级变化概率预测）和 **Prototype Few-Shot Head**（基于原型网络的变化分类），仅使用现有 105 个标注 polygon 进行 head 级监督训练。

---

## Part 1 — Gradio 前端展示界面优化

### 1.1 现状诊断

当前 `demo_v2/app.py` 包含 8 个 Tab：
1. 📖 Project Intro（纯文本，价值低）
2. 🗺️ Data & Embedding Field（Folium 地图浏览）
3. 🔍 Spatial Anomaly（使用率低）
4. 🔥 Change Detection（核心，但交互弱）
5. 🌆 Three-Type Change Detection（与 Change Detection 功能重叠）
6. 🌊 Downstream Tasks（WorldCover 分类演示，独立价值）
7. 📈 Performance（静态表格，更新滞后）
8. ⚖️ Model Comparison（与 Performance 重叠）

**问题**：
- 用户进入页面后需要点击 4-5 个 Tab 才能完成一次"选点 → 看时序 → 看变化"的闭环。
- Change Detection Tab 内缺少**时间维度控件**，Before/After 窗口靠下拉框选择，不够直观。
- 结果展示只有单张热力图，缺少**原始影像对比**和**变化区域高亮叠加**。
- 没有全局视角：用户无法一眼看到所有 patch 中哪些区域变化最剧烈。

### 1.2 优化目标

将 8 个 Tab 精简重组为 **3 个核心视图**：

| 新 Tab | 包含内容 | 目标 |
|---|---|---|
| **🗺️ 全域概览 (Overview)** | Folium 地图 + 鹰眼缩略图 + KPI 统计卡片 | 一眼识别高变化区域 |
| **🔍 变化检测 (Change Detection)** | 交互式时间轴 + 分屏对比 + 热力图叠加 + 时序曲线 | 深度分析单个 patch 的变化过程 |
| **🧪 下游任务 (Downstream)** | Few-shot 采样 + WorldCover 分割 + 指标仪表盘 | 展示模型在下游任务上的即战力 |

### 1.3 具体改动清单

#### A. 精简 Tab 结构 (`demo_v2/app.py`)
- **移除**：Project Intro、Spatial Anomaly、Three-Type Change Detection、Performance、Model Comparison、AlphaEarth Official
- **保留/合并**：Data & Embedding Field + Global Change Map → 新 Tab `🗺️ 全域概览`
- **保留/增强**：Change Detection → 新 Tab `🔍 变化检测`
- **保留/增强**：Downstream Tasks → 新 Tab `🧪 下游任务`

#### B. 新增/增强组件

**`demo_v2/components/overview_tab.py`（新建）**
- **左侧**：Folium 地图（保留现有 `data_browser` 的选点能力）。
- **右侧**：全局鹰眼缩略图。
  - 调用 `ChangeDetectionEngine.compute_global_change_map()` 生成全区域变化强度拼接图。
  - 在缩略图上叠加所有 patch 的边界框，并用颜色编码平均变化强度（红=高变化，蓝=低变化）。
  - 点击缩略图上的 patch 可直接跳转到 `🔍 变化检测` Tab 并自动加载该 patch。
- **顶部 KPI 卡片（3 列）**：
  - 卡片 1：当前选中 patch 的变化强度均值（数字 + 进度条）
  - 卡片 2：当前模型版本（v2 / v3）
  - 卡片 3：可检测变化类型标签（建筑工地 / 房屋拆除 / 非农非粮）

**`demo_v2/components/change_detection_tab.py`（改造）**
- **新增交互式时间轴控件**：
  - 用 `gr.Slider` 或双滑块（`gr.Range`）替代现有的 Before/After 下拉框。
  - 时间轴上标注有数据的月份刻度点，用户拖动即可定义窗口。
- **分屏对比布局**：
  ```
  ┌─────────────────┬─────────────────┐
  │  Before RGB     │  After RGB      │
  ├─────────────────┼─────────────────┤
  │  Change Heatmap │  Overlay (RGB+  │
  │                 │  Annotation)    │
  └─────────────────┴─────────────────┘
  ```
  - 底部新增**时序变化曲线**：展示该 patch 在不同相邻月份对的平均变化强度（折线图）。
- **默认选中第一个有标注的 patch**，页面加载后直接展示结果，减少用户首次空白感。

**`demo_v2/components/downstream_tab.py`（改造）**
- 将现有的 sklearn-based few-shot 面板升级为**指标仪表盘**：
  - 左侧：patch 选择器 + shot 数量滑块（1–500）
  - 右侧：3 列卡片展示 AUC / Balanced Accuracy / F1
  - 底部：prob_map + gt_mask + 错误分析图（FP/FN 高亮）

#### C. 统一视觉风格 (`demo_v2/app.py` CSS)
```css
.gradio-container {
    max-width: 1600px !important;  /* 从 1800 收窄，阅读更舒适 */
    margin: auto !important;
    font-family: "Inter", "WenQuanYi Micro Hei", sans-serif;
}
/* KPI 卡片圆角阴影 */
.kpi-card {
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px;
    background: #fafafa;
}
/* 隐藏 Gradio 底部 branding */
footer { display: none !important; }
```

### 1.4 前端实施路线图

| 阶段 | 时间 | 任务 |
|---|---|---|
| 1 | 2h | 重写 `app.py` Tab 结构；新建 `overview_tab.py`；完成全局鹰眼图 |
| 2 | 2h | 改造 `change_detection_tab.py`：加入时间轴滑块、分屏对比、时序曲线 |
| 3 | 1h | 改造 `downstream_tab.py`：指标仪表盘 + 错误分析图 |
| 4 | 1h | 联调测试、统一 CSS、性能优化（预计算全局 change map 缓存） |

---

## Part 2 — 通过微调任务头提升模型效果

### 2.1 理论基础与行业共识

根据网络检索的近期遥感论文（2024–2025），**冻结 backbone + 微调任务头**是遥感 few-shot/变化检测的标准范式：

- **GE-FSOD** (arXiv 2025)：在 few-shot 阶段冻结 Backbone，只微调 Neck 和 Detection Head，显著减少过拟合并提升新类别检测能力。
- **DBF** (arXiv 2024)：提出 Dynamic Backbone Freezing，证明在长程训练中交替冻结 backbone 可保留通用特征并学习领域特定知识。
- **Few-Shot RS Object Detection** (RS 2024)：明确指出 "During the fine-tuning stage, the backbone is frozen, while the remaining parts follow the training strategy."
- **Comparative Evaluation in Low-Data Regime** (Neurocomputing 2024)：在遥感目标检测中，将预训练 backbone 完全冻结，仅优化 YOLOv8 的 neck 和 detection head，取得优于全量微调的结果。

**核心洞察**：
- Backbone（尤其是基于 STP Transformer 的编码器）已经学到了强大的**通用时空特征**。
- 直接端到端微调容易因 temporal loss 的梯度冲突或数据分布偏移而导致**表征退化**（如本次实验中观察到的 soft collapse）。
- **Head 的参数量通常只占全模型的 1–5%**，在少量标注数据上快速收敛，且不会破坏 backbone 的预训练知识。

### 2.2 当前模型 Head 结构分析

`src/models/model.py` 中现有 heads：

```python
self.classification_head = CosineClassificationHead(m.embedding_dim, d.num_classes)
self.aux_cls_head = CosineClassificationHead(m.precision_dim, d.num_classes)
self.bottleneck_cls_head = CosineClassificationHead(m.embedding_dim, d.num_classes)
```

**问题**：
- 这些 heads 都是**全局图像分类头**（CosineClassificationHead），只能输出整张图属于 11 个地物类别的概率。
- **没有像素级变化检测头**：现有变化检测依赖 embedding 的 cosine distance，本质是无监督/自监督的，无法利用 105 个标注 polygon 的像素级监督信号。
- **没有 segmentation head**：WorldCover 分类由 reconstruction decoder 的 argmax 输出，但 decoder 参数量大、收敛慢，且与 temporal loss 存在竞争。

### 2.3 新增任务头设计方案

在**冻结全部 backbone** 的前提下，新增两个轻量级任务头：

#### Head A — Change Detection Head（像素级变化概率预测）

**架构**：轻量 UNet-style 或 2-layer MLP
```python
class ChangeDetectionHead(nn.Module):
    """输入 before/after embedding map，输出像素级变化概率 [B, 1, H, W]."""
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 通道拼接: [B, 2*D, H, W]
        self.conv1 = nn.Conv2d(embedding_dim * 2, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim // 2)
        self.conv3 = nn.Conv2d(hidden_dim // 2, 1, 1)

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb_before, emb_after], dim=1)  # [B, 2D, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return torch.sigmoid(self.conv3(x))  # [B, 1, H, W]
```

**输入**：L2-normalized `embedding_map`（before & after），尺寸 `[B, D, H, W]`。

**损失函数**：Weighted BCE + Dice Loss
```python
loss_bce = F.binary_cross_entropy(pred, mask, weight=pos_weight)
loss_dice = 1 - (2 * pred * mask).sum() / (pred + mask).sum().clamp(min=1e-6)
loss_cd = loss_bce + loss_dice
```

**数据**：直接用现有的 105 个标注 polygon 光栅化为 `[H, W]` mask 作为监督。

**预期效果**：
- 该 head 专门学习"哪些像素在 two-window 之间发生了变化"，将无监督的 cosine distance 升级为**有监督的像素级分类器**。
- 由于 backbone 固定，head 只需学会在已有时空特征空间中做**线性/浅层非线性决策边界**，收敛极快（10–30 epochs）。
- 验证 AUC 有望从 0.50 直接提升到 **0.70+**。

#### Head B — Prototype Few-Shot Head（基于原型网络的变化分类）

**动机**：现有 `fewshot_engine.py` 使用 sklearn 的 LogisticRegression/kNN 在 embedding 上训练，模型本身并不参与 few-shot 学习。Prototype Network 可以让模型在推理时自动根据 support set 计算类别原型，实现真正的**端到端 few-shot 变化检测**。

**架构**：
```python
class PrototypeChangeHead(nn.Module):
    """基于 embedding 差分向量计算变化原型."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        # 将 before/after embedding 压缩到差分空间
        self.diff_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, emb_before: torch.Tensor, emb_after: torch.Tensor):
        # emb: [B, D] 全局向量
        diff = torch.cat([emb_before, emb_after], dim=-1)
        z = self.diff_encoder(diff)  # [B, D]
        return F.normalize(z, p=2, dim=-1)

    def classify(self, z_query: torch.Tensor, z_support: torch.Tensor, y_support: torch.Tensor):
        # 计算原型
        prototypes = torch.stack([
            z_support[y_support == c].mean(0)
            for c in torch.unique(y_support)
        ])  # [n_classes, D]
        logits = z_query @ prototypes.T  # [B, n_classes]
        return logits
```

**训练方式**：
- 从 105 个标注 polygon 中采样 support/query set。
- 对变化像素（positive）和不变化像素（negative）分别计算原型。
- 使用 cross-entropy 损失优化 `diff_encoder`。

**与现有 fewshot_engine 的对比**：
- 现有方案：固定 embedding + sklearn 训练（模型不参与 few-shot 适应）
- 新方案：固定 backbone + 可学习的 diff_encoder，模型根据 support set 自动调整特征空间，**即插即用**

#### Head C — Segmentation Head（可选，快速修复 WorldCover 坍缩）

由于 HR-only small 的 WorldCover 输出完全坍缩为单一颜色，主模型的分类头也可能存在类似问题。可以新增一个轻量 Segmentation Head：

```python
class SegmentationHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, embedding_map: torch.Tensor):
        return self.conv(embedding_map)  # [B, num_classes, H, W]
```

**训练方式**：使用 WorldCover tif 标签作为像素级监督，CrossEntropy loss，只训练 head。

### 2.4 冻结策略与训练配置

**冻结范围**（`src/models/model.py` 中的全部 backbone）：
```python
for param in model.sensor_encoder_bank.parameters(): param.requires_grad = False
for param in model.stp_blocks.parameters(): param.requires_grad = False
for param in model.summary_query.parameters(): param.requires_grad = False
for param in model.bottleneck.parameters(): param.requires_grad = False
for param in model.per_source_decoders.parameters(): param.requires_grad = False
for param in model.classification_head.parameters(): param.requires_grad = False
for param in model.aux_cls_head.parameters(): param.requires_grad = False
for param in model.bottleneck_cls_head.parameters(): param.requires_grad = False
```

**可训练参数**（新增 heads）：
- `change_detection_head`
- `prototype_fs_head`
- `segmentation_head`（可选）

**优化器与超参**：
```yaml
optimizer: AdamW
lr: 0.0001          # head 可以用比 backbone 更高的 lr
weight_decay: 0.01
epochs: 50
batch_size: 8       # 只训 head，显存占用极低
checkpoint_interval: 10
```

**训练数据管道**：
- 复用现有 `HarbinPatchDataset` 的 `__getitem__`。
- 在 `trainer.py` 中增加一个 `train_head_only` 模式（或新建 `scripts/train_heads.py`）。
- 每个 batch 随机抽取 8 个 patch，对每个 patch 提取 before/after embedding，用标注 polygon 生成 mask。

### 2.5 与前端的联动

新增 heads 训练完成后，Gradio 前端需要对接：

1. **Change Detection Tab**：
   - 新增"模型模式"切换开关：
     - `Cosine Distance`（现有无监督方法）
     - `Supervised Head`（新训练的 ChangeDetectionHead）
   - 选择 `Supervised Head` 后，直接调用 head 输出概率图，叠加在 RGB 影像上。

2. **Downstream Tab**：
   - Few-shot 面板新增 `Prototype Head` 选项卡，与现有 sklearn 结果并列展示，形成对比。

### 2.6 预期效果量化

| 指标 | 当前（Backbone 训练） | 预期（Head 微调后） |
|---|---|---|
| 训练时间/epoch | ~200s（3 GPU DDP） | ~5s（单 GPU，只训 head） |
| 总训练时间 | ~8 小时（150 epochs） | ~5 分钟（50 epochs） |
| 验证 AUC | 0.50 | **0.70–0.80** |
| 像素级 IoU | 无 | **0.40–0.55** |
| 下游 F1 (few-shot) | 0.55–0.65 | **0.75+** |
| 对 backbone 的破坏 | 高（soft collapse） | **零**（完全冻结） |

---

## 实施路线图与优先级

### 第一阶段：任务头微调（高优先级，2 天出结果）
1. **Day 1 上午**：实现 `ChangeDetectionHead` + `PrototypeChangeHead`，插入 `model.py`。
2. **Day 1 下午**：编写 `scripts/train_heads.py`，配置冻结策略和数据管道。
3. **Day 2 上午**：在 GPU 5 上运行 50 epochs head 训练，监控 BCE/Dice loss 收敛。
4. **Day 2 下午**：运行 `validate_v2.py` / `validate_comparison.py`，对比 AUC 提升。

### 第二阶段：Gradio 前端改造（中优先级，3 天）
1. **Day 3**：重构 Tab 结构，完成 `overview_tab.py` 的全局鹰眼图。
2. **Day 4**：改造 `change_detection_tab.py`，加入时间轴滑块和分屏对比。
3. **Day 5**：改造 `downstream_tab.py`，对接 Prototype Head，完成指标仪表盘。

### 第三阶段：联调与展示（1 天）
1. 将训练好的 head 权重加载到 Demo 中，实现 `Cosine Distance` / `Supervised Head` 模式切换。
2. 录制演示视频或截图，准备汇报。

---

## 关键文件修改清单

| 文件 | 操作 | 内容 |
|---|---|---|
| `src/models/model.py` | 修改 | 新增 `ChangeDetectionHead`、`PrototypeChangeHead`、`SegmentationHead`；在 `AEFModel.__init__` 中实例化 |
| `src/models/heads.py` | 新建 | 存放所有新增任务头的类定义 |
| `scripts/train_heads.py` | 新建 | 冻结 backbone + 只训练 heads 的脚本 |
| `src/training/head_trainer.py` | 新建 | head-only 训练器（BCE + Dice + Prototype CE） |
| `demo_v2/app.py` | 修改 | 精简为 3 个 Tab |
| `demo_v2/components/overview_tab.py` | 新建 | 全局鹰眼图 + KPI 卡片 |
| `demo_v2/components/change_detection_tab.py` | 修改 | 时间轴滑块 + 分屏对比 + 时序曲线 |
| `demo_v2/components/downstream_tab.py` | 修改 | 指标仪表盘 + Prototype Head 选项卡 |
| `demo_v2/engines/change_detection.py` | 修改 | 支持 `supervised_head` 模式调用 |

---

## 结论

本次实验的 backbone 训练已经证明存在结构性困难（temporal InfoNCE 的 soft collapse + 巨大的训练成本）。按照遥感领域的主流实践，**最合理、最经济、最有效的下一步是立即转向"冻结 backbone + 任务头微调"**。这一方案不仅能在数小时内利用现有 105 个标注产出可用的像素级变化检测器，还能通过 Prototype Head 将 few-shot 能力真正融入模型本身，最终通过 Gradio 前端升级形成完整的产品闭环。
