# 玄女底座汇报 PPT 大纲（v4 办公室风格）

**风格锁定**：白底，无动画，每页 ≤3 个关键数字，口语化讲解  
**总页数**：11 页  
**预计时长**：30~35 分钟  

---

## P1 封面：玄女底座——自研地球嵌入基座

**页面类型**：封面页  
**视觉元素**：
- 标题：「玄女底座」大字，副标题「自研地球嵌入基座 · 阶段性汇报」
- 右下角：汇报人 + 日期
- 背景：纯白，无纹理，可放一张淡化的卫星影像底图（哈尔滨区域）

**关键数字**：无（封面不讲数字）  
**讲解钩子**：「任总，上次您说想看看咱们这块到底做成什么样了，今天给您汇报一下。」

---

## P2 模型原理：输入 → STP 编码器 → 三重损失 → 64 维输出

**页面类型**：流程图页  
**视觉元素**：
- 顶部横向流程图（4 个 box，带箭头）：
  - Input: S2 + S1 + Landsat (Time Series) — 蓝色
  - STP Encoder: Space + Time + Precision — 紫色
  - VMF Bottleneck: Skip L2 (training) / L2 + VMF (inference) — 红色
  - Output: 64-dim Embedding, 64 bytes (int8) — 绿色
- 底部两个 box：Reconstruction Loss (7 targets) + Anti-Collapse Loss (Uniformity + Decorrelation + Variance + Orthogonality) — 青色 + 橙色
- 底部一行小字：Temporal Contrastive Loss: Two non-overlapping time windows → different embeddings

**配图**：`references/charts/05_model_architecture.png`

**关键数字**：64 维、64 字节  
**讲解要点**：训练时跳过 L2 防坍缩，推理时标准 L2+VMF，时序对比损失是核心差异。

---

## P3 核心亮点：三句话讲清楚我们跟别人的区别

**页面类型**：要点页  
**视觉元素**：
- 页面分三行，每行一个图标 + 一句话 + 半句解释：
  1. 🔒 反坍缩训练 — Raw Uniformity + Decorrelation + Variance + Orthogonality，强制嵌入均匀散开
  2. ⏱ 时序敏感 — 不重叠双窗口训练，月度级变化检测，天然对时间敏感
  3. 🏗 纯自研 — 从零搭建，知识产权干净，升级完全自主可控

**关键数字**：无（概念页，不讲数字）  
**讲解要点**：三句话对应三个核心差异化，每一句都是别人没有的或做得不够好的。

---

## P4 变化检测：有 CD Head 时 BA 79.8%，高于 AEF 的 71.3%

**页面类型**：柱状图对比页  
**视觉元素**：
- 中央柱状图，三个柱子：
  - 灰色：AEF (Google DeepMind) — BA 71.3%
  - 红色：XuanNv Base + CD Head — BA 79.8%
  - 黄色：XuanNv Base (Raw Embedding) — BA ~50%
- 横轴：模型名称，纵轴：Balanced Accuracy
- 红色虚线：Random Baseline (0.5)

**配图**：`references/charts/01_cd_comparison_ba.png`

**关键数字**：79.8%、71.3%、~50%  
**讲解要点**：
- 红色柱子超过 AEF，是实打实的结果
- 黄色柱子诚实披露，不加 CD Head 时还有差距
- 这个差距不是结构性的，路线图里有补上的计划

---

## P5 竞品能力雷达：我们在时序和中国场景验证上领先

**页面类型**：雷达图页  
**视觉元素**：
- 中央雷达图，6 个维度：
  - 压缩效率、时序敏感性、无监督变化检测、多模态融合、检索工具、中国场景验证
- 三条线：
  - 灰色：AEF
  - 蓝色：Clay
  - 红色：XuanNv Base

**配图**：`references/charts/02_competitor_radar.png`

**关键数字**：3 个领先维度（时序、压缩、中国验证）  
**讲解要点**：AEF 强在通用性和多模态，但在时序和检索工具上弱；我们在时序和中国场景上填补了空白。

---

## P6 V5 评估数据分布：69 个 patch 的 BA 和 AUC 分布

**页面类型**：数据分布页  
**视觉元素**：
- 左右并置两个直方图：
  - 左：BA 分布（红色），均值 0.798，叠加 AEF 基准线 0.713（灰色虚线）
  - 右：AUC 分布（蓝色），均值 0.955
- 横轴：指标值，纵轴：patch 数量
- 图下方一行小字：Harbin 695km², 69 manually annotated patches

**配图**：`references/charts/03_v5_ba_auc_distribution.png`

**关键数字**：0.798、0.955、69 patches  
**讲解要点**：分布稳定，不是 cherry-pick 的好结果；69 个 patch 是人工标注的真实数据。

---

## P7 轻量化检索：23TB → 3GB → 16MB

**页面类型**：阶梯/对比页  
**视觉元素**：
- 中央三层阶梯图，从上到下：
  - 顶层（深色）：Raw Satellite Imagery (MajorTOM) — 23 TB
  - 中层（蓝色）：EarthEmbeddingExplorer (6 models, float32) — 3 GB
  - 底层（红色）：XuanNv Base (64-dim int8) — 16 MB
- 层间标注压缩倍数：~7,600x、~190x
- 底部注释框：「23TB → 3GB → 16MB = ~1,500,000x 压缩。任何设备可以离线运行。」
- 页面右侧或底部：预留视频播放窗口（EarthEmbeddingExplorer 演示视频）

**配图**：`references/charts/06_compression_ladder.png`

**关键数字**：23TB、3GB、16MB  
**讲解要点**：数字本身就有冲击力；16MB 比一张高清照片还小，边缘设备可用。

---

## P8 效率提升：开发周期缩短 3 倍，存储成本降低 5 倍

**页面类型**：横向柱状图页  
**视觉元素**：
- 中央横向柱状图，5 个维度：
  - Development Cycle: 6mo → 2mo = 3x
  - Human Effort: 8 → 3 people = 2.7x
  - Storage Cost: 100TB/yr → 20TB/yr = 5x
  - Coverage: 30% → 100% = 3.3x
  - Response Speed: 504hr → 24hr = 21x
- 柱子颜色：从左到右红→橙→黄→绿→蓝

**配图**：`references/charts/04_efficiency_comparison.png`

**关键数字**：3x（周期）、5x（存储）、24hr（响应）  
**讲解要点**：数字来自实际项目经验，不是理论值；存储降低是因为下游不再需要原始影像。

---

## P9 诚实面对：V5 原始嵌入 BA ~50%，与 AEF 71.3% 还有差距

**页面类型**：坦诚页 / 挑战页  
**视觉元素**：
- 页面中央大号数字对比：
  - 左侧灰色：AEF Raw BA = 71.3%
  - 右侧黄色：XuanNv V5 Raw BA = ~50%
  - 中间红色向下箭头 + 「差距 ≈ 21 个百分点」
- 下方三行原因分析（小字）：
  1. 数据量：AEF 全球级，我们目前仅哈尔滨 695km²
  2. 模型深度：AEF 参数量更大，表征能力更强
  3. 但差距非结构性，V6/V6.5 已规划补上

**关键数字**：71.3%、~50%、21pp gap  
**讲解要点**：
- 主动披露短板，建立信任
- 强调差距来源是数据量和规模，不是架构缺陷
- 路线图（P10）是解决这个问题的答案

---

## P10 升级路线图：V6 → V6.5 → V7，目标 raw BA > 70%

**页面类型**：路线图页  
**视觉元素**：
- 中央横向柱状图，四个版本：
  - V5 (Current): 红色，Raw BA ~50%，混合尺度双窗口 + CD Head
  - V6: 橙色，目标 Raw BA > 60%，Pixel-level temporal loss
  - V6.5: 黄色，目标 Raw BA > 70%，Gap-aware temporal sensing
  - V7 (Planned): 绿色，目标 Raw BA ~75%，Temporal Transformer + 全国数据
- 叠加灰色虚线：AEF Baseline (BA = 0.713)
- 每个版本上方标注关键特性（小字）

**配图**：`references/charts/07_roadmap.png`

**关键数字**：60%、70%、75%  
**讲解要点**：
- V5 已验证，V6 正在训练，V6.5 已设计，V7 是长期目标
- 每一步的目标数字清晰，可追踪
- V7 的全国级数据是商业化关键里程碑

---

## P11 总结：一张图讲清楚玄女底座的价值

**页面类型**：总结页  
**视觉元素**：
- 页面中央三行大字，每行一个关键词 + 一句话：
  1. **够轻** — 64 字节 / km²，23TB → 16MB，任何设备离线可用
  2. **够敏** — 月度级时序敏感，变化检测 BA 79.8%
  3. **够诚** — 从零自研，优势不夸大，短板不回避，路线图清晰
- 底部：「感谢任总的时间，欢迎提问。」
- 右下角：联系方式 / 二维码（可选）

**关键数字**：64 字节、79.8%  
**讲解要点**：三句话收尾，轻、敏、诚，对应技术、应用、态度三个层面。

---

## 附录：素材清单

| 页码 | 素材文件名 | 说明 |
|------|-----------|------|
| P2 | `05_model_architecture.png` | 模型原理流程图 |
| P4 | `01_cd_comparison_ba.png` | BA 对比柱状图 |
| P5 | `02_competitor_radar.png` | 竞品雷达图 |
| P6 | `03_v5_ba_auc_distribution.png` | BA/AUC 分布直方图 |
| P7 | `06_compression_ladder.png` | 压缩阶梯图 |
| P8 | `04_efficiency_comparison.png` | 效率提升横向柱状图 |
| P10 | `07_roadmap.png` | V6/V6.5/V7 升级路线图 |
| P7 | `EarthEmbddingExplorer-演示视频.mp4` | 轻量化检索演示视频 |

---

*大纲结束。所有图表均为英文标签，白底风格，可直接插入 PowerPoint 或 Keynote。*
