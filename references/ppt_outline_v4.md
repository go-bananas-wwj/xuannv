# 玄女底座汇报 PPT 大纲（v4 科研风 + 去AI化）

**风格锁定**：白底，无动画，每页 ≤3 个关键数字，工程师口语  
**总页数**：11 页  
**预计时长**：30 分钟  

---

## P1 封面：玄女底座——自研地球嵌入基座

**页面类型**：封面页  
**视觉元素**：
- 标题大字：「玄女底座」
- 副标题：「自研地球嵌入基座 · 阶段性进展」
- 右下角：汇报人 + 日期
- 纯白底，无纹理，可加淡色哈尔滨卫星底图

**关键数字**：无  
**讲解钩子**：「任总，玄女底座的进展，今天聊聊。」

---

## P2 模型原理：输入 → STP 编码器 → VMF 瓶颈 → 64 维输出

**页面类型**：流程图页  
**视觉元素**：
- 顶部 4 个蓝灰色调方块，横向排列：
  - Input: S2 + S1 + Landsat (Time Series) — 深灰蓝
  - STP Encoder — 学术蓝
  - VMF Bottleneck — 中蓝
  - Output: 64-dim, 64 bytes — 灰蓝
- 底部 2 个灰蓝色方块：Reconstruction Loss + Anti-Collapse Loss
- 最底部一行小字框：Temporal Contrastive

**配图**：`references/charts/05_model_architecture.png`

**关键数字**：64 维、64 字节  
**讲解要点**：训练 skip L2 防坍缩，推理 L2+VMF，时序对比是核心差异。

---

## P3 核心亮点：反坍缩 · 时序敏感 · 纯自研

**页面类型**：三栏要点页  
**视觉元素**：
- 三行等宽排列，每行一个数字圆圈 + 标题 + 一行解释：
  1. 01 — 反坍缩训练 — Uniformity 为负，嵌入均匀散开
  2. 02 — 时序敏感 — 不重叠双窗口，月度级变化可检测
  3. 03 — 纯自研 — 代码/结构/训练流程全部自主

**关键数字**：无（概念页）  
**讲解要点**：三个差异化，每个都是别人没有或做得不够的。提到"第一次训练崩到-0.1"。

---

## P4 变化检测：BA 79.8%（含 CD Head）/ 50%（Raw）

**页面类型**：柱状图对比页  
**视觉元素**：
- 中央柱状图，三个柱子，蓝灰色系：
  - 灰色：AEF — 0.713
  - 深蓝：XuanNv + CD Head — 0.798
  - 浅蓝：XuanNv Raw — 0.500
- 浅灰虚线：Random Baseline (0.50)
- 数据直接标注在柱顶

**配图**：`references/charts/01_cd_comparison_ba.png`

**关键数字**：0.798、0.713、0.500  
**讲解要点**：加 CD Head 超 AEF 是事实；不加时 50% 也是事实， roadmap 有补上计划。

---

## P5 竞品雷达：时序 + 压缩 + 中国场景领先

**页面类型**：雷达图页  
**视觉元素**：
- 中央雷达图，6 维度：
  Compression, Temporal, Change Detection, Multi-modal, Retrieval, China Scenes
- 三条线，无填充，空心圆标记：
  - 灰色虚线：AEF
  - 浅蓝实线：Clay
  - 深蓝实线：XuanNv

**配图**：`references/charts/02_competitor_radar.png`

**关键数字**：3 个领先维度  
**讲解要点**：AEF 强在通用性，弱在时序和检索；我们填补了这两个空白。

---

## P6 V5 评估分布：69 patches 的 BA / AUC 直方图

**页面类型**：数据分布页  
**视觉元素**：
- 左右并置两个直方图：
  - (a) BA 分布：学术蓝，均值线 0.798，AEF 基准线 0.713
  - (b) AUC 分布：浅蓝，均值线 0.955
- 图下方小字：Harbin, 695 km², 69 manually annotated patches

**配图**：`references/charts/03_v5_ba_auc_distribution.png`

**关键数字**：0.798、0.955、69  
**讲解要点**：分布稳定，不是 cherry-pick；提到 3 个异常值和标注歧义问题。

---

## P7 轻量化：23TB → 3GB → 16MB

**页面类型**：阶梯对比页  
**视觉元素**：
- 中央三层阶梯，蓝灰色系从上到下：
  - Raw Satellite Imagery (MajorTOM) — 23 TB — 深灰蓝
  - EarthEmbeddingExplorer — 3 GB — 学术蓝
  - XuanNv Base — 16 MB — 浅蓝
- 层间标注压缩倍数
- 底部注释框：「Total: ~1,500,000× compression. Offline-capable.」

**配图**：`references/charts/06_compression_ladder.png`

**关键数字**：23TB、3GB、16MB  
**讲解要点**：16MB 比表情包还小，边缘设备离线可用。

---

## P8 效率提升：开发周期 3× / 存储 5× / 响应 21×

**页面类型**：横向柱状图页  
**视觉元素**：
- 中央横向柱状图，学术蓝统一色系：
  - Dev Cycle: 6mo → 2mo = 3.0×
  - Team Size: 8 → 3 = 2.7×
  - Storage: 100TB → 20TB = 5.0×
  - Coverage: 30% → 100% = 3.3×
  - Latency: 504hr → 24hr = 21.0×
- 数值直接标注在柱尾

**配图**：`references/charts/04_efficiency_comparison.png`

**关键数字**：3×、5×、24hr  
**讲解要点**：数字来自实际项目；存储降是因为下游不需要原始影像。

---

## P9 诚实面对：Raw BA 差距 21 个百分点

**页面类型**：坦诚页  
**视觉元素**：
- 中央大号数字对比：
  - 左侧灰色：AEF Raw BA = 0.713
  - 右侧浅蓝：XuanNv V5 Raw BA ≈ 0.50
  - 中间向下箭头：差距 ≈ 21pp
- 下方两行原因（小字）：
  1. 数据量：AEF 全球级 vs 哈尔滨 695km²
  2. 模型规模：AEF 更深、参数更多
  3. 差距非结构性，V6/V6.5 已规划

**关键数字**：0.713、0.50、21pp  
**讲解要点**：主动披露建立信任；差距来源是数据量和规模，不是架构缺陷。

---

## P10 升级路线：V5 → V6 → V6.5 → V7

**页面类型**：路线图页  
**视觉元素**：
- 中央横向柱状图，蓝灰色系渐变：
  - V5 (Current): 学术蓝，Raw BA ~0.50
  - V6: 中蓝，目标 >0.60
  - V6.5: 浅蓝，目标 >0.70
  - V7 (Planned): 更浅蓝，目标 ~0.75
- 叠加灰色虚线：AEF Baseline (0.713)
- 每版本上方标注关键特性

**配图**：`references/charts/07_roadmap.png`

**关键数字**：0.60、0.70、0.75  
**讲解要点**：V5 已验证，V6 在训，V6.5 已设计，V7 是长期目标。

---

## P11 总结：够轻 · 够敏 · 够诚

**页面类型**：总结页  
**视觉元素**：
- 页面中央三行大字：
  1. **够轻** — 64 字节 / km²，23TB → 16MB，离线可用
  2. **够敏** — 月度级时序敏感，BA 79.8%
  3. **够诚** — 从零自研，优势不夸大，短板不回避
- 底部：「有问题随时打断。」

**关键数字**：64 字节、79.8%  
**讲解要点**：三词收尾，不喊口号，真诚邀请提问。

---

## 附录：素材清单

| 页码 | 素材文件名 | 说明 |
|------|-----------|------|
| P2 | `05_model_architecture.png` | 模型架构流程图（蓝灰科研风） |
| P4 | `01_cd_comparison_ba.png` | BA 对比柱状图 |
| P5 | `02_competitor_radar.png` | 竞品雷达图（无填充） |
| P6 | `03_v5_ba_auc_distribution.png` | BA/AUC 分布直方图 |
| P7 | `06_compression_ladder.png` | 压缩阶梯图 |
| P8 | `04_efficiency_comparison.png` | 效率提升横向柱状图 |
| P10 | `07_roadmap.png` | V6/V6.5/V7 路线图 |
| P7 | `EarthEmbddingExplorer-演示视频.mp4` | 轻量化检索演示（可选） |

---

*大纲结束。所有图表统一蓝灰色调，白底，可直接插入 PPT。*
