# 玄女底座汇报 PPT 大纲（v5 模型+平台全栈版）

**风格锁定**：白底，无动画，每页 ≤3 个关键数字，工程师+产品经理混合语气  
**总页数**：12 页  
**预计时长**：35 分钟  
**核心叙事**：从模型到产品 — 先讲底座能力，再讲平台形态，最后讲商业化价值

---

## P1 封面：玄女底座 — 模型+平台，双轮驱动

**页面类型**：封面页  
**视觉元素**：
- 标题大字：「玄女底座」
- 副标题：「自研地球嵌入基座 + 可视化展示平台」
- 底部一行小字：「模型训练 · 下游推理 · 交互标注 · AI 智能体 — 全栈产品」
- 纯白底，可加淡色卫星底图

**关键数字**：无  
**讲解钩子**："今天聊的不只是模型，是一整套已经跑起来的东西。"

---

## P2 模型原理：输入 → STP 编码器 → VMF 瓶颈 → 64 维输出

**页面类型**：流程图页  
**视觉元素**：
- 顶部 4 个蓝灰色调方块（同 v4 科研风）
- 底部 2 个灰蓝色方块：Reconstruction + Anti-Collapse Loss
- 最底部 Temporal Contrastive 小字框

**配图**：`references/charts/05_model_architecture.png`

**关键数字**：64 维、64 字节  
**讲解要点**：训练 skip L2 防坍缩，推理 L2+VMF，提到"第一次崩到-0.1"。

---

## P3 核心亮点：反坍缩 · 时序敏感 · 纯自研

**页面类型**：三栏要点页  
**视觉元素**：
- 三行等宽排列，数字圆圈 + 标题 + 一行解释

**关键数字**：无（概念页）  
**讲解要点**：三个差异化，每个都是别人没有或做得不够的。

---

## P4 变化检测：BA 79.8%（含 CD Head）/ 50%（Raw）

**页面类型**：柱状图对比页  
**视觉元素**：
- 三个柱子，蓝灰色系：AEF 0.713 / XuanNv+CDH 0.798 / XuanNv Raw 0.500
- 浅灰虚线 Random Baseline

**配图**：`references/charts/01_cd_comparison_ba.png`

**关键数字**：0.798、0.713、0.500  
**讲解要点**：加 CD Head 超 AEF 是事实；不加时 50% 也是事实。

---

## P5 展示平台总览（⭐ 新增重点页）

**页面类型**：产品 showcase 页  
**视觉元素**：
- **左侧 60%**：平台首页截图占位（用户提供 hero section 截图）
  - 截图尺寸建议：16:9 比例，分辨率 ≥1920×1080
  - 截图内容：Three.js 地球 / 卫星背景 / "让遥感模型分析像用电一样便捷"
- **右侧 40%**：三大模块列表，每行一个图标 + 一句话：
  1. 🌍 三维地球可视化 — Three.js 全球训练样本分布
  2. 📊 下游监测能力 — 5 大 Task Head 动态切换
  3. 🤖 AI 智能体报告 — DeepSeek-V4 自然语言生成报告
- **底部**：技术栈标签行 — React 19 · TypeScript · Three.js · MapLibre · FastAPI · Docker

**截图素材**：用户提供 `platform_hero.png`（建议命名）  
**备用素材**：`references/charts/hero_screenshot_placeholder.jpg`

**关键数字**：3 大模块、React 19 + FastAPI  
**讲解要点**：不是 PPT，是真能点、能切、能出报告的网页；技术栈拿出去跟任何一家比都不虚。

---

## P6 五大下游任务（⭐ 新增重点页）

**页面类型**：能力矩阵页  
**视觉元素**：
- 中央：5 行表格/矩阵图（`references/charts/08_downstream_tasks.png`）
  - 每行：任务名 | 技术方案 | 计算资源 | 状态
  - 5 个任务全部标 ✅ 已上线
- 表格下方两行关键数字：
  - "4 / 5 任务纯 CPU 推理，单 patch ~10ms"
  - "CD Head 动态选择空闲 GPU（8×4090 自动调度）"

**配图**：`references/charts/08_downstream_tasks.png`

**关键数字**：5 个任务、~10ms、8×GPU  
**讲解要点**：全部已部署在平台上，不是规划，是已经能点的那种。

---

## P7 交互式标注训练（⭐ 新增重点页）

**页面类型**：流程图页  
**视觉元素**：
- 中央横向流程图（`references/charts/09_annotation_pipeline.png`）：
  - Upload → SAM3 Auto-Segment → Interactive Correction → Train Linear Probe → Inference
- 每个步骤下方配一行小字解释
- 底部高亮："Zero-code custom training · Full pipeline ~5 min"
- **右侧或底部**：用户可提供 AnnotatePage 实际截图作为补充

**配图**：`references/charts/09_annotation_pipeline.png`  
**截图素材**（可选）：用户提供 `annotate_page.png`

**关键数字**：5 分钟、零代码  
**讲解要点**：业务人员自己就能训专用分类模型；传统方式至少两周。

---

## P8 AI 智能体报告（⭐ 新增重点页）

**页面类型**：交互示意页  
**视觉元素**：
- 左右分栏示意（`references/charts/10_agent_report.png`）：
  - 左：自然语言输入框（mock 文字）
  - 中：DeepSeek V4 箭头
  - 右：Markdown 报告输出（mock 表格+分析）
- 底部三圆流程：flash parse → task execute → pro polish
- **右侧或底部**：用户可提供 AgentSection 实际截图作为补充

**配图**：`references/charts/10_agent_report.png`  
**截图素材**（可选）：用户提供 `agent_section.png`

**关键数字**：DeepSeek-V4、3 类任务、10 秒出报告  
**讲解要点**：已经接通，不是规划；输一句话就能出带表格和分析的报告。

---

## P9 轻量化：23TB → 3GB → 16MB

**页面类型**：阶梯对比页  
**视觉元素**：
- 三层阶梯，蓝灰色系：Raw 23TB / EEE 3GB / XuanNv 16MB
- 底部注释框

**配图**：`references/charts/06_compression_ladder.png`

**关键数字**：23TB、3GB、16MB  
**讲解要点**：16MB 比表情包小；平台上所有推理底层都是这 64 字节。

---

## P10 效率提升

**页面类型**：横向柱状图页  
**视觉元素**：
- 5 个维度横向柱状图，学术蓝统一色系

**配图**：`references/charts/04_efficiency_comparison.png`

**关键数字**：3×（周期）、5×（存储）、24hr（响应）  
**讲解要点**：数字来自实际项目经验；存储降是因为不需要原始影像。

---

## P11 升级路线：V5 → V6 → V6.5 → V7

**页面类型**：路线图页  
**视觉元素**：
- 4 个版本柱状图，蓝灰色系渐变
- 灰色虚线：AEF Baseline (0.713)

**配图**：`references/charts/07_roadmap.png`

**关键数字**：0.60、0.70、0.75  
**讲解要点**：V5 已验证，V6 在训，V6.5 已设计，V7 长期目标。

---

## P12 总结：够轻 · 够敏 · 够全 · 够诚

**页面类型**：总结页  
**视觉元素**：
- 四行大字（从三行扩展为四行）：
  1. **够轻** — 64 字节 / km²，23TB → 16MB，离线可用
  2. **够敏** — 月度级时序敏感，BA 79.8%
  3. **够全** — 模型+平台+5 大任务+交互训练+AI 智能体
  4. **够诚** — 从零自研，优势不夸大，短板不回避
- 底部：「有问题随时打断。」

**关键数字**：64 字节、79.8%、5 大任务  
**讲解要点**：四词收尾，不喊口号，新增"够全"强调产品完整性。

---

## 附录 A：完整素材清单

| 页码 | 素材文件名 | 说明 | 来源 |
|------|-----------|------|------|
| P2 | `05_model_architecture.png` | 模型架构流程图 | 脚本生成 |
| P4 | `01_cd_comparison_ba.png` | BA 对比柱状图 | 脚本生成 |
| P5 | `hero_screenshot_placeholder.jpg` | 平台首屏卫星背景 | 从 xuannv_show 复制 |
| P5 | `platform_hero.png` | 平台实际运行截图 | **用户提供** |
| P6 | `08_downstream_tasks.png` | 五大下游任务矩阵 | 脚本生成 |
| P6 | `monitoring_section.png` | 任务切换界面截图 | **用户提供** |
| P7 | `09_annotation_pipeline.png` | 交互标注流程图 | 脚本生成 |
| P7 | `annotate_page.png` | 标注界面截图 | **用户提供（可选）** |
| P8 | `10_agent_report.png` | AI 智能体交互示意 | 脚本生成 |
| P8 | `agent_section.png` | 智能体界面截图 | **用户提供（可选）** |
| P9 | `06_compression_ladder.png` | 压缩阶梯图 | 脚本生成 |
| P10 | `04_efficiency_comparison.png` | 效率提升图 | 脚本生成 |
| P11 | `07_roadmap.png` | 升级路线图 | 脚本生成 |

---

## 附录 B：用户需提供的截图清单

| 序号 | 截图内容 | 建议文件名 | 用途 |
|------|---------|-----------|------|
| 1 | 平台首页 / Three.js 地球 | `platform_hero.png` | P5 左侧主图 |
| 2 | MonitoringSection 任务切换 + 结果图 | `monitoring_section.png` | P6 补充展示 |
| 3 | AnnotatePage 标注界面 | `annotate_page.png` | P7 补充展示（可选） |
| 4 | AgentSection 输入+报告 | `agent_section.png` | P8 补充展示（可选） |

> 截图建议：浏览器全屏截取，分辨率 ≥1920×1080，PNG 格式。如果平台当前无法启动，可延后补充，先用占位图+文字描述。

---

*大纲结束。全部图表统一蓝灰色调科研风，白底，可直接插入 PPT。*
