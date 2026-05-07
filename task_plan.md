# 玄女底座汇报改进计划 — 方案B（完整路线）

## 目标
为任总准备一场30分钟深度汇报，展示玄女底座自研模型的技术亮点、与AEF及全球竞品的差异化优势，以及嵌入向量的实际应用价值。采用雷军式对比叙事结构。

## 叙事核心
> "AEF没做到的，我们做到了" —— 在反坍缩、时序敏感性、变化检测精度三个维度实现超越。

---

## 阶段与进度

### Phase 1: AEF论文精读与评估方法论梳理
- [ ] 精读AEF论文正文+补充材料S2/S4
- [ ] 提取评估体系：15 evaluations, 1/10/max-shot, kNN/Linear, BA/R²
- [ ] 提取变化检测评估方法（direct vs unsupervised）
- [ ] 明确AEF技术边界与未解决问题
- **状态**: in_progress
- **截止**: Day 1

### Phase 2: 数据获取与评估复刻
- [ ] 下载AEF原版哈尔滨embedding (ModelScope)
- [ ] 复刻变化检测评估（BA + AUC）
- [ ] 计算时序敏感性指标（cos_sim分布）
- [ ] 复刻EuroSAT kNN/Linear评估（验证通用性）
- [ ] 整合EarthEmbeddingExplorer视频素材
- **状态**: pending
- **截止**: Day 2-4

### Phase 3: 竞品深度分析与差异化定位
- [ ] 建立AEF/Clay/Tessera/OlmoEarth/DINOv3对比矩阵
- [ ] 提炼"玄女底座"技术独特性（3个超越点）
- [ ] 产出竞品能力雷达图数据
- **状态**: pending
- **截止**: Day 5

### Phase 4: 汇报素材生产
- [ ] AEF vs 我们的变化检测对比图
- [ ] 时序敏感性直方图（变化vs未变化cos_sim分布）
- [ ] Uniformity对比图（V1 vs V2/V6）
- [ ] 竞品能力雷达图
- [ ] EarthEmbeddingExplorer截图/关键帧
- [ ] 哈尔滨变化检测典型案例（Before/After+热力图）
- **状态**: pending
- **截止**: Day 6-8

### Phase 5: 汇报脚本与PPT大纲
- [ ] 雷军式五幕剧脚本（逐页备注）
- [ ] PPT结构大纲（含每页设计说明）
- [ ] 排练建议
- **状态**: pending
- **截止**: Day 9-10

---

## 关键资源
- ModelScope Token: `ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977`
- 视频素材: `/workspace/EarthEmbddingExplorer-演示视频.mp4`
- AEF论文: arXiv:2507.22291
- EarthEmbeddingExplorer: https://github.com/OpenGeoScope/EarthEmbeddingExplorer
- 商业计划书: `/workspace/xuannv/玄女底座商业计划书_和稿版.docx`

## 命名约定
- 模型名称: **玄女底座**（非AEF改进版，独立自研）
- 评估口径: 优先计算BA，与AEF统一口径；若效果不佳则明确说明指标差异
