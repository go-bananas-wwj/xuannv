# AEF 对齐实验 — 进度日志

## 2026-05-16 Session

### 已完成
- [x] AGENTS.md 压缩 (706 → 510 行)
- [x] v2 实验评估 (8 实验全部 AUC ≈ 0.5)
- [x] 时间窗口修复 (2024 → 2025)
- [x] Bare AUC 验证 (0.5363，确认 temporal blindness)
- [x] AEF 论文研究 (Recon+BatchUnif+Consistency，无显式 temporal loss)
- [x] 变化检测 mask 可视化 ✅
- [x] /grill-me 对齐检查 (发现 10+ 不对齐项)
- [x] 创建 task_plan.md
- [x] 创建 findings.md

### 进行中
- [x] 创建 progress.md
- [x] 修改 decoder 恢复条件注入
- [x] 修改 bottleneck skip_l2=false
- [x] 修改 target_sources 降低 static 权重
- [x] 创建 8 个实验配置 (64D, 100 patches)
- [x] 创建 unified 训练脚本
- [x] 启动 8 卡并行训练
- [ ] 监控训练进度
- [ ] 训练完成后评估 Bare AUC

### 关键发现
1. **Decoder 条件注入被禁用** — V13 的 ConditionInjector 直接返回 embedding，这是 temporal blindness 的根因
2. **Bottleneck skip L2** — 训练时不在球面上
3. **Static 57%** — 远超 AEF 的 22%
4. **多余损失** — VICReg/Decorr/Orth/CLS 等 AEF 不用的损失

### 下一步
立即开始 P0 修复，然后创建配置并启动训练。
