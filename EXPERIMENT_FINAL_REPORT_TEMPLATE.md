# V13 Uniformity 坍缩对照实验 — 最终报告

## 实验设计

| 实验 | 名称 | 核心修改 |
|------|------|----------|
| Exp1 | Spatial Uniformity | uniformity 输入: embedding_map[B,D,H,W] |
| Exp2 | High Weight | batch_uniformity_weight: 0.05 → 0.5 |
| Exp3 | VICReg Fix | vicreg_min_std: 1.0 → 0.1 |
| Exp4 | Combined | Exp1 + Exp2 + Exp3 |
| Exp5 | Pre-norm Raw | raw_uniformity_loss 替代 L2 uniformity |

## 完整结果

### Epoch 1-10 趋势

【待填充】

## 结论

### 各策略效果评估

1. **Spatial Uniformity**
   - 效果: 【待填充】
   - 优势: 【待填充】
   - 局限: 【待填充】

2. **增大 Uniformity 权重**
   - 效果: 【待填充】
   - 优势: 【待填充】
   - 局限: 【待填充】

3. **修正 VICReg min_std**
   - 效果: 【待填充】
   - 优势: 【待填充】
   - 局限: 【待填充】

4. **组合策略**
   - 效果: 【待填充】
   - 优势: 【待填充】
   - 局限: 【待填充】

5. **Pre-norm Raw Uniformity**
   - 效果: 【待填充】
   - 优势: 【待填充】
   - 局限: 【待填充】

## 推荐方案

基于实验结果，推荐以下改进策略：

【待填充】

## 下一步实验

【待填充】
