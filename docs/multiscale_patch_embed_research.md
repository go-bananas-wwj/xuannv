# 多尺度Patch Embed调研总结

## 1. OlmoEarth的做法

> "Inspired by Galileo, we use a flexible patch-embedding layer... we keep the actual projection weights the same size and **resize the input image to mimic changing the patch size**."

- **共享权重**：所有源用同一套patch embed权重
- **变尺寸输入**：高分辨率源resize到更大尺寸 → 更多token
- **统一潜空间**：所有token维度相同（如768D）

## 2. Galileo的做法

> "Galileo: learning global & local features of many remote sensing modalities"

- 轻量Transformer架构
- 没有特别强调patch embed的多分辨率处理
- 主要关注global masked modeling + local contrastive

## 3. DOFA的做法（最相关）

> "a hypernetwork that **dynamically generates network weights** based on the central wavelengths of each spectral band"

- **动态权重生成器**：根据输入波长动态生成patch embed权重
- **不是固定权重**，也不是独立stem
- 权重是根据输入动态计算的
- 批评了独立patch embed的做法："not flexible enough when the number of bands changes"

## 4. 关于"映射头权重固定"的结论

| 做法 | 模型 | 效果 |
|------|------|------|
| **训练所有参数** | MAE, ViT, SatMAE, Prithvi | 标准做法，效果最好 |
| **固定backbone** | DINOv2微调 | 只在下游任务freeze |
| **动态生成权重** | DOFA | 根据输入动态计算 |
| **固定patch embed** | — | 无文献支持，自监督中不推荐 |

**结论：不建议固定patch embed权重。**
- 自监督模型需要端到端学习
- 固定权重会限制模型适应不同源统计特性的能力
- 遥感不同源（SAR纹理 vs 光学RGB）统计特性差异极大，固定stem会损害性能

## 5. 推荐方案

**回到128×128 + 独立stem + 端到端训练 + 统一潜空间**

| 源 | 分辨率 | Patch Size | 原始Grid | 操作 | 统一Grid | Token数 |
|----|--------|-----------|---------|------|---------|--------|
| Planet | 3m | 4×4 | 32×32=1024 | AvgPool 2×2 | 16×16 | **256** |
| SAR | 3m | 4×4 | 32×32=1024 | AvgPool 2×2 | 16×16 | **256** |
| S2 | 10m | 8×8 | 16×16=256 | 无 | 16×16 | **256** |
| Landsat | 30m | 16×16 | 8×8=64 | Upsample 2×2 | 16×16 | **256** |

- 所有源统一输出 **256 tokens**
- 所有stem输出维度统一为 **embed_dim=512**
- **权重端到端训练**（不固定）
- 与AEF全局embedding对齐（[B, 64]）
