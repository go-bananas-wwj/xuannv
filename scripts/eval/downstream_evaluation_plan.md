# 下游任务评估方案

## 可用数据

| 数据集 | 类型 | 类别数 | 时间特性 |
|--------|------|--------|----------|
| WorldCover | 土地利用分类 | 10类 | 静态 (2020) |
| JRC Water | 水体提取 | 2类 | 静态 |
| Dynamic World | 动态土地利用 | 9类 | 季度动态 |

数据路径：
- `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/worldcover/{patch_id}/static.tif`
- `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/jrc_water/{patch_id}/static.tif`
- `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/dynamic_world/{patch_id}/2023Q1.tif`

## 推荐方案：冻结 Backbone + 训练下游 Head

### 步骤
1. 加载 baseline checkpoint (`epoch_best_epoch20.pt`)
2. 冻结 backbone 所有参数
3. 添加下游分类 Head：
   - 方案 A: `PixelMLPHead` (已有实现)
   - 方案 B: 简单 Conv1x1 (轻量)
4. 分别对 3 个下游任务训练 head
5. 评估指标：OA、mIoU、per-class F1

### 时间估算
- 训练3个下游 head: ~30-60 分钟
- 评估: ~10 分钟

### 备选方案：KNN 评估（更快）
- 提取 embedding → KNN 分类
- 无需训练，直接评估 embedding 线性可分性
- 时间: ~5 分钟

## 建议
先执行 **KNN 评估** 快速验证 backbone 表征质量，
如果结果有潜力，再执行 **Head 训练** 获得更准确的评估。
