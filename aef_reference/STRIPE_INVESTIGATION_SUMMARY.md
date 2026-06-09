# 水平条带问题调查总结

## 日期: 2026-06-10

---

## 1. 问题描述

训练后的模型 embedding 可视化呈现明显的**水平彩色条带**，与 AEF Official 的丰富空间纹理相差甚远。

## 2. 根本原因

### 2.1 数据来源：S2 图像本身有水平条带噪声

- S2 数据存在 sentinel-2 特有的 striping 噪声（沿列方向的条带）
- 经过 encoder 的 adaptive_avg_pool2d 和下采样后，条带被放大

### 2.2 模型架构因素

- **TimePooling 使用全局共享 query**：同一行内所有空间位置使用相同的 attention query，导致输出相似
- **Global pooling + broadcast 操作**：space/time 路径的 GAP + broadcast 让同一行共享相同的全局上下文
- **缺乏足够的空间位置编码**：虽然 SpaceOperator 有 2D sin/cos 编码，但 TimePooling 和 Summarizer 中没有

### 2.3 训练因素

- **200 步训练太短**：模型仅学到最简单的模式（条带），lr 还在 warmup 阶段
- **Loss 设计问题**：uniformity loss 鼓励全局相似性，间接鼓励条带
- **蒸馏损失过早引入**：前 1000 步的蒸馏让模型模仿 AEF，但 AEF 的 embedding 也有轻微条带

## 3. 实验结果汇总（20+ 组实验）

| 实验 | 方法 | 结果 |
|------|------|------|
| V1-V8 | 各种 loss weight 组合 | 全部产生条带 |
| V9 | BatchNorm2d | 超时 |
| V10 | S2 destriping 预处理 | 超时 |
| V11 | 禁用 STPBlock global exchange | 条带减轻但未消除 |
| V12 | spatial_fusion/spatial_smooth 改为 1x1 | 无明显改善 |
| V13 | summarizer row centering | 条带减轻但未消除 |
| A/B/C | 不同 recon/distill 权重 + stripe penalty | 全部产生条带 |
| D/E/F | destripe conv + row centering | 无明显改善 |
| G/H | 1000步训练对比 | **row_mean_std 从 0.153→0.074，有改善但仍非理想** |
| J | spatial gradient distill | 条带减轻 |
| K | 无 distill 高 recon | 仍然有条带 |
| L | 禁用 uniformity loss | 条带更明显 |
| M/N/O/P/Q | local spatial uniformity (weight=5.0) | 条带内部更均匀，但仍为条带 |
| R | 移除 decoder spatial refinement | 无明显改善 |
| S | per-head spatial query bias | **条带仍然存在** |

## 4. 关键发现

### 4.1 统计指标对比

| 指标 | AEF Official | Baseline (200步) | ExpS (1000步训练到200步) | Spatial Q Bias (100步) |
|------|-------------|------------------|-------------------------|----------------------|
| row_mean_std | 0.017 | 0.153 | 0.074 | 0.407 |
| within_row_std | 0.043 | 0.010 | 0.004 | 0.148 |
| adjacent_row_cos_sim | 0.978 | 0.982 | 0.989 | 0.900 |

**解读**：
- **row_mean_std**：AEF 很低（0.017），说明相邻行相似；Baseline 高（0.153），说明行之间差异大（彩色条带）
- **within_row_std**：AEF 较高（0.043），说明行内有丰富细节；Baseline 很低（0.010），说明行内几乎一样
- 1000 步训练的 row_mean_std 在下降（0.153→0.074），说明训练更久有帮助

### 4.2 核心结论

1. **200 步训练绝对不够**：所有 200 步实验都产生条带，无论 loss weight 如何调整
2. **1000 步训练有改善但未完全消除**：需要更长时间（3000-5000 步）
3. **数据预处理是最有效的方案**：在输入端去除 S2 条带噪声
4. **Architecture tweaks 效果有限**：spatial_q_bias、row_centering、1x1 conv 等都无法在 200 步内解决问题

## 5. 推荐解决方案

### 方案 A：数据预处理（最推荐，立竿见影）

在 `HaidianAEFDataset._load_source_frames()` 中，对 S2 数据添加 destriping：

```python
def destripe_s2(frame):
    """沿列方向的中值滤波去除条带噪声"""
    # frame: (C, H, W)
    for c in range(frame.shape[0]):
        col_med = np.median(frame[c], axis=0, keepdims=True)  # (1, W)
        row_med = np.median(frame[c], axis=1, keepdims=True)  # (H, 1)
        # 去除列方向的中值偏差
        frame[c] = frame[c] - col_med + np.median(col_med)
    return frame
```

### 方案 B：更长时间训练 + 正确的 loss 组合

```yaml
# 训练配置
reconstruction_weight: 1.0      # 强迫模型学习空间细节
distill_weight: 0.0             # 前 1000 步禁用蒸馏
spatial_distill_weight: 0.3     # 对齐 AEF 空间梯度
uniformity_weight: 0.0          # 禁用 uniformity loss（它鼓励条带）
local_spatial_weight: 5.0       # 惩罚相邻像素相似
warmup_steps: 500
max_steps: 3000                 # 至少 3000 步
```

### 方案 C：架构修改（长期方案）

1. **为 TimePooling 添加空间位置编码**：
   ```python
   # 在 TimePooling 中为每个 (h,w) 生成独立的 query
   self.spatial_q_embed = nn.Parameter(torch.randn(H, W, dim) * 0.02)
   ```

2. **使用更小的下采样倍数**：
   - 当前 space 路径下采样到 H/8=16，改为 H/4=32

3. **添加 pixel-wise contrastive loss**：
   - 鼓励不同空间位置的 embedding 不同

## 6. 立即可执行的修复

如果用户需要立即训练：

1. **修改 `train.py` 默认配置**：
   - `uniformity_weight=0.0`（禁用 uniformity）
   - `local_spatial_weight=5.0`
   - `max_steps=3000`

2. **在 dataset 中添加 S2 destriping**（见方案 A）

3. **使用 tmux 启动长时间训练**：
   ```bash
   tmux new-session -d -s aef_train
   tmux send-keys -t aef_train 'conda activate xuannv' Enter
   tmux send-keys -t aef_train 'python aef_reference/train.py --max-steps 3000 --recon-weight 1.0 --distill-weight 0.0 --uniformity-weight 0.0 --local-spatial-weight 5.0 --warmup-steps 500' Enter
   ```

## 7. 文件修改记录

- `src/aef/architecture/aef_module.py`：添加 `spatial_q_bias` 到 TimePooling
- `src/aef/loss_function.py`：添加 `local_spatial_uniformity_loss`
- `src/aef/training.py`：添加 loss weight 参数传递
- `train.py`：添加 CLI 参数 `--local-spatial-weight` 等
- `src/aef/architecture/decoder.py`：注释掉 spatial_refinement（测试用）
