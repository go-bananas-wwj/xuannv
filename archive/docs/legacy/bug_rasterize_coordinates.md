# Bug 总结：光栅化标注坐标映射错误

## 发现时间
2026-05-09，在 V8 Clean CD Head 训练验证阶段发现

## 影响范围
**几乎所有验证/训练脚本**（10+ 个文件）

受影响的文件：
- `scripts/eval/validate_v2.py`
- `scripts/eval/validate_v4_official.py`
- `scripts/eval/validate_v4_level1_bare.py`
- `scripts/eval/validate_v5_level1_bare.py`
- `scripts/eval/validate_v6_level1_bare.py`
- `scripts/eval/validate_v6_5_level1_bare.py`
- `scripts/eval/validate_v7_level1_bare.py`
- `scripts/eval/validate_v8_clean.py`
- `scripts/eval/train_cd_head_v8.py`
- `scripts/eval/analyze_v5_embedding_space.py`
- `scripts/eval/evaluate_alphaearth_vs_v4.py`

## Bug 描述

### 错误代码
```python
for px in range(H):          # px 在 mask 中是 row（第1维，对应 y 方向）
    for py in range(W):      # py 在 mask 中是 col（第2维，对应 x 方向）
        wx = bounds[0] + (px + 0.5) * resolution      # ❌ 错误：px(row) 被当作 x
        wy = bounds[3] - (py + 0.5) * resolution_y    # ❌ 错误：py(col) 被当作 y
        if geom.contains(Point(wx, wy)):
            mask[px, py] = 1.0                          # ❌ 写入位置也错了
```

### 正确代码
```python
for row in range(H):         # row -> y 方向（南北）
    for col in range(W):     # col -> x 方向（东西）
        wx = bounds[0] + (col + 0.5) * resolution       # ✅ col -> x
        wy = bounds[3] - (row + 0.5) * resolution_y     # ✅ row -> y
        if geom.contains(Point(wx, wy)):
            mask[row, col] = 1.0                          # ✅ row, col
```

### 错误本质
| 变量 | 在 mask 中的含义 | 在错误代码中被当作 |
|------|----------------|------------------|
| px (range H) | row = 第1维 = y方向 | ❌ x 坐标 |
| py (range W) | col = 第2维 = x方向 | ❌ y 坐标 |

当 H = W = 64（正方形 patch）时，错误表现为：**mask 沿主对角线翻转**。

### 为什么容易犯这个错
1. 变量名 `px`, `py` 容易让人联想到 "pixel x", "pixel y"
2. 但 `for px in range(H)` 中 `px` 实际上遍历的是**行号**（row index）
3. 在 numpy array 中 `mask[px, py]` = `mask[row, col]`，不是 `mask[x, y]`

### 为什么 patch_000350 碰巧正确
该 patch 的变化区域在左上角（row=0, col=0），翻转后仍然在左上角，所以碰巧一致。

## 影响后果

1. **CD Head 训练**：模型学到的是错误位置的"变化"，导致 val AUC 无法提升（0.56）
2. **Bare AUC 验证**：变化 score 与错误位置的 mask 计算，AUC 失真
3. **历史所有 AUC 结果**：可能被污染，包括 V4 的 AUC=0.896

## 修复措施

### 已执行
1. 2026-05-09 批量修复所有 10+ 个脚本中的坐标映射
2. 生成可视化对比图验证修复正确性
3. 用真实 S2 图像叠加 mask 确认

### 预防措施
1. **命名规范**：遍历 mask 时必须使用 `row/col`，禁止使用 `px/py/x/y`
2. **可视化验证**：任何新的光栅化代码必须先用真实图像叠加验证
3. **单元测试**：光栅化函数应添加坐标正确性断言测试
4. **代码审查**：涉及坐标转换的代码必须双人 review

## 修复后的验证结果

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| V8 Bare AUC | 0.5056 | 0.5086 | +0.003 |
| V8 CD Head AUC | 0.5360 | 0.5649 | +0.029 |

虽然标签修复带来了一定提升，但 V8 backbone 本身仍然是时间盲的，AUC 仍远低于可用水平。
