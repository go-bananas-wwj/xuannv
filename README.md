# xuannv_embdding — AlphaEarth Foundations 改进版

> 从零实现的 AlphaEarth Foundations 改进版本，核心解决**嵌入坍缩**与**时间敏感性不足**两大问题。
> 所有代码位于 `/workspace/xuannv/`，不影响原有 `/workspace/AEF/` 的代码。

## 与原版的差异

| 方面 | 原版 AEF (`/workspace/AEF/`) | xuannv_embdding (`/workspace/xuannv/`) |
|------|----------|----------|
| **Uniformity Loss** | `batch_uniformity_loss` (球面空间，L2 norm后) | `raw_uniformity_loss` (欧氏空间，自适应t) |
| **Bottleneck 训练** | L2 norm + VMF noise | 跳过 L2 norm，保留原始幅度 |
| **反坍缩机制** | 多个 loss 竞争，梯度被 L2 屏障衰减 | raw_uniformity + decorrelation + variance + orthogonality 四件套，均在 pre-norm 空间工作 |
| **时间窗口** | 固定宽窗口 (~2.4年) | 训练时随机裁剪 2~24 帧窗口 |
| **数据模式** | 季度合成 (YYYYQN) | 原生支持单景 (YYYYMMDD) + 季度兼容 |

## 核心改进原理

### 嵌入坍缩问题

```
原版流程:
  Conv1x1 → L2 Norm → Uniformity Loss
  问题: L2 Norm 的 Jacobian (I - uu^T)/||x|| 在坍缩态秩降为 D-1
        梯度被杀死 → uniform 无法推动 embedding 分散

改进流程:
  Conv1x1 → raw_uniformity_loss (原始幅度空间)
  → 梯度永远非零 (幅度差异被保留)
  → 推理时再做 L2 Norm

raw_uniformity_loss 原理:
  1. 标准化到零均值、全局单位方差
  2. 自适应 t = 2/D
  3. 欧氏空间 RBF uniformity: log(mean exp(-t * ||zi - zj||²))
```

### 时间敏感性不足

```
原版:
  训练 valid_period = [2023.3, 2025.8] (固定宽窗口 ~2.4年)
  → WindowEncoder 从未见过窄窗口
  → 推理时窄窗口查询完全 OOD → cosine distance ≈ 1e-7

改进:
  50% 概率随机裁剪 valid_period 为 4~24 帧窗口
  → WindowEncoder 在训练中见过各种宽度
  → 推理时不再 OOD → 可以生成不同的 before/after embedding
```

## 快速开始

### 1. 安装

```bash
cd /workspace/xuannv
pip install -e .
```

### 2. 训练 (NPU DDP)

```bash
cd /workspace/xuannv

# 4卡 NPU DDP 示例 (V2 海淀区基线)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
    scripts/train/train_xuannv_v2.py \
    --config configs/xuannv_v2_haidian_baseline.yaml
```

> **注意**: 本项目已全面适配华为 Ascend 910B NPU (`torch_npu` + `hccl`)，训练脚本默认使用 NPU。

### 3. 详细文档

完整信息（代码结构、配置系统、损失函数体系、训练监控、开发规范等）见 **[AGENTS.md](AGENTS.md)**。

## 训练监控指标

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `l2unif` | -0.9 ~ -0.7 | > -0.5 表示坍缩 |
| `std_mean` | > 0.60 | < 0.50 方差不足 |
| `active` | 64/64 | < 50 维度坍缩 |
| `recon` | < 0.1 | > 0.3 重建失败 |
| `orth` | < 0.3 | > 0.5 权重不正交 |

## 与原版代码的关系

- **完全独立实现**: 所有代码从零编写，不复制原版
- **参考接口设计**: 数据加载协议、模型输入输出格式与原版兼容
- **可并行使用**: 训练输出到 `/workspace/outputs/`，不影响原版输出
