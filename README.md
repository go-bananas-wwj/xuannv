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

## 目录结构

```
xuannv_embdding/
├── README.md
├── pyproject.toml
├── configs/
│   └── qwen_v1_scenes.yaml      # 训练配置
├── src/
│   ├── __init__.py
│   ├── config.py                # 配置系统 (YAML 数据类)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # 数据集 (单景+temporal aug)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py             # 主模型 (AEFModel)
│   │   ├── bottleneck.py        # VMF 瓶颈 (训练 skip L2) ★核心改进
│   │   ├── blocks.py            # STP block (三路径)
│   │   ├── sensor_encoders.py   # 传感器编码器
│   │   ├── decoders.py          # 条件解码器
│   │   └── time_encoding.py     # 时间/窗口编码
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py            # 损失函数 ★核心改进
│   │   └── trainer.py           # DDP 训练器
│   └── utils/
│       ├── __init__.py
│       └── io.py
└── scripts/
    ├── train/                   # 训练脚本
    │   └── train_ddp.py         # DDP 训练入口
    ├── inference/               # 推理脚本
    ├── eval/                    # 评估脚本
    └── visualize/               # 可视化脚本
```

## 快速开始

### 1. 安装

```bash
cd /workspace/xuannv
pip install -e .
```

### 2. 训练 (DDP 多GPU)

```bash
cd /workspace/xuannv

# 3卡 DDP
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
    scripts/train/train_ddp.py --config configs/qwen_v1_scenes.yaml \
    --save-every 50 --warmup-epochs 10
```

### 3. 提取月度 Embedding

```bash
cd /workspace/xuannv
python scripts/inference/extract_monthly_embeddings_all_patches.py \
    --gpu_idx 0 --total_gpus 2
```

## 训练配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| embedding_dim | 128 | embedding 维度 |
| vmf_kappa | 2000 | VMF 浓度 (原版500→2000) |
| max_frames | 32 | 最大帧数 (单景数据更多) |
| skip_l2_norm_training | true | ★ 训练跳过 L2 |
| temporal_window_augmentation | true | ★ 时间窗口增强 |
| raw_uniformity_weight | 0.3 | 主 uniformity 信号 |
| decorrelation_weight | 0.01 | 去相关 |
| variance_weight | 1.0 | 方差正则 |
| orthogonality_weight | 1.0 | 权重正交 |

## 训练监控指标

训练中应关注以下指标:

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `raw_unif` | -4.0 ~ -1.0 | > -0.5 表示坍缩 |
| `pre_unif` | 接近 raw_unif | 差距大说明有问题 |
| `recon` | < 0.3 | > 0.5 重建质量差 |
| `var_reg` | 接近 0 | > 0.5 方差坍缩 |
| `orth` | < 0.3 | > 0.5 权重不正交 |
| `decorr` | < 1.0 | > 2.0 强相关 |

## 与原版代码的关系

- **完全独立实现**: 所有代码从零编写，不复制原版
- **参考接口设计**: 数据加载协议、模型输入输出格式与原版兼容
- **可并行使用**: 训练输出到 `/workspace/outputs/xuannv_embdding_v1/`，不影响原版输出
