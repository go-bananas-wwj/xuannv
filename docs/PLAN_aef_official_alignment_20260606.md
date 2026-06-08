# AEF 官方对齐重构计划（审查修正版）

> 目标：将当前 `src/aef/` 下的 AEF 复现代码对齐到官方论文 (arXiv:2507.22291, Google DeepMind) 的架构设计，并在官方基础上增加蒸馏分阶段训练。
> 
> 生成时间: 2026-06-06
> 
> **审查状态**: 已通过 3 个独立审查子智能体验证（架构对齐审查 + Bug 猎手 + NPU/稳定性审查）

---

## 一、审查发现的关键问题（已纳入本计划）

### 🔴 P0 — 致命 Bug（1 个）

| # | 问题 | 位置 | 影响 | 修复方案 |
|---|------|------|------|---------|
| 1 | **Teacher embedding 被 `detach()`，encoder 丧失 reconstruction 梯度** | `aef_module.py:284` | Reconstruction + 所有反坍缩损失无法回传到 encoder/summarizer，模型几乎无法学习 | **移除 `detach()`**，consistency loss 中对 teacher 单独 detach |

### 🟡 P1 — 严重问题（8 个，核心已纳入）

| # | 问题 | 位置 | 修复方案 |
|---|------|------|---------|
| 2 | `LearnedSpatialResampling` upsample 仅实现 2×，与声明的 4×/8×/16× 不符 | `laplacian_pyramid_exchange.py` | 改为 `nn.Upsample(scale_factor=scale_factor) + Conv2d` |
| 3 | `valid_periods` 在 forward 内从 Python list 构造 Tensor（Host-Device 同步） | `aef_module.py:281` | 在 collate_fn 中预拼成 tensor |
| 4 | Teacher 分支未用 `torch.no_grad()`，浪费显存 | `aef_module.py:270` | 包裹 `with torch.no_grad()` |
| 5 | `torch.set_num_threads(4)` 未设置 | `train.py` | 在 main() 开头添加 |
| 6 | Loss 函数 N=1 时崩溃/NaN | `loss_function.py` | 添加 N<2 保护 |
| 7 | `find_unused_parameters=True` 带来 10-30% DDP 开销 | `train.py` | 评估后改为 False |
| 8 | `_perturb_inputs` batch-level 随机，缺乏 per-sample 多样性 | `aef_module.py:203` | 改为 per-sample 随机 |
| 9 | eval DDP sync 默认 float64，NPU 兼容性差 | `training.py:352` | 显式指定 `dtype=torch.float32` |

### 🟢 P2 — 警告（多个，部分纳入）

- 棋盘格伪影 → 已纳入（Upsample+Conv2d）
- VMF 采样 for-loop → 保持现状（num_samples=1 不受影响）
- resume scheduler 逐个 step 循环 → 暂不修复（非核心）
- 各 rank 不同 seed → 暂不修复（影响轻微）

---

## 二、改动总览

| 改动项 | 涉及文件 | 工作量 | 风险 |
|--------|---------|--------|------|
| **P0: 移除 teacher `detach()`，修复 encoder 梯度流** | `aef_module.py`, `loss_function.py` | 低 | 训练动态剧变 |
| **提升输出分辨率 64×64 → 128×128** | `encoder.py`, `STPBlock.py`, `aef_module.py` | 中 | OOM 风险 |
| **修复 LearnedSpatialResampling（任意倍数 + 棋盘格）** | `laplacian_pyramid_exchange.py` | 低 | 无 |
| **修复 VMF kappa 10 → 8000（固定）** | `decoder.py` | 低 | 无 |
| **Teacher 分支加 `torch.no_grad()` 省显存** | `aef_module.py` | 低 | 无 |
| **精简 Loss（12项 → 5项）** | `loss_function.py`, `training.py` | 中 | 训练动态变化 |
| **增加分阶段蒸馏训练（1000 step warmup）** | `training.py`, `train.py` | 中 | 需调参 |
| **Loss edge case 保护 + DDP float32 + set_num_threads** | `loss_function.py`, `training.py`, `train.py` | 低 | 无 |
| **适配新分辨率的 target 下采样** | `training.py` | 低 | 无 |

---

## 三、详细改动清单

### Step 1: P0 致命 Bug 修复（最高优先级）

#### 3.1 `src/aef/architecture/aef_module.py` — 修复 encoder 梯度流

**当前问题**：Line 284 `mu_t = self.summarizer(feats_teacher.detach(), ts, vp)` 阻止了 reconstruction loss 回传到 encoder。

**改动**：
```python
# 改动前 (line 270, 284)
feats_teacher = self.encoder(x, ts)
# ...
mu_t = self.summarizer(feats_teacher.detach(), ts, vp)  # ❌ 阻断梯度
mu_s = self.summarizer(feats_student, ts_student, vp)

# 改动后
with torch.no_grad():  # Teacher 分支不保留激活，省显存
    feats_teacher = self.encoder(x, ts)
mu_t = self.summarizer(feats_teacher, ts, vp)  # ✅ 正常梯度流
mu_s = self.summarizer(feats_student, ts_student, vp)
```

**同时修改 consistency loss**（`loss_function.py`）：
```python
# 改动前
def consistency_loss(self, teacher_embeddings, student_embeddings):
    mu = torch.nn.functional.normalize(teacher_embeddings, p=2, dim=-1)
    mu_s = torch.nn.functional.normalize(student_embeddings, p=2, dim=-1)
    dots = (mu * mu_s).sum(dim=-1)
    return ((1.0 - dots) * 0.5).mean()

# 改动后
def consistency_loss(self, teacher_embeddings, student_embeddings):
    mu = torch.nn.functional.normalize(teacher_embeddings.detach(), p=2, dim=-1)  # teacher stop-gradient
    mu_s = torch.nn.functional.normalize(student_embeddings, p=2, dim=-1)
    dots = (mu * mu_s).sum(dim=-1)
    return ((1.0 - dots) * 0.5).mean()
```

### Step 2: 架构修复（分辨率 + kappa + 棋盘格 + 显存优化）

#### 3.2 `src/aef/architecture/encoder.py` — 提升 precision 分辨率

**改动**：
```python
# precision pathway: H/2 → H (保持 128×128)
precision_features = F.adaptive_avg_pool2d(
    rearrange(x_proj, 'b t h w c -> (b t) c h w'),
    (H, W)   # ← 128×128
)

# final resample scale 调整
self.final_space_resample = LearnedSpatialResampling(self.space_dim, self.precision_dim, 16.0)   # 8×8 → 128×128
self.final_time_resample = LearnedSpatialResampling(self.time_dim, self.precision_dim, 8.0)     # 16×16 → 128×128
```

#### 3.3 `src/aef/architecture/STPBlock.py` — 跨尺度交换 scale 调整

| 变量名 | 改动前 | 改动后 |
|--------|--------|--------|
| `space_to_precision` | 8.0 | **16.0** |
| `time_to_precision` | 4.0 | **8.0** |
| `precision_to_space` | 0.125 | **0.0625** |
| `precision_to_time` | 0.25 | **0.125** |

#### 3.4 `src/aef/architecture/laplacian_pyramid_exchange.py` — 棋盘格修复 + 任意倍数支持

**改动前**：
```python
if scale_factor > 1:
    self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
```

**改动后**：
```python
if scale_factor > 1:
    assert scale_factor == int(scale_factor), f"scale_factor must be integer, got {scale_factor}"
    self.upsample = nn.Upsample(scale_factor=int(scale_factor), mode='bilinear', align_corners=False)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def forward(self, x):
    if hasattr(self, 'upsample'):
        x = self.upsample(x)
    return self.conv(x)
```

> **审查发现**：原代码 `scale_factor=8.0` 时仅上采样 2×，其余靠 `F.interpolate` fallback，导致"learned"部分只有 2× 是可学习的。

#### 3.5 `src/aef/architecture/decoder.py` — VMF kappa 修复

**改动**：
```python
# 改动前
self.log_kappa = nn.Parameter(torch.log(torch.tensor(10.0)))

# 改动后（固定为 8000，不可学习）
self.register_buffer('kappa', torch.tensor(8000.0))

# forward 中
# kappa = torch.exp(self.log_kappa)  →  kappa = self.kappa
```

#### 3.6 `src/aef/architecture/aef_module.py` — 移除 decoder 后上采样

Decoder 直接输出 128×128，**删除 interpolate 代码**（line 306-310）。

### Step 3: Loss 精简 + Edge Case 保护

#### 3.7 `src/aef/loss_function.py` — 精简 Loss + 动态权重 + N<2 保护

**保留的 5 项 loss**：

| Loss | 阶段 1（0~1000 steps） | 阶段 2（1001+ steps） |
|------|----------------------|----------------------|
| Reconstruction | **0.1** | **1.0** |
| Batch Uniformity | 0.05 | 0.05 |
| Consistency | 0.02 | 0.02 |
| Text Contrastive | 0.0 | 0.001 |
| **AEF Distill** | **5.0** | **0.2** |

**关闭的 loss**（weight=0）：raw_uniformity, variance, covariance, erank, coding_rate, magnitude, decorrelation

**新增 `set_stage()` 接口**：
```python
def set_stage(self, stage: str):
    if stage == "distill_align":
        self.reconstruction_weight = 0.1
        self.distill_weight = 5.0
        self.uniformity_weight = 0.05
        self.consistency_weight = 0.02
        self.text_weight = 0.0
    elif stage == "normal":
        self.reconstruction_weight = 1.0
        self.distill_weight = 0.2
        self.uniformity_weight = 0.05
        self.consistency_weight = 0.02
        self.text_weight = 0.001
```

**Edge case 保护**（所有 loss 函数添加）：
```python
# raw_uniformity_loss, vicreg_variance_loss, vicreg_covariance_loss, 等
if N < 2:
    return x.new_tensor(0.0)
```

**consistency_loss 修改**（teacher detach）：
```python
mu = torch.nn.functional.normalize(teacher_embeddings.detach(), p=2, dim=-1)
```

### Step 4: 训练系统修复

#### 3.8 `src/aef/training.py` — 分阶段逻辑 + DDP 修复

**分阶段训练循环**：
```python
for step in range(resume_step + 1, max_steps + 1):
    if step <= self.distill_warmup_steps:
        self.loss_fn.set_stage("distill_align")
        if step == 1 or step == self.distill_warmup_steps:
            self._log(f"[Stage] distill_align at step {step}")
    else:
        self.loss_fn.set_stage("normal")
        if step == self.distill_warmup_steps + 1:
            self._log(f"[Stage] normal at step {step}")
    # ... 原有训练逻辑 ...
```

**DDP 修复**：
```python
# eval 时 all_reduce 显式 float32
torch.tensor(total_recon, dtype=torch.float32, device=device)

# loss all_reduce 前 detach
for k in losses:
    if isinstance(losses[k], torch.Tensor):
        tensor = losses[k].detach()
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
```

#### 3.9 `src/aef/train.py` — CLI 参数 + 基础修复

**新增参数**：
```python
parser.add_argument("--distill-warmup-steps", type=int, default=1000)
parser.add_argument("--distill-weight-align", type=float, default=5.0)
parser.add_argument("--distill-weight-normal", type=float, default=0.2)
parser.add_argument("--recon-weight-align", type=float, default=0.1)
parser.add_argument("--recon-weight-normal", type=float, default=1.0)
parser.add_argument("--save-every", type=int, default=500)  # 每 500 step 保存
```

**基础修复**：
```python
# main() 开头
torch.set_num_threads(4)

# 所有 rank 相同 seed（或移除 +rank）
set_seed(args.seed)

# DDP: 评估后改为 False（确认所有参数参与梯度）
# find_unused_parameters=False
```

### Step 5: 数据预处理优化

#### 3.10 `src/aef/data.py` / `data/haidian_dataset.py` — valid_periods 预转 tensor

在 `collate_fn` 中将 `valid_periods` 预拼成 `(B, 2)` 的 tensor，避免 forward 中的 Host-Device 同步：
```python
valid_periods = torch.tensor([[s, e] for s, e in batch_valid_periods], dtype=torch.float32)
```

---

## 四、执行顺序

```
Step 1: decoder.py (kappa 8000)                    — 零风险
Step 2: laplacian_pyramid_exchange.py (棋盘格修复)  — 零风险
Step 3: encoder.py + STPBlock.py (分辨率提升)      — 中风险（OOM）
Step 4: aef_module.py (detach修复 + no_grad + 移除interpolate) — 高风险（训练动态）
Step 5: loss_function.py (精简 + set_stage + edge case) — 中风险
Step 6: training.py (分阶段 + DDP修复 + target分辨率) — 中风险
Step 7: train.py (CLI参数 + 基础修复)               — 低风险
Step 8: data.py (valid_periods预转tensor)           — 低风险
Step 9: 冒烟测试
```

---

## 五、回退方案

1. **OOM 回退**：
   - batch_size 4 → 2 + gradient_accumulation_steps=2
   - 或 precision pathway 保持 64×64，仅 final resample 上采样到 128×128
   
2. **Loss 不稳定回退**：
   - 逐步重新启用 raw_uniformity (weight=1.0) 或 variance (weight=1.0)

3. **detach 修复后训练异常**：
   - 回退到 reconstruction 基于 student embedding（mu_s）计算

4. **Git 回退**：单 commit，`git revert` 一键恢复

---

## 六、冒烟测试计划

```bash
# 1. 快速前向测试
python -c "
import torch
from src.aef.architecture.aef_module import AlphaEarthFoundations
model = AlphaEarthFoundations()
x = {'sentinel2': torch.randn(2, 4, 128, 128, 5)}
ts = {'sentinel2': torch.randint(0, 1000, (2, 4))}
vp = torch.tensor([[0, 1000], [0, 1000]], dtype=torch.float32)
out = model(x, ts, vp)
print('embeddings:', out['embeddings'].shape)      # 期望 (2, 128, 128, 64)
print('recon:', out['reconstructions']['sentinel2'].shape)  # 期望 (2, 1, 128, 128, 5)
"

# 2. Loss 分阶段测试
python -c "
from src.aef.loss_function import AEFLoss
loss_fn = AEFLoss()
loss_fn.set_stage('distill_align')
print('Stage 1:', loss_fn.reconstruction_weight, loss_fn.distill_weight)
loss_fn.set_stage('normal')
print('Stage 2:', loss_fn.reconstruction_weight, loss_fn.distill_weight)
"

# 3. 单卡 10 steps 训练
python src/aef/train.py --batch-size 2 --max-steps 10 --save-every 5 --eval-every 5 --distill-warmup-steps 5
```

---

## 七、预期结果

| 指标 | 改动前 | 改动后预期 |
|------|--------|-----------|
| Encoder 梯度流 | ❌ 被 detach 阻断 | ✅ 正常回传 |
| Embedding 分辨率 | 64×64 | **128×128** |
| VMF kappa | 10 (可学习) | **8000 (固定)** |
| Learned upsample | 仅 2× 可学习 | **任意倍数可学习** |
| 棋盘格伪影 | 有 | **消除** |
| Loss 项数 | 12 | **5** |
| 阶段 1 distill weight | 0.5 | **5.0** |
| 阶段 2 reconstruction weight | 0.5 | **1.0** |
| Teacher 显存占用 | 与 Student 叠加 | **降低 3-4GB** |
| Save 频率 | 默认 | **每 500 step** |
| Warmup 步数 | 无 | **1000 steps** |

---

*审查修正版计划制定完毕，请审核后确认执行。*
