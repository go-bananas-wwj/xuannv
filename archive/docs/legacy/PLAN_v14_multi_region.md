# V14 多区域混合训练 + 共享参数Teacher-Student 详细执行计划

> 版本: V14 (Round 7 successor)  
> 数据: 哈尔滨 + 大庆 + 海淀 + 齐齐哈尔 = **1,534 patches** (3.6×哈尔滨)  
> 核心改进: 共享参数Teacher-Student + Coding Rate Loss + 大batch  
> 执行方式: 4卡×2实验并行

---

## 一、数据现状

| 区域 | S2 patches | S2帧数(原始) | S1 patches | Landsat | DEM | WorldCover | 云筛选 | 状态 |
|------|-----------|-------------|-----------|---------|-----|-----------|--------|------|
| 哈尔滨 | 425 | ~22 (已筛选) | 424 | 407 | 424 | 424 | ✅ 已完成 | 可用 |
| 大庆 | 309 | ~164 (未筛选) | 400 | 400 | 400 | 400 | ❌ 待做 | 待处理 |
| 海淀 | 400 | ~119 (未筛选) | 400 | 400 | 400 | 400 | ❌ 待做 | 待处理 |
| 齐齐哈尔 | 400 | ~78 (未筛选) | 400 | 400 | 400 | 400 | ❌ 待做 | 待处理 |

**关键发现**:
- 时间格式: 全部为 YYYYMMDD（日度），与哈尔滨一致 ✅
- 波段数: S2=6ch, S1=2ch, Landsat=6ch，与哈尔滨一致 ✅
- 尺寸: 132×132，与哈尔滨一致 ✅
- **phase2数据未云筛选**: S2帧数78-164，远超哈尔滨云筛选后的~22帧，必须筛选
- 大庆S2比其他源少91个patch（309 vs 400），代码已优雅处理缺失源
- 齐齐哈尔S1仅6帧（但patch数量足够）

---

## 二、Phase 0: 数据准备（2-3天）

### P0.1 S2云筛选（大庆/海淀/齐齐哈尔）

**目标**: 将phase2各城市的S2帧按月筛选，保留最clear的2帧/月

```bash
# 大庆
python scripts/preprocessing/filter_cloudy_frames.py \
    --input_dir /workspace/raw/phase2_heilongjiang/daqing/s2 \
    --output_dir /workspace/raw/phase2_heilongjiang/daqing/s2_cloud_filtered \
    --max-per-month 2 --cloud-threshold 0.3 --workers 16

# 海淀
python scripts/preprocessing/filter_cloudy_frames.py \
    --input_dir /workspace/raw/phase2_heilongjiang/haidian/s2 \
    --output_dir /workspace/raw/phase2_heilongjiang/haidian/s2_cloud_filtered \
    --max-per-month 2 --cloud-threshold 0.3 --workers 16

# 齐齐哈尔
python scripts/preprocessing/filter_cloudy_frames.py \
    --input_dir /workspace/raw/phase2_heilongjiang/qiqihar/s2 \
    --output_dir /workspace/raw/phase2_heilongjiang/qiqihar/s2_cloud_filtered \
    --max-per-month 2 --cloud-threshold 0.3 --workers 16
```

**预期**: 筛选后各城市S2约20-30帧/patch

### P0.2 计算全局统计量

```bash
# 创建统计量输出目录
mkdir -p /workspace/statistics/multi_region

# 各区域分别计算（用于后续归一化）
python scripts/preprocessing/compute_statistics.py \
    --regions harbin_scenes_cloud_filtered,phase2_heilongjiang/daqing,phase2_heilongjiang/haidian,phase2_heilongjiang/qiqihar \
    --sources s2 s1 landsat dem worldcover \
    --output_dir /workspace/statistics/multi_region
```

### P0.3 创建多区域manifest

**manifest格式**:
```json
{
  "regions": {
    "harbin": {
      "data_root": "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered",
      "patches": ["patch_000000", "patch_000001", ...],
      "s2_count": 425
    },
    "daqing": {
      "data_root": "/workspace/raw/phase2_heilongjiang/daqing",
      "patches": ["patch_000000", ...],
      "s2_count": 309
    },
    "haidian": { ... },
    "qiqihar": { ... }
  }
}
```

### P0.4 数据集适配修改

**修改点** (src/data/dataset.py):

1. **多区域数据加载**:
```python
class MultiRegionPatchDataset(HarbinPatchDataset):
    """支持多区域混合加载的数据集."""
    
    def __init__(self, cfg, region_manifest_path, ...):
        # 加载manifest
        self.region_manifest = load_manifest(region_manifest_path)
        # 合并所有patch
        self.all_patches = []
        for region, info in self.region_manifest.items():
            for patch_id in info['patches']:
                self.all_patches.append((region, patch_id))
        
    def _resolve_source_dir(self, source_name, patch_id, region):
        """根据区域解析数据源目录."""
        region_root = self.region_manifest[region]['data_root']
        return Path(region_root) / source_name / patch_id
```

2. **区域ID编码**: patch_id格式改为 `{region}_{local_id}`，避免跨区域重复

3. **各区域独立预加载缓存**:
```python
# 按区域分别预加载，最后合并
cache_paths = {
    'harbin': '/workspace/outputs/cache_harbin.pt',
    'daqing': '/workspace/outputs/cache_daqing.pt',
    'haidian': '/workspace/outputs/cache_haidian.pt',
    'qiqihar': '/workspace/outputs/cache_qiqihar.pt',
}
```

### P0.5 验证数据一致性

```bash
# 抽样验证各区域数据格式
python scripts/verify_multi_region_data.py \
    --manifest configs/multi_region_manifest.json \
    --sample 10
```

---

## 三、Phase 1: 训练器架构升级（3-4天）

### P1.1 共享参数 Teacher-Student 训练器

**核心设计**: 教师和学生是**同一个模型**，只是输入不同。一次前向传播跑两次。

```python
class DDPv14Trainer:
    """V14: 共享参数Teacher-Student + 多区域混合训练."""
    
    def __init__(self, cfg, local_rank):
        self.model = AEFModel(cfg).to(device)
        self.model = DDP(self.model, ...)
        # ❌ 不再创建EMA Teacher
        # self.teacher = copy.deepcopy(self.model.module)
        
    def forward_teacher_student(self, batch):
        """共享参数的前向传播: 教师(完整输入) + 学生(扰动输入)."""
        
        # --- Teacher: 完整输入 ---
        teacher_out = self.model(
            source_frames=batch['source_frames'],
            source_timestamps_ms=batch['source_timestamps_ms'],
            source_frame_mask=batch['source_frame_mask'],      # 完整
            source_input_mask=batch['source_input_mask'],      # 完整
            source_type_ids=batch['source_type_ids'],
            valid_start_ms=batch['valid_start_ms'],
            valid_end_ms=batch['valid_end_ms'],
            target_relative_time=batch['target_relative_time'],
            target_metadata=batch['target_metadata'],
        )
        
        # --- Student: 扰动输入（同一模型参数）---
        student_frames, student_frame_mask, student_input_mask, perturb_stats = \
            self._perturb_input(
                batch['source_frames'],
                batch['source_frame_mask'],
                batch['source_input_mask'],
                drop_rate=0.5,          # S2 drop 50%
                source_drop_rate=0.3,    # S1/Landsat drop 30%
            )
        
        student_out = self.model(
            source_frames=student_frames,
            source_timestamps_ms=batch['source_timestamps_ms'],
            source_frame_mask=student_frame_mask,      # 扰动后
            source_input_mask=student_input_mask,      # 扰动后
            source_type_ids=batch['source_type_ids'],
            valid_start_ms=batch['valid_start_ms'],    # 可能与教师不同
            valid_end_ms=batch['valid_end_ms'],
            target_relative_time=batch['target_relative_time'],
            target_metadata=batch['target_metadata'],
        )
        
        return teacher_out, student_out, perturb_stats
    
    def _perturb_input(self, frames, frame_mask, input_mask, drop_rate, source_drop_rate):
        """构建学生视图的扰动输入 — 对齐AEF原文."""
        # Stage 1: 源级别drop (AEF: S2永不drop, S1/Landsat 30%概率)
        # Stage 2: 帧级别drop (S2 50%, S1/Landsat 30%)
        # Stage 3: 前后截断 (各25%概率截断前/后1/4时间)
        ...
```

**关键细节**:
- 梯度同时来自: 教师重建损失 + 学生重建损失 + 一致性损失
- 无需EMA更新，参数天然同步
- 学生valid period可能与教师不同（AEF原文行为）

### P1.2 Coding Rate Loss (MCR²)

```python
def coding_rate_loss(embeddings: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Maximal Coding Rate Reduction损失.
    
    基于率失真理论，对低秩构型施加无限惩罚。
    当embedding坍缩时，logdet -> -inf，损失 -> +inf。
    """
    N, D = embeddings.shape
    # 中心化
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    # 协方差
    cov = (emb_centered.T @ emb_centered) / N  # [D, D]
    I = torch.eye(D, device=embeddings.device, dtype=embeddings.dtype)
    # 编码率: R(Z; eps) = 0.5 * log det(I + d/(N*eps^2) * ZZ^T)
    logdet = torch.logdet(I + (D / (N * eps**2)) * cov)
    return -0.5 * logdet  # 负号：最大化编码率
```

### P1.3 有效秩监控

```python
def effective_rank(embeddings: torch.Tensor) -> float:
    """计算embedding的有效秩."""
    s = torch.linalg.svdvals(embeddings).cpu().numpy()
    p = s / np.sum(s)
    entropy = -(p * np.nan_to_num(np.log(p), neginf=0)).sum()
    return float(np.exp(entropy))

def log_metrics(self, embeddings):
    """每epoch记录有效秩."""
    erank = effective_rank(embeddings)  # [N, D]
    sv_ratio = s[0] / s[-1]  # 条件数
    print(f"erank={erank:.1f}/{D} ({erank/D*100:.1f}%), sv_ratio={sv_ratio:.1f}")
```

### P1.4 损失组合

```python
total_loss = (
    recon_weight * (loss_recon_teacher + loss_recon_student) / 2 +
    uniformity_weight * loss_batch_uniformity +
    consistency_weight * loss_consistency +  # 教师vs学生cosine similarity
    coding_rate_weight * loss_coding_rate +
    variance_weight * loss_vicreg_variance +
    covariance_weight * loss_vicreg_covariance
)
```

---

## 四、Phase 2: 实验设计

### 实验分组（4卡×2组并行）

**Group A: NPU 0-3 (4卡DDP)**

| 实验名 | 核心变量 | 数据 | epochs | 目标 |
|--------|----------|------|--------|------|
| **v14_multi_shared_ts** | 共享参数TS + 多区域 + Coding Rate | 4区域混合 | 100 | 核心验证 |
| **v14_multi_baseline** | 当前EMA Teacher（对照组） | 4区域混合 | 100 | 对比基线 |

**Group B: NPU 4-7 (4卡DDP)**

| 实验名 | 核心变量 | 数据 | epochs | 目标 |
|--------|----------|------|--------|------|
| **v14_multi_high_batch** | 共享TS + accumulation=4 (有效batch=64) | 4区域混合 | 100 | 大batch效果 |
| **v14_harbin_only** | 共享TS（单区域对照） | 仅哈尔滨 | 100 | 验证多区域价值 |

### 关键配置参数

```yaml
# 通用配置
model:
  embedding_dim: 64
  num_blocks: 8
  vmf_kappa: 5000
  skip_l2_norm_training: false
  
data:
  batch_size: 4
  image_size: 128
  max_frames: 32
  num_samples: 1534          # 多区域总patch数
  manifest_path: /workspace/xuannv/configs/multi_region_manifest.json
  preload: true
  temporal_window_augmentation: true
  temporal_window_prob: 0.5
  
training:
  epochs: 100
  lr: 0.0001
  lr_schedule: cosine_no_restart
  warmup_epochs: 10
  grad_clip_norm: 1.0
  
  # 共享TS配置
  consistency_weight: 0.05     # 教师-学生一致性
  student_frame_drop_rate: 0.5
  student_source_drop_rate: 0.3
  student_front_drop_prob: 0.15
  student_back_drop_prob: 0.15
  
  # 反坍缩
  batch_uniformity_weight: 0.1
  coding_rate_weight: 0.1      # 新增
  variance_weight: 0.5
  covariance_weight: 0.1
  
  # 重建
  reconstruction_weight: 1.0
  source_recon_weights: [1.0, 1.0, 1.0, 0.05]  # S2, S1, Landsat, DEM
  
  # 大batch配置（仅v14_multi_high_batch）
  gradient_accumulation_steps: 4   # 有效batch = 4*4*4 = 64
```

---

## 五、Phase 3: 下游评估方案

### 3.1 PEFT CD Head（方案D）

```python
class LoRACDHead(nn.Module):
    """LoRA增强的变化检测头."""
    
    def __init__(self, embedding_dim=64, lora_rank=8):
        super().__init__()
        # LoRA注入: embedding_dim -> lora_rank -> embedding_dim
        self.lora_a = nn.Conv2d(embedding_dim, lora_rank, 1, bias=False)
        self.lora_b = nn.Conv2d(lora_rank, embedding_dim, 1, bias=False)
        nn.init.zeros_(self.lora_b.weight)  # 零初始化，初始时LoRA不影响输出
        
        # 标准CD Head
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )
        self.change_head = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(),
            nn.Conv2d(embedding_dim // 2, 1, 1),
        )
    
    def forward(self, emb_w1, emb_w2):
        # LoRA增强embedding
        emb_w1 = emb_w1 + self.lora_b(self.lora_a(emb_w1))
        emb_w2 = emb_w2 + self.lora_b(self.lora_a(emb_w2))
        
        # 差分编码
        diff = torch.cat([emb_w1, emb_w2], dim=1)
        feat = self.diff_encoder(diff)
        change_logit = self.change_head(feat)
        return change_logit
```

### 3.2 评估流程

1. **Embedding提取**: 多区域各提取embedding
2. **KNN/MLP分类**: 各区域独立评估 + 跨区域评估
3. **变化检测**: 哈尔滨69patch标注做CD AUC
4. **有效秩分析**: erank, 奇异值谱, 条件数

### 3.3 关键评估问题

| 问题 | 验证方法 |
|------|----------|
| 多区域训练是否提升跨区域泛化？ | 在齐齐哈尔/海淀上测试MLP分类 |
| 共享TS是否优于EMA？ | v14_multi_shared_ts vs v14_multi_baseline |
| 大batch是否提升uniformity？ | v14_multi_high_batch的erank vs 其他 |
| Coding Rate是否防止维度坍缩？ | active_dims, erank对比 |

---

## 六、时间线

| 阶段 | 任务 | 时长 | 依赖 |
|------|------|------|------|
| **P0** | 云筛选phase2 S2数据 | 1天 | 无 |
| **P0** | 计算多区域统计量 | 0.5天 | P0.1 |
| **P0** | 创建manifest + 数据集适配 | 1天 | P0.2 |
| **P0** | 验证数据一致性 | 0.5天 | P0.3 |
| **P1** | 共享参数TS训练器 | 2天 | P0 |
| **P1** | Coding Rate Loss + 监控 | 1天 | P1.1 |
| **P1** | LoRA CD Head | 1天 | P1.1 |
| **P2** | 实验配置生成 | 0.5天 | P1 |
| **P3** | 训练执行（4卡×2组） | 3-4天 | P2 |
| **P4** | 下游评估 | 1-2天 | P3 |
| | **总计** | **~10-12天** | |

---

## 七、风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| phase2数据格式不兼容 | 中 | 高 | 提前验证，编写格式转换脚本 |
| NPU内存不足（大batch） | 中 | 中 | 降低accumulation或启用gradient checkpointing |
| 共享TS训练不稳定 | 中 | 高 | 保留EMA对照组，可随时切换 |
| 多区域数据导致过拟合 | 低 | 中 | 监控各区域验证loss，独立评估 |
| 缓存文件过大 | 高 | 中 | 4区域缓存预计4×27GB=108GB，需确保磁盘空间 |

---

## 八、需要你确认的事项

### 确认1: 云筛选策略
```
□ 对大庆/海淀/齐齐哈尔的S2执行云筛选（保留每月最clear的2帧）
□ 云筛选后数据保存到新目录（不覆盖原始数据）
```

### 确认2: 实验范围
```
□ 执行全部4个实验（Group A + Group B）
□ 或仅执行核心实验（v14_multi_shared_ts + v14_harbin_only对照）
□ 或仅执行共享TS实验（跳过baseline对照）
```

### 确认3: 静态目标处理
```
□ phase2有DEM和WorldCover，是否参与重建？
□ 若参与，权重设为多少？（建议DEM=0.05, WorldCover=0.3）
□ phase2没有DynamicWorld和JRCWater，是否跳过？
```

### 确认4: 训练资源
```
□ 8 NPU全部用于训练？
□ 训练最长允许几天？
□ 是否需要保留部分NPU给其他任务？
```

### 确认5: 启动顺序
```
□ 先执行P0数据准备（2-3天），完成后再启动训练
□ 或先启动哈尔滨单区域实验（数据已就绪），同步准备多区域数据
```

---

请确认以上事项，我立即开始执行！
