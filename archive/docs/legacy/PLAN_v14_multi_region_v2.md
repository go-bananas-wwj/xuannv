# V14 多区域混合训练 v2 — 最终执行计划

> 已确认决策：
> - ❌ 不使用齐齐哈尔数据
> - ✅ 云筛选保存到 `s2_cloud_filtered/`
> - ✅ 静态目标只用 DEM，权重=0.05
> - ✅ NPU 0,1,2,3,4,5（6卡）

---

## 一、最终可用数据

| 区域 | S2 Patches | S1 Patches | Landsat | DEM | 状态 |
|------|-----------|-----------|---------|-----|------|
| 哈尔滨（云筛选后） | 425 | 424 | 407 | 424 | ✅ 就绪 |
| 大庆（待云筛选） | 309 | 400 | 400 | 400 | ⚠️ 需处理 |
| 海淀（待云筛选） | 400 | 400 | 400 | 400 | ⚠️ 需处理 |
| **总计** | **1,134** | **1,224** | **1,207** | **1,224** | |

**vs 哈尔滨单区域**: 2.67× patches

---

## 二、Phase 0: 数据准备（详细步骤）

### P0.1 S2云筛选（大庆 + 海淀）

**命令**:
```bash
conda activate xuannv
cd /workspace/xuannv

# 大庆
python scripts/preprocessing/filter_cloudy_frames.py \
    --input_dir /workspace/raw/phase2_heilongjiang/daqing/s2 \
    --output_dir /workspace/raw/phase2_heilongjiang/daqing/s2_cloud_filtered \
    --max-per-month 2 \
    --cloud-threshold 0.3 \
    --workers 16

# 海淀
python scripts/preprocessing/filter_cloudy_frames.py \
    --input_dir /workspace/raw/phase2_heilongjiang/haidian/s2 \
    --output_dir /workspace/raw/phase2_heilongjiang/haidian/s2_cloud_filtered \
    --max-per-month 2 \
    --cloud-threshold 0.3 \
    --workers 16
```

**预期输出**:
- 大庆: ~20-25帧/patch
- 海淀: ~20-25帧/patch

### P0.2 计算全局统计量

**命令**:
```bash
# 哈尔滨已有统计量: /workspace/statistics/harbin_scenes/
# 需为大庆、海淀单独计算

python scripts/preprocessing/compute_statistics.py \
    --data_root /workspace/raw/phase2_heilongjiang/daqing \
    --sources s2 s1 landsat dem \
    --output_dir /workspace/statistics/daqing \
    --workers 16

python scripts/preprocessing/compute_statistics.py \
    --data_root /workspace/raw/phase2_heilongjiang/haidian \
    --sources s2 s1 landsat dem \
    --output_dir /workspace/statistics/haidian \
    --workers 16
```

**输出文件**:
```
/workspace/statistics/daqing/
  ├── s2_stats.json
  ├── s1_stats.json
  ├── landsat_stats.json
  └── dem_stats.json

/workspace/statistics/haidian/
  ├── s2_stats.json
  ├── s1_stats.json
  ├── landsat_stats.json
  └── dem_stats.json
```

### P0.3 创建多区域 manifest

**文件**: `configs/multi_region_manifest.json`
```json
{
  "harbin": {
    "data_root": "/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered",
    "stats_dir": "/workspace/statistics/harbin_scenes",
    "patches": ["patch_000000", "patch_000001", ...],
    "s2_count": 425,
    "has_dynamic_world": true,
    "has_jrc_water": true
  },
  "daqing": {
    "data_root": "/workspace/raw/phase2_heilongjiang/daqing",
    "stats_dir": "/workspace/statistics/daqing",
    "patches": ["patch_000000", ..., "patch_000308"],
    "s2_count": 309,
    "has_dynamic_world": false,
    "has_jrc_water": false
  },
  "haidian": {
    "data_root": "/workspace/raw/phase2_heilongjiang/haidian",
    "stats_dir": "/workspace/statistics/haidian",
    "patches": ["patch_000000", ..., "patch_000399"],
    "s2_count": 400,
    "has_dynamic_world": false,
    "has_jrc_water": false
  }
}
```

### P0.4 数据集适配（修改点）

**文件**: `src/data/dataset.py`

**修改1**: 多区域数据加载
```python
class MultiRegionPatchDataset(HarbinPatchDataset):
    """支持多区域混合加载."""
    
    def __init__(self, cfg, split="train", ...):
        # 加载manifest
        manifest_path = cfg.data.multi_region_manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        
        # 合并所有patch
        self.all_patches = []
        for region, info in self.manifest.items():
            for patch_id in info['patches']:
                self.all_patches.append((region, patch_id))
        
        self.num_samples = len(self.all_patches)  # 1134
```

**修改2**: 区域感知的源解析
```python
def _resolve_source_dir(self, source_name, patch_id, region):
    """根据区域解析数据源目录."""
    region_root = self.manifest[region]['data_root']
    
    # S2云筛选后的路径
    if source_name == 's2' and region != 'harbin':
        s2_dir = Path(region_root) / 's2_cloud_filtered' / patch_id
        if s2_dir.exists():
            return s2_dir
        # fallback到原始s2
        return Path(region_root) / 's2' / patch_id
    
    return Path(region_root) / source_name / patch_id
```

**修改3**: 区域感知的统计量加载
```python
def _load_stats(self, region, source_name):
    """加载对应区域的统计量."""
    stats_dir = self.manifest[region]['stats_dir']
    stats_path = Path(stats_dir) / f"{source_name}_stats.json"
    with open(stats_path) as f:
        return json.load(f)
```

**修改4**: 缺失静态目标处理
```python
def _load_target_frame(self, target_source, patch_id, region):
    """加载目标帧，处理区域间缺失的静态目标."""
    # DEM和WorldCover：所有区域都有
    if target_source in ['dem', 'worldcover']:
        return super()._load_target_frame(target_source, patch_id)
    
    # DynamicWorld / JRCWater：仅哈尔滨有
    if target_source in ['dynamic_world', 'jrc_water']:
        if not self.manifest[region].get('has_dynamic_world', False):
            # 返回全零 + target_mask=False
            return np.zeros((self.num_classes, self.image_size, self.image_size)), False
        return super()._load_target_frame(target_source, patch_id)
```

**修改5**: 预加载缓存分区域
```python
# 各区域独立缓存
cache_paths = {
    'harbin': '/workspace/outputs/cache_harbin_v14.pt',
    'daqing': '/workspace/outputs/cache_daqing_v14.pt',
    'haidian': '/workspace/outputs/cache_haidian_v14.pt',
}
```

### P0.5 验证

```bash
python scripts/test_multi_region_data.py \
    --manifest configs/multi_region_manifest.json \
    --sample 20
```

**验证项**:
- [ ] 各区域patch数量正确
- [ ] S2云筛选后帧数合理（15-30帧）
- [ ] 统计量加载正确
- [ ] 缺失源（大庆S2的91个patch）正确处理
- [ ] 缺失静态目标（phase2无DW/JRC）weight=0

---

## 三、Phase 1: 训练器实现

### P1.1 共享参数 Teacher-Student

**文件**: `src/training/ddp_v14_trainer.py`

```python
class DDPv14Trainer:
    """V14: 共享参数Teacher-Student + 多区域混合 + Coding Rate Loss."""
    
    def __init__(self, cfg, local_rank):
        self.model = AEFModel(cfg).to(device)
        self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        
        # ❌ 不再创建EMA Teacher
        # self.teacher = copy.deepcopy(self.model.module)
        
        self.optimizer = build_optimizer(self.model.module, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
    
    def train_step(self, batch):
        """共享参数前向: 教师(完整) + 学生(扰动)."""
        
        # --- Teacher Forward ---
        teacher_out = self.model(
            source_frames=batch['source_frames'],
            source_timestamps_ms=batch['source_timestamps_ms'],
            source_frame_mask=batch['source_frame_mask'],
            source_input_mask=batch['source_input_mask'],
            source_type_ids=batch['source_type_ids'],
            valid_start_ms=batch['valid_start_ms'],
            valid_end_ms=batch['valid_end_ms'],
            target_relative_time=batch['target_relative_time'],
            target_metadata=batch['target_metadata'],
        )
        
        # --- Student Forward (扰动输入) ---
        student_frames, student_frame_mask, student_input_mask = \
            self._build_student_view(
                batch['source_frames'],
                batch['source_frame_mask'],
                batch['source_input_mask'],
                cfg.training.student_frame_drop_rate,
                cfg.training.student_source_drop_rate,
            )
        
        student_out = self.model(
            source_frames=student_frames,
            source_timestamps_ms=batch['source_timestamps_ms'],
            source_frame_mask=student_frame_mask,
            source_input_mask=student_input_mask,
            source_type_ids=batch['source_type_ids'],
            valid_start_ms=batch['valid_start_ms'],
            valid_end_ms=batch['valid_end_ms'],
            target_relative_time=batch['target_relative_time'],
            target_metadata=batch['target_metadata'],
        )
        
        # --- Losses ---
        loss_recon_teacher = self._compute_reconstruction(teacher_out, batch)
        loss_recon_student = self._compute_reconstruction(student_out, batch)
        loss_recon = (loss_recon_teacher + loss_recon_student) / 2
        
        loss_consist = consistency_loss(
            teacher_out.embedding,      # [B, D]
            student_out.embedding       # [B, D]
        )
        
        # Gather for uniformity / coding rate
        gathered_emb = gather_all(teacher_out.pre_norm_embedding)
        
        loss_unif = batch_uniformity_loss_l2(gathered_emb)
        loss_cr = coding_rate_loss(gathered_emb, eps=0.01)
        loss_var = variance_regularizer(gathered_emb)
        loss_cov = covariance_loss(gathered_emb)
        
        total_loss = (
            cfg.training.reconstruction_weight * loss_recon +
            cfg.training.consistency_weight * loss_consist +
            cfg.training.batch_uniformity_weight * loss_unif +
            cfg.training.coding_rate_weight * loss_cr +
            cfg.training.variance_weight * loss_var +
            cfg.training.covariance_weight * loss_cov
        )
        
        return total_loss, {
            'recon': loss_recon.item(),
            'consist': loss_consist.item(),
            'unif': loss_unif.item(),
            'cr': loss_cr.item(),
            'var': loss_var.item(),
            'cov': loss_cov.item(),
        }
```

### P1.2 Coding Rate Loss

**文件**: `src/training/losses.py`

```python
def coding_rate_loss(embeddings: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """MCR² Coding Rate Loss.
    
    R(Z; eps) = 0.5 * log det(I + d/(N*eps^2) * ZZ^T)
    最大化编码率 → 对低秩施加无限惩罚。
    """
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    
    N, D = embeddings.shape
    # 中心化
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    # 协方差
    cov = (emb_centered.T @ emb_centered) / N
    
    I = torch.eye(D, device=embeddings.device, dtype=embeddings.dtype)
    # 加上eps防止奇异
    logdet = torch.logdet(I + (D / (N * eps**2 + 1e-8)) * cov + 1e-6 * I)
    
    return -0.5 * logdet
```

### P1.3 有效秩监控

```python
def effective_rank(embeddings: torch.Tensor) -> float:
    """计算有效秩 (Roy & Vetterli 2007)."""
    s = torch.linalg.svdvals(embeddings).cpu().numpy()
    p = s / np.sum(s)
    entropy = -(p * np.nan_to_num(np.log(p), neginf=0)).sum()
    return float(np.exp(entropy))

def information_abundance(embeddings: torch.Tensor) -> float:
    """信息丰度 = ||sigma||_1 / ||sigma||_inf."""
    s = torch.linalg.svdvals(embeddings)
    return (s.sum() / s.max()).item()
```

---

## 四、Phase 2: 实验配置

### 实验设计（6卡分2组）

| Group | NPU | 实验名 | 核心变量 | epochs |
|-------|-----|--------|----------|--------|
| **A** | 0,1,2 | v14_multi_shared_ts | 共享TS + CodingRate + 3区域 | 100 |
| **A** | 0,1,2 | v14_multi_baseline | EMA Teacher对照（同数据） | 100 |
| **B** | 3,4,5 | v14_multi_high_batch | 共享TS + accumulation=4 | 100 |
| **B** | 3,4,5 | v14_harbin_only | 共享TS + 仅哈尔滨 | 100 |

> 注：6卡分4个实验，每个实验实际只用1-2卡。如需加速，可改为每组3卡。

### 核心配置

```yaml
# v14_multi_shared_ts.yaml
data:
  batch_size: 4
  image_size: 128
  max_frames: 32
  num_samples: 1134
  num_input_sources: 3
  num_target_sources: 2          # S2, S1, Landsat, DEM (只有DEM是静态)
  multi_region_manifest: /workspace/xuannv/configs/multi_region_manifest.json
  preload: true
  temporal_window_augmentation: true
  temporal_window_prob: 0.5
  
  target_sources:
    - name: s2
      sensor_src: s2
      out_channels: 6
      loss_type: 0
    - name: s1
      sensor_src: s1
      out_channels: 2
      loss_type: 0
    - name: landsat
      sensor_src: landsat
      out_channels: 6
      loss_type: 0
    - name: dem
      sensor_src: null
      out_channels: 1
      loss_type: 0

model:
  embedding_dim: 64
  num_blocks: 8
  vmf_kappa: 5000
  skip_l2_norm_training: false
  
training:
  epochs: 100
  lr: 0.0001
  lr_min: 1.0e-06
  lr_schedule: cosine_no_restart
  warmup_epochs: 10
  grad_clip_norm: 1.0
  
  # 共享TS配置
  consistency_weight: 0.05
  student_frame_drop_rate: 0.5
  student_source_drop_rate: 0.3
  student_front_drop_prob: 0.15
  student_back_drop_prob: 0.15
  
  # 反坍缩
  batch_uniformity_weight: 0.1
  coding_rate_weight: 0.1          # 新增
  variance_weight: 0.5
  covariance_weight: 0.1
  
  # 重建
  reconstruction_weight: 1.0
  source_recon_weights: [1.0, 1.0, 1.0, 0.05]   # S2, S1, Landsat, DEM
  
  # 高batch配置（仅v14_multi_high_batch）
  gradient_accumulation_steps: 4
```

---

## 五、Phase 3: 启动命令

### 启动脚本

```bash
# ===== Group A: NPU 0,1,2 =====
export ASCEND_RT_VISIBLE_DEVICES=0,1,2

# A1: 共享TS + 多区域
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp_v14.py \
    --config configs/v14_multi_shared_ts.yaml \
    --save-every 20

# A2: EMA对照
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp_v14.py \
    --config configs/v14_multi_baseline.yaml \
    --save-every 20

# ===== Group B: NPU 3,4,5 =====
export ASCEND_RT_VISIBLE_DEVICES=3,4,5

# B1: 高batch
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp_v14.py \
    --config configs/v14_multi_high_batch.yaml \
    --save-every 20

# B2: 哈尔滨单区域
torchrun --nproc_per_node=3 \
    scripts/train/train_ddp_v14.py \
    --config configs/v14_harbin_only.yaml \
    --save-every 20
```

---

## 六、监控指标

### 训练日志必须包含

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `recon` | < 0.3 | > 0.5 |
| `consist` | 0.05-0.15 | > 0.2 |
| `unif` | 0.3-0.7 | < 0.2 (坍缩) |
| `cr` | > -10 | 正数 (严重坍缩) |
| `erank` | > 32 (64×0.5) | < 20 |
| `active_dims` | > 40/64 | < 20 |

### 每20 epoch验证

```bash
python scripts/eval/validate_v14.py \
    --checkpoint /workspace/outputs/v14_multi_shared_ts/epoch_20.pt \
    --region harbin   # 单独评估哈尔滨

python scripts/eval/validate_v14.py \
    --checkpoint /workspace/outputs/v14_multi_shared_ts/epoch_20.pt \
    --region daqing   # 单独评估大庆
```

---

## 七、时间线

| 阶段 | 任务 | 时长 | 产出 |
|------|------|------|------|
| **P0.1** | S2云筛选（大庆+海淀） | 4-6h | s2_cloud_filtered/ |
| **P0.2** | 计算统计量 | 2-3h | statistics/daqing, haidian/ |
| **P0.3** | 创建manifest | 0.5h | multi_region_manifest.json |
| **P0.4** | 数据集适配 | 4-6h | dataset.py修改 |
| **P0.5** | 验证数据 | 1h | 验证通过 |
| **P1** | 训练器实现 | 1-2天 | ddp_v14_trainer.py |
| **P2** | 配置+启动脚本 | 0.5天 | 4个yaml + launch脚本 |
| **P3** | 训练执行 | 3-4天 | 4个实验各100epoch |
| **P4** | 下游评估 | 1天 | 评估报告 |
| | **总计** | **~7-9天** | |

---

## 八、关键决策确认清单

```
□ 不使用齐齐哈尔数据
□ 云筛选保存到 s2_cloud_filtered/
□ 静态目标只用 DEM，权重=0.05
□ NPU 0,1,2,3,4,5（6卡），留6,7给同事
□ 4个实验：共享TS + EMA对照 + 高batch + 哈尔滨单区域
□ 训练100 epoch
□ 先完成数据准备，再启动训练
```

---

**请确认以上计划，我立即开始执行 Phase 0 数据准备！**
