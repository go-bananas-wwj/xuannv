# V14 数据处理严重问题报告与修复方案

> 生成时间: 2026-05-18
> 严重程度: 🔴 **CRITICAL** — 直接影响模型训练有效性
> 影响范围: 所有使用当前预处理逻辑的实验（V1~V14）

---

## 一、问题概述

### 1.1 核心发现

**哈尔滨S2预处理后标准差为 0.0000** —— 所有像素值被压缩到同一个常数（约-0.85），输入信号**完全丢失**。

```
预处理后S2: min=-0.8494, max=-0.8493, mean=-0.8494, std=0.0000
```

### 1.2 根因定位

问题出在 **`compute_statistics.py` 与 `normalize_data()` 的步骤不匹配**：

```
当前代码流程（错误）:
  1. 统计量计算: 原始值 mean=1394, std=1641  ← compute_statistics.py
  2. 预处理: log(x+1)/10 → z-score用原始统计量  ← normalize_data()
  
  结果: z = (log(4000)/10 - 1394) / 1641 = (0.84 - 1394) / 1641 = -0.85
  
  log后的值范围只有 [0.75, 0.87]，而统计量是原始值的 [1394±1641]
  两者完全不在一个量级，导致所有值被压缩到同一个常数
```

**正确做法应该是**：
- 方法A：先z-score（用原始统计量），再log
- 方法B：先log，再计算log后的统计量，然后用log统计量做z-score

### 1.3 实际验证数据

| 步骤 | 哈尔滨S2 Band 0 | 范围 | 说明 |
|------|----------------|------|------|
| 原始值 | 3476 ~ 5288 | [3000, 6000] | uint16，反射率×10000 |
| log(x+1)/10 | 0.746 ~ 0.869 | [0.75, 0.87] | 对数压缩后 |
| **当前z-score** | **-0.8494 ~ -0.8493** | **[≈0]** | **❌ 完全坍缩！** |
| 方法1（先z-score再log）| 0.019 ~ 0.133 | [0.02, 0.13] | 有变化但范围窄 |
| 方法2（log统计量）| -2.59 ~ +1.61 | [-2.6, +1.6] | ✅ 健康的标准正态 |

---

## 二、问题详细分析

### 2.1 各区域数据格式差异

| 源 | 哈尔滨 | 大庆 | 海淀 | 问题 |
|------|--------|------|------|------|
| **S2 dtype** | uint16 | float32 | float32 | 格式不一致 |
| **S2 值范围** | 68 ~ 5917 | 0.15 ~ 0.91 | 0.12 ~ 0.62 | 哈尔滨是×10000，大庆/海淀是0-1 |
| **S1 dtype** | float64 | float32 | float32 | 格式不一致 |
| **S1 值范围** | -39.9 ~ +14.3 | 9 ~ 228 | 20 ~ 1269 | 哈尔滨是dB，大庆/海淀是原始DN |
| **Landsat dtype** | uint16 | ? | ? | 待确认 |

### 2.2 跨区域统计量对比

#### S2 统计量

| 区域 | Band 0 mean | Band 0 std | 说明 |
|------|------------|-----------|------|
| 哈尔滨 | 1394.66 | 1641.05 | 原始反射率×10000 |
| 大庆 | 0.52 | 0.16 | **0-1范围，已经是归一化值** |
| 海淀 | 0.12 | 0.04 | **0-1范围，已经是归一化值** |

#### S1 统计量

| 区域 | Band 0 mean | Band 0 std | 说明 |
|------|------------|-----------|------|
| 哈尔滨 | -12.43 | 5.38 | dB值 |
| 大庆 | 82.51 | 43.40 | **原始DN值，不是dB** |
| 海淀 | 178.92 | 122.99 | **原始DN值，不是dB** |

### 2.3 问题清单

| # | 问题 | 严重程度 | 影响 |
|---|------|---------|------|
| 1 | **统计量与预处理步骤不匹配** | 🔴 CRITICAL | S2预处理后std=0，信号完全丢失 |
| 2 | **跨区域S2数据格式不一致** | 🔴 CRITICAL | 哈尔滨uint16(×10000)，大庆/海淀float32(0-1) |
| 3 | **跨区域S1单位不一致** | 🟠 HIGH | 哈尔滨dB，大庆/海淀原始DN，预处理逻辑不适用 |
| 4 | **大庆/海淀S2未做云筛选** | 🟡 MEDIUM | 帧数是哈尔滨5~7倍，训练效率低 |
| 5 | **Landsat从43×43 bilinear resize到128×128** | 🟡 MEDIUM | 30m→10m超分辨率，可能引入插值模糊 |

---

## 三、影响评估

### 3.1 对当前训练的影响

当前V14 6个实验全部受此影响：

- **S2输入信号几乎为零**：模型无法从S2学到有效特征
- **模型被迫依赖S1和Landsat**：但S1在大庆/海淀也有单位问题
- **重建损失recon下降可能是假象**：因为输入没有变化，模型可能学会了恒等映射或其他捷径
- **embedding质量低下**：有效秩(eff_rank)约25，但可能来自S1/Landsat的有限信息

### 3.2 为什么之前没发现？

1. **监控指标没有直接检查输入分布**：只看了recon、eff_rank等间接指标
2. **单点采样时被误认为"均匀patch"**：雪地/云覆盖的patch确实std低，误以为是正常现象
3. **预处理后值在clip范围内**：±6σ clip没有触发异常
4. **模型仍能训练**：因为S1和Landsat还有信号，模型靠它们勉强学习

---

## 四、修复方案

### 方案A：修复统计量计算（推荐，改动最小）

**修改 `scripts/preprocessing/compute_statistics.py`**：

```python
def compute_source_stats(dataset, source_name, max_patches=50):
    all_samples = []
    for patch_id in patch_ids[:max_patches]:
        for tif_path in sample_files:
            data = read_tif(tif_path, dataset.image_size)
            if data is None:
                continue
            
            # ★ 修复：对需要做log变换的源，先log再收集
            if source_name in {"s2", "landsat", "s2_hr"}:
                data = np.log(np.clip(data, 0, None) + 1) / 10.0
            
            # ★ 修复：SAR源先clip再收集
            if source_name in {"s1", "s1_hr"}:
                data = np.clip(data, -30.0, 10.0)
            
            all_samples.append(data)
    
    # 计算统计量（在变换后的值上）
    for c in range(n_channels):
        channel_vals = np.concatenate([s[c].flatten() for s in all_samples])
        mean = float(np.mean(channel_vals))
        std = float(np.std(channel_vals))
        stats[f"band_{c}"] = {"mean": mean, "std": std if std > 1e-8 else 1.0}
```

**效果**：
- 哈尔滨S2 log后的统计量：mean≈0.82, std≈0.03
- z-score后：范围[-2.6, +1.6]，健康标准正态

### 方案B：修改预处理顺序（备选）

**修改 `src/data/transforms.py:normalize_data()`**：

```python
def normalize_data(data, source_name, stats, num_classes=11):
    # 1. 先z-score（用原始统计量）
    source_stats = stats.get(source_name)
    if source_stats is not None:
        for c in range(data.shape[0]):
            mean = source_stats[f"band_{c}"]["mean"]
            std = source_stats[f"band_{c}"]["std"]
            data[c] = (data[c] - mean) / std
    
    # 2. 再log变换
    if source_name in LOG_TRANSFORM_SOURCES:
        data = np.log(np.clip(data, 0, None) + 1) / 10.0
    
    # 3. clip
    data = np.clip(data, -SIGMA_CLIP, SIGMA_CLIP)
    return data
```

**与AEF论文的一致性**：
- 论文说"先log再标准化"，方案B改变了顺序
- 但从效果看，方案B也能产生有效分布
- **推荐方案A**，因为更符合论文描述

### 方案C：统一跨区域数据格式（长期）

对于大庆/海淀的数据格式不一致问题：

**选项C1：在GEE导出时统一**
- 重新导出大庆/海淀S2为uint16(×10000)
- 重新导出大庆/海淀S1为dB值

**选项C2：在代码中适配**
- 在normalize_data中检测数据范围，自动选择预处理路径
- 如果值在[0,1]范围内，先×10000再正常处理
- 如果S1值不在[-50,50]dB范围，先做dB转换

```python
# 自适应预处理示例
def adaptive_normalize(data, source_name, stats):
    # S2: 检测是否已经是0-1范围
    if source_name == "s2" and data.max() < 1.5:
        data = data * 10000  # 还原到原始反射率范围
    
    # S1: 检测是否已经是dB
    if source_name == "s1" and data.max() > 50:
        data = 10 * np.log10(data ** 2) - 83  # 转换为dB
    
    # 然后正常预处理
    ...
```

---

## 五、修复执行步骤

### 步骤1：立即停止当前训练（建议）

当前V14 6个实验的S2输入信号几乎为零，继续训练是浪费算力。

### 步骤2：修复统计量计算脚本

```bash
# 修改 scripts/preprocessing/compute_statistics.py
# 在收集all_samples之前加入变换逻辑
```

### 步骤3：重新计算所有区域的统计量

```bash
conda activate xuannv
cd /workspace/xuannv
python scripts/preprocessing/compute_statistics.py
```

### 步骤4：验证新统计量

```python
import json
with open("/workspace/statistics/harbin/s2_stats.json") as f:
    stats = json.load(f)
print(f"S2 Band 0: mean={stats['band_0']['mean']:.4f}, std={stats['band_0']['std']:.4f}")
# 预期: mean≈0.82, std≈0.03 (log后的值)
```

### 步骤5：验证预处理后的分布

```python
# 采样一个TIFF，验证预处理后是否有有效变化
```

### 步骤6：删除旧缓存，重启训练

```bash
find /workspace/outputs -name "dataset_cache_*.pt" -delete
```

---

## 六、附录：各源完整处理流程（修复后）

### S2（Sentinel-2 光学）

```
原始卫星数据 (GEE已处理)
    ├── 大气校正 (L2A)
    ├── 重投影到 UTM (EPSG:32652)
    ├── 裁剪到 1.28km × 1.28km patch
    ├── 反射率 × 10000 → uint16
    └── 波段选择 (B2,B3,B4,B8,B11,B12)
    
本地预处理
    ├── 云筛选 (哈尔滨已做，大庆/海淀未做)
    ├── log(x+1)/10 变换
    ├── 计算统计量 (在log后的值上)
    ├── z-score: (x - log_mean) / log_std
    └── clip to [-6σ, +6σ]
    
模型输入: [B, 3, 32, 6, 128, 128]
```

### S1（Sentinel-1 SAR）

```
原始卫星数据 (GEE已处理)
    ├── 辐射定标
    ├── 地形校正
    ├── dB转换 (γ = 10·log₁₀(DN²) - 83)  ← 哈尔滨已做，大庆/海淀未做！
    ├── 重投影到 UTM
    └── 裁剪到 patch
    
本地预处理
    ├── clip to [-30, +10] dB
    ├── 计算统计量
    ├── z-score
    └── clip to [-6σ, +6σ]
    
模型输入: [B, 3, 32, 2, 128, 128]
```

### Landsat

```
原始卫星数据 (GEE已处理)
    ├── 大气校正
    ├── 重投影到 UTM
    ├── 反射率 × 10000 → uint16
    └── 裁剪到 patch (30m分辨率 → 43×43)
    
本地预处理
    ├── bilinear resize 43×43 → 128×128
    ├── log(x+1)/10 变换
    ├── 计算统计量 (在log后的值上)
    ├── z-score: (x - log_mean) / log_std
    └── clip to [-6σ, +6σ]
    
模型输入: [B, 3, 32, 6, 128, 128]
```

---

## 七、代码速查

| 文件 | 作用 | 修复点 |
|------|------|--------|
| `scripts/preprocessing/compute_statistics.py` | 计算统计量 | 对log源先log再计算 |
| `src/data/transforms.py:normalize_data()` | 归一化 | 确保使用正确的统计量 |
| `src/data/transforms.py:read_tif()` | 读取TIFF | 无需修改 |
| `src/data/dataset.py:_get_item()` | 采样逻辑 | 无需修改 |
| `scripts/preprocessing/filter_cloudy_frames.py` | 云筛选 | 扩展到大庆/海淀 |

---

*本报告由Agent自动生成，基于对实际数据文件和代码的逐行分析。*
