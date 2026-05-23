# 完整执行计划 — 数据下载 + OSM训练调研

> **范围**: 海淀区全区域（2025-2026）+ 哈尔滨新区补充  
> **状态**: 待用户最终确认后执行  
> **文档路径**: `/workspace/xuannv/docs/complete_execution_plan_20260510.md`

---

## 第一部分: OSM 能否用于训练？— 调研结论

### 1.1 核心结论

**OSM 可以用于训练，但只能作为"弱标签"（weak labels），不能直接作为 ground truth。**

学术界已有大量研究证实这一点：
- **ETH Zurich (2017)**: "Learning Aerial Image Segmentation from Online Maps" — 用OSM作为弱标签训练语义分割
- **ONERA (2017)**: "Joint Learning from Earth Observation and OpenStreetMap" — OSM作为输入层 + 训练目标，显著提升收敛速度
- **2024年最新综述**: 多篇论文系统性地用OSM预训练建筑和道路提取模型

### 1.2 OSM 用于训练的研究证据

| 研究 | 用途 | 核心发现 |
|------|------|---------|
| ETH Zurich | 建筑/道路/背景三分类 | OSM弱标签训练 vs 人工标签，性能差距可接受 |
| ONERA | 语义分割输入层+目标 | OSM作为额外输入通道，提升准确率和收敛速度 |
| Bonafilia et al. | 全球建筑预训练 | OSM弱标签预训练 + 少量精标微调 = 接近全监督效果 |
| Maggiori et al. | 噪声标签预训练 | **Encoder对标签噪声有鲁棒性**，浅层特征不受噪声影响 |
| SFL (Self-Filtered Learning) | 噪声标签过滤 | 迭代过滤OSM噪声标签，可提升建筑提取性能 |
| CromSS | 跨模态预训练 | OSM噪声标签预训练的encoder，迁移到下游任务有效 |

### 1.3 OSM 训练的优势

```
✅ 免费、全球覆盖
✅ 数据量巨大（弥补精度不足）
✅ 道路数据质量极高（志愿者最积极标注的就是道路）
✅ 城市区域建筑轮廓较完整
✅ 研究证明：大量弱标签 + 少量精标 > 少量精标 alone
```

### 1.4 OSM 训练的限制（关键！）

```
❌ 噪声标签（noisy labels）— 不完整、偏移、错误
❌ 耕地/农田标签几乎不存在（志愿者不标这些）
❌ 编辑时间 ≠ 实际变化时间（无法做变化检测标签）
❌ 城乡结合部覆盖差（海淀区北部耕地分布区）
❌ 直接训练会限制模型性能上限
```

### 1.5 对我们的具体建议

#### 方案A: 作为额外重建目标（推荐）

```
当前AEF模型重建目标: 7类
  ├── 输入3类: S2, S1, Landsat
  └── 静态4类: DEM, WorldCover, DynamicWorld, JRC_Water

可新增: OSM_Raster（第8类重建目标）
  ├── OSM_Building — 建筑轮廓栅格化
  └── OSM_Road — 道路中心线缓冲区栅格化

实现方式:
  1. 下载海淀区OSM矢量数据（Geofabrik / Overpass API）
  2. 栅格化到与S2相同的分辨率（10m）
  3. 作为decoder的一个额外输出通道
  4. 训练时用BCE loss重建OSM栅格

价值:
  - 帮助encoder学习"人造结构"特征
  - 建筑/道路变化是耕地"非农化"的主要去向
  - 不依赖时间对齐（OSM是"当前状态"快照）
```

#### 方案B: 作为辅助输入特征

```
将OSM栅格作为输入的第N个通道:
  输入: [S2_bands, S1_bands, Landsat_bands, OSM_building, OSM_road]

价值:
  - 为模型提供"先验结构信息"
  - ONERA论文证实：OSM输入显著提升语义分割性能

风险:
  - 推理时也需要OSM数据，增加部署复杂度
  - 模型可能过度依赖OSM，降低泛化能力
```

#### 方案C: 预训练专用（最保守）

```
Step 1: 用OSM弱标签预训练一个U-Net或FCN
Step 2: 冻结encoder权重，迁移到AEF模型
Step 3: 用精标数据微调decoder

价值:
  - 充分利用OSM的大规模数据
  - encoder学到robust特征
  - 精标数据只用于微调，节省标注成本

局限:
  - AEF架构特殊（STPBlocks + VMFBottleneck），
    预训练的encoder可能无法直接移植
```

### 1.6 最终建议

| 用途 | 可行性 | 建议 |
|------|--------|------|
| **变化检测标签** | ❌ 不可行 | 时间不匹配，季度级完全不行 |
| **重建目标（decoder）** | ✅ 推荐 | 新增OSM_Building + OSM_Road栅格化目标 |
| **辅助输入通道** | 🟡 可选 | 增加推理复杂度，谨慎使用 |
| **预训练encoder** | 🟡 可选 | 架构差异可能限制迁移效果 |
| **验证/后处理** | ✅ 可用 | OSM建筑覆盖用于验证"建成区变化" |

> **决策建议**: 下载海淀区OSM数据，栅格化后作为AEF模型的**第8个重建目标**（OSM_Building + OSM_Road），帮助模型学习人造结构特征。不用于变化检测标签。

---

## 第二部分: 完整数据下载计划（已确认版）

### 2.1 用户确认清单

| 项目 | 用户确认 | 计划采用 |
|------|---------|---------|
| 哈尔滨补充下载 | ✅ 需要 | 2025-11 ~ 2026-05 |
| 吉林一号数据集 | ✅ 必须 | 教育用户认证后下载 |
| GEE账号 | ✅ 有 | 作为主要下载渠道 |
| S2云量阈值 | ✅ 70% | cloud cover < 70% |
| Landsat | ✅ 必须 | 保留 |
| OSM训练标签 | 调研中 | 作为第8重建目标 |

### 2.2 数据集清单

#### 海淀区（2025-01-01 ~ 2026-05-31）

| # | 数据集 | 来源 | 大小 | 用途 | 下载工具 |
|---|--------|------|------|------|---------|
| 1 | Sentinel-1 GRD | ESA Copernicus | ~86 GB | SAR主数据源 | sentinelsat + GEE |
| 2 | Sentinel-2 L2A | ESA Copernicus | ~62 GB(原始)→~25 GB(筛选) | 光学辅助 | sentinelsat + GEE |
| 3 | Landsat-8/9 L2 | USGS | ~38 GB | 光学备份 | landsatxplore + GEE |
| 4 | Copernicus DEM | AWS/GEE | <10 MB | 地形 | boto3 / GEE |
| 5 | ESA WorldCover | 官网/GEE | ~50 MB | 粗标签 | 直接下载 |
| 6 | Microsoft TEMPO | GitHub/Azure | ~2 GB | 季度建筑验证 | 直链下载 |
| 7 | CLCD | Zenodo | ~30 MB | 年度土地覆盖 | 直接下载 |
| 8 | OSM矢量数据 | Geofabrik/Overpass | ~100 MB | 训练标签/重建目标 | osmnx / overpass |

#### 哈尔滨新区补充（2025-11-01 ~ 2026-05-31）

| # | 数据集 | 大小 | 备注 |
|---|--------|------|------|
| 9 | Sentinel-1 GRD | ~35 GB | 约35景 |
| 10 | Sentinel-2 L2A | ~25 GB(原始) | 约25景，云筛选后~8 GB |
| 11 | Landsat-8/9 L2 | ~16 GB | 约13景 |

#### 通用预训练/验证数据集

| # | 数据集 | 来源 | 大小 | 用途 |
|---|--------|------|------|------|
| 12 | 吉林一号耕地变化 | 吉林一号网 | ~24 GB | 预训练/耕地变化验证 |
| 13 | OSCD | 官网 | ~5 GB | Sentinel-2变化检测基准 |

**总计约: 330 GB**

### 2.3 下载工具安装脚本

```bash
#!/bin/bash
# install_download_tools.sh
# 运行前: conda activate xuannv

echo "=== 安装数据下载工具 ==="

# 1. sentinelsat — Sentinel-1/2下载
pip install sentinelsat

# 2. earthengine-api — Google Earth Engine
pip install earthengine-api
# 运行: earthengine authenticate （需手动交互）

# 3. landsatxplore — Landsat下载
pip install landsatxplore

# 4. boto3 — AWS S3
pip install boto3

# 5. osmnx / osm2geojson — OSM数据获取
pip install osmnx osm2geojson

# 6. geopandas / shapely / rasterio — 地理处理
pip install geopandas shapely rasterio

# 7. 通用工具
pip install requests tqdm

echo "=== 安装完成 ==="
echo "下一步:"
echo "  1. earthengine authenticate"
echo "  2. 注册Copernicus账号 (https://scihub.copernicus.eu)"
echo "  3. 注册USGS账号 (https://earthexplorer.usgs.gov)"
```

### 2.4 账号注册指南

#### ① Copernicus Open Access Hub（Sentinel-1/2）
```
网址: https://scihub.copernicus.eu/
步骤:
  1. 点击 "Login" → "Sign Up"
  2. 填写邮箱、用户名、密码
  3. 邮箱验证（5分钟内收到邮件）
  4. 登录后获取用户名密码（用于sentinelsat API）
耗时: 5分钟
```

#### ② USGS Earth Explorer（Landsat）
```
网址: https://earthexplorer.usgs.gov/
步骤:
  1. 点击 "Register"
  2. 填写邮箱、用户名、密码、机构信息
  3. 邮箱验证
  4. 登录即可使用
耗时: 5分钟
```

#### ③ Google Earth Engine（已有账号，只需认证）
```
命令: earthengine authenticate
步骤:
  1. 终端运行上述命令
  2. 会弹出浏览器窗口或提供URL
  3. 用Google账号登录
  4. 授权Earth Engine访问
  5. 复制授权码回终端
耗时: 2分钟
```

#### ④ 吉林一号网教育用户（重点）
```
网址: https://www.jl1mall.com/
步骤:
  1. 点击右上角 "注册"
  2. 选择 "教育用户注册"
  3. 填写信息:
     - 真实姓名
     - 学校/机构名称
     - .edu邮箱（关键！用于验证）
     - 手机号
     - 设置密码
  4. 提交后等待审核（1-3个工作日）
  5. 审核通过后， edu邮箱会收到通知
  6. 登录后进入:
     【遥感商城】→【资源中心】→【大赛数据集】
     → 找到"耕地变化检测数据集"下载

注意事项:
  - 必须用.edu邮箱，否则无法通过教育用户审核
  - 如没有.edu邮箱，可尝试联系客服说明科研用途
  - 数据集约24GB，建议用下载工具或网盘客户端

替代方案（如审核不通过）:
  - 联系吉林一号网客服 (官网有联系方式)
  - 说明科研项目用途，申请直接下载链接
```

### 2.5 目录结构

```
/workspace/raw/
├── harbin_scenes/                    # 现有（保持不变）
│   ├── harbin/
│   ├── harbin_scenes/
│   └── harbin_scenes_cloud_filtered/
│
├── haidian/                          # ★ 新增: 海淀区
│   ├── s1/                           # Sentinel-1 GRD (2025-2026)
│   │   ├── 2025/
│   │   └── 2026/
│   ├── s2/                           # Sentinel-2 L2A 原始
│   │   ├── 2025/
│   │   └── 2026/
│   ├── s2_cloud_filtered/            # Sentinel-2 云筛选后
│   │   └── ...
│   ├── landsat/                      # Landsat-8/9 L2
│   │   └── ...
│   ├── dem/                          # Copernicus DEM
│   │   └── haidian_dem_30m.tif
│   ├── worldcover/                   # ESA WorldCover 2021
│   │   └── haidian_worldcover_2021.tif
│   ├── tempo/                        # Microsoft TEMPO季度数据
│   │   ├── 2025Q1/
│   │   ├── 2025Q2/
│   │   ├── 2025Q3/
│   │   ├── 2025Q4/
│   │   ├── 2026Q1/
│   │   └── 2026Q2/
│   ├── clcd/                         # CLCD年度土地覆盖
│   │   └── ...
│   ├── osm/                          # ★ OSM数据
│   │   ├── haidian_osm_buildings.gpkg   # 建筑矢量
│   │   ├── haidian_osm_roads.gpkg       # 道路矢量
│   │   ├── haidian_osm_buildings_10m.tif  # 建筑栅格
│   │   └── haidian_osm_roads_10m.tif      # 道路栅格
│   ├── aoi/
│   │   └── haidian_aoi.geojson
│   └── manifest.json                 # 数据清单
│
└── harbin_supplement/                # ★ 新增: 哈尔滨补充
    ├── s1/                           # 2025-11 ~ 2026-05
    ├── s2/
    └── landsat/

/workspace/datasets/
├── jilin1_farmland_cd/               # 吉林一号耕地变化
│   ├── train/
│   ├── test/
│   └── README.md
└── oscd/                             # OSCD基准数据集
    ├── train/
    └── test/

/workspace/statistics/
├── harbin_scenes/                    # 现有
└── haidian/                          # ★ 新增
    ├── s1_stats.json
    ├── s2_stats.json
    └── landsat_stats.json
```

### 2.6 下载执行脚本框架

#### 脚本1: 海淀区 Sentinel-1 下载
```python
#!/usr/bin/env python
# scripts/preprocessing/download_haidian_s1.py
"""下载海淀区Sentinel-1 GRD数据 (2025-2026)"""

import os
from datetime import date
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

# 配置
OUTPUT_DIR = "/workspace/raw/haidian/s1"
AOI_PATH = "/workspace/raw/haidian/aoi/haidian_aoi.geojson"
START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 5, 31)

# 认证（需填入实际账号）
API_USER = "YOUR_COPERNICUS_USERNAME"
API_PASS = "YOUR_COPERNICUS_PASSWORD"

api = SentinelAPI(API_USER, API_PASS, "https://scihub.copernicus.eu/dhus")
footprint = geojson_to_wkt(read_geojson(AOI_PATH))

products = api.query(
    footprint,
    date=(START_DATE, END_DATE),
    platformname="Sentinel-1",
    producttype="GRD",
    sensoroperationalmode="IW",
    polarisationmode="VV VH",
)

print(f"找到 {len(products)} 景 Sentinel-1 数据")
os.makedirs(OUTPUT_DIR, exist_ok=True)
api.download_all(products, directory_path=OUTPUT_DIR)
```

#### 脚本2: 海淀区 Sentinel-2 下载（GEE版）
```python
#!/usr/bin/env python
# scripts/preprocessing/download_haidian_s2_gee.py
"""用GEE下载海淀区Sentinel-2 L2A数据"""

import ee
import os

ee.Initialize()

# 海淀区AOI
aoi = ee.Geometry.Rectangle([116.05, 39.88, 116.38, 40.15])

# 云量筛选 < 70%
collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2025-01-01", "2026-05-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
)

print(f"找到 {collection.size().getInfo()} 景 Sentinel-2 数据")

# 批量导出到Google Drive（然后下载到本地）
# 或使用Export.toAsset导出到GEE Asset
```

#### 脚本3: OSM数据下载与栅格化
```python
#!/usr/bin/env python
# scripts/preprocessing/download_haidian_osm.py
"""下载海淀区OSM数据并栅格化"""

import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

# 配置
AOI_BOUNDS = (116.05, 39.88, 116.38, 40.15)  # minx, miny, maxx, maxy
OUTPUT_DIR = "/workspace/raw/haidian/osm"
RESOLUTION = 10  # 与Sentinel-2对齐

# 下载OSM建筑
print("下载OSM建筑数据...")
buildings = ox.features.features_from_bbox(
    AOI_BOUNDS[3], AOI_BOUNDS[1], AOI_BOUNDS[2], AOI_BOUNDS[0],  # north, south, east, west
    tags={"building": True}
)
buildings.to_file(f"{OUTPUT_DIR}/haidian_osm_buildings.gpkg", driver="GPKG")

# 下载OSM道路
print("下载OSM道路数据...")
roads = ox.features.features_from_bbox(
    AOI_BOUNDS[3], AOI_BOUNDS[1], AOI_BOUNDS[2], AOI_BOUNDS[0],
    tags={"highway": True}
)
roads.to_file(f"{OUTPUT_DIR}/haidian_osm_roads.gpkg", driver="GPKG")

# 栅格化（与S2对齐）
# ... 栅格化代码 ...
```

### 2.7 时间线

```
Day 1: 环境准备
  ├── 安装所有Python库 (30分钟)
  ├── 注册Copernicus账号 (10分钟)
  ├── 注册USGS账号 (10分钟)
  ├── GEE认证 (已有账号，2分钟)
  ├── 注册吉林一号网教育用户 (30分钟，等待审核1-3天)
  └── 创建目录结构 + AOI GeoJSON (30分钟)

Day 2-3: P0遥感数据下载（后台运行）
  ├── 启动海淀区S1下载 (sentinelsat，后台)
  ├── 启动海淀区S2下载 (GEE导出，后台)
  ├── 启动海淀区Landsat下载 (后台)
  ├── 启动哈尔滨补充S1/S2/Landsat下载 (后台)
  └── 同步: 下载DEM + WorldCover + CLCD + TEMPO (快速)

Day 4-5: 数据预处理
  ├── S2云筛选 (cloud cover < 70%，保留可用影像)
  ├── 生成数据清单 manifest.json
  ├── 统计各源mean/std
  └── OSM数据下载 + 栅格化

Day 6-7: 吉林一号 + OSCD（等审核通过）
  ├── 吉林一号耕地变化数据集下载
  ├── OSCD数据集下载（如需要）
  └── 数据完整性校验

Day 8: 最终校验
  ├── 核对所有文件数量和大小
  ├── 抽查影像覆盖范围
  └── 生成下载报告

总计: 8天（其中吉林一号审核可能延长到3-5天）
```

---

## 第三部分: 后续执行计划（数据就绪后）

### Phase 1: 数据预处理（下载完成后立即执行）

| 任务 | 内容 | 预计时间 |
|------|------|---------|
| S2云筛选 | 剔除云量>70%的影像，保留最清晰的 | 4-8小时 |
| 计算统计 | 各源的mean/std，生成stats.json | 2-4小时 |
| Patch划分 | 生成海淀区grid（~430 patches @128×128）| 2小时 |
| 预加载缓存 | 生成训练缓存，加速后续训练 | 4-8小时 |
| OSM栅格化 | 矢量→栅格，与S2对齐 | 1-2小时 |

### Phase 2: 模型迁移与训练

| 任务 | 内容 | 预计时间 |
|------|------|---------|
| 配置创建 | haidian_v1.yaml（含OSM目标） | 2小时 |
| Warm Start | 哈尔滨checkpoint迁移 | 4小时 |
| Zero-shot验证 | 预训练模型直接推理海淀数据 | 4-8小时 |
| 微调训练 | 海淀数据微调（冻结部分层） | 1-2天 |

### Phase 3: 11天高频检测

| 任务 | 内容 | 预计时间 |
|------|------|---------|
| 检测管道 | S1 12天间隔配对 + embedding提取 | 1天 |
| 耕地专项 | 永久基本农田掩码 + 变化检测 | 1天 |
| 效果验证 | 与TEMPO/年度变更调查对比 | 1天 |

---

## 第四部分: 风险与应对

| 风险 | 应对方案 |
|------|---------|
| 吉林一号教育用户审核不通过 | 联系客服说明科研用途；或用OSCD+LEVIR-CD替代 |
| S2云量过高导致可用影像极少 | 阈值已放宽到70%；冬季主要依赖S1+Landsat |
| GEE导出队列拥堵 | fallback到sentinelsat直接下载 |
| 下载中途网络中断 | 所有脚本加入断点续传和重试机制 |
| OSM栅格化与S2配准偏差 | 用同一AOI边界保证空间对齐，后续微调配准 |

---

## 第五部分: 待最终确认事项

你的所有需求已纳入计划，最终确认以下细节即可开始执行：

1. **OSM作为重建目标是否采纳？**
   - 建议: 新增OSM_Building + OSM_Road作为第8个decoder重建目标
   - 好处: 帮助模型学习人造结构特征，不增加推理复杂度
   - 代价: 下载OSM数据 + 栅格化，约半天工作量

2. **吉林一号数据集下载顺序？**
   - 方案A: 等教育用户审核通过后再开始全部下载（可能延迟3-5天）
   - 方案B: 先启动遥感数据下载（不依赖审核），吉林一号通过后单独下载
   - **建议方案B**

3. **是否下载OSCD基准数据集？**
   - OSCD是Sentinel-2变化检测标准benchmark
   - 可用于验证模型在10m分辨率下的变化检测能力
   - 大小约5GB
   - **建议下载**

---

*计划版本: 最终确认版*  
*生成时间: 2026-05-10*  
*状态: 待用户最终确认后执行*  
**确认前不安装工具、不下载数据、不执行代码**
