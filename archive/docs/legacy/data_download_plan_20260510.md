# 数据下载计划 — 海淀区（2025-2026）+ 哈尔滨补充

> **目标区域**: 海淀区全区域（430.77 km²）、哈尔滨新区（补充2025年数据）  
> **时间范围**: 2025-01-01 ~ 2026-05-31（约17个月）  
> **状态**: 待用户确认后执行  
> **文档路径**: `/workspace/xuannv/docs/data_download_plan_20260510.md`

---

## 一、存储空间评估

```
/workspace 总容量: 3.5 TB
已使用: 1.9 TB
可用: 1.5 TB

现有数据:
  /workspace/raw/harbin_scenes/    30 GB
  /workspace/outputs/              139 GB
  /workspace/statistics/           ~1 GB
  /workspace/embeddings/           待统计

本次下载预计新增: ~200-300 GB
下载后可用空间: > 1.2 TB ✅ 充足
```

---

## 二、数据集清单与优先级

### 优先级说明
- 🔴 **P0（必须）**: 核心训练/检测数据，无替代方案
- 🟡 **P1（重要）**: 验证/标签数据，显著影响模型质量
- 🟢 **P2（可选）**: 辅助数据，有替代方案

---

### 海淀区数据集

#### 1. Sentinel-1 GRD — SAR主数据源 🔴 P0

| 属性 | 详情 |
|------|------|
| 来源 | ESA Copernicus Open Access Hub / Google Earth Engine |
| 产品等级 | GRD (Ground Range Detected) |
| 成像模式 | IW (Interferometric Wide Swath) |
| 分辨率 | 20m (range) × 22m (azimuth) |
| 极化方式 | VV + VH (双极化) |
| 时间范围 | 2025-01-01 ~ 2026-05-31 |
| 重访周期 | ~12天（单星），~6天（A/B双星叠加）|
| 覆盖要求 | 海淀区全区域（AOI: E116°03'-116°23', N39°53'-40°09'）|

**数据量估算**:
```
时间跨度: 17个月 ≈ 515天
每轨道周期: ~6天（双星叠加后）
总景数: 515 / 6 ≈ 86 景
单景大小: ~1.0 GB (zip压缩后)
总大小: 86 × 1.0 GB ≈ 86 GB

存储路径: /workspace/raw/haidian/s1/
子目录结构: s1/{year}/{month}/S1A_IW_GRDH_*.zip
```

**下载方式**:
```
方案A: sentinelsat（推荐，Python API）
  - 安装: pip install sentinelsat
  - 需注册Copernicus账号获取用户名/密码
  - 支持按AOI、时间、产品类型筛选
  - 支持断点续传

方案B: Google Earth Engine (GEE)
  - 安装: pip install earthengine-api
  - 需Google账号 + GEE项目授权
  - 导出到Google Drive或Cloud Storage
  - 优势: 可云端预处理（去噪、配准）

方案C: Copernicus Data Space Ecosystem (CDSE) — 新平台
  - 网址: https://dataspace.copernicus.eu/
  - SentinelHub兼容API
  - 新用户推荐
```

**下载脚本框架**:
```python
# sentinelsat示例
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import datetime

api = SentinelAPI('username', 'password', 'https://scihub.copernicus.eu/dhus')

# 海淀区AOI (GeoJSON)
footprint = geojson_to_wkt(read_geojson('haidian_aoi.geojson'))

products = api.query(
    footprint,
    date=(datetime.date(2025, 1, 1), datetime.date(2026, 5, 31)),
    platformname='Sentinel-1',
    producttype='GRD',
    sensoroperationalmode='IW',
    polarisationmode='VV VH'
)

api.download_all(products, directory_path='/workspace/raw/haidian/s1/')
```

**预计耗时**: 下载86景 × 1GB ≈ **6-12小时**（取决于网络带宽，可后台运行）

---

#### 2. Sentinel-2 L2A — 光学辅助数据 🔴 P0

| 属性 | 详情 |
|------|------|
| 来源 | ESA Copernicus / GEE |
| 产品等级 | L2A (大气校正后地表反射率) |
| 分辨率 | 10m (B02-B04-B08), 20m (B05-B07-B8A-B11-B12), 60m (B01-B09-B10) |
| 波段 | 13个波段 |
| 时间范围 | 2025-01-01 ~ 2026-05-31 |
| 重访周期 | ~5天（双星S2A+S2B）|

**数据量估算**:
```
时间跨度: 17个月 ≈ 515天
每轨道周期: ~5天
总景数: 515 / 5 ≈ 103 景
单景大小: ~600 MB (L2A, zip压缩)
总大小（原始）: 103 × 600 MB ≈ 62 GB

云筛选后保留: 约30%（北京冬季云多）
筛选后大小: ~20 GB

存储路径: /workspace/raw/haidian/s2/
子目录结构: s2/{year}/{month}/S2A_MSIL2A_*.zip
```

**下载方式**: 同Sentinel-1（sentinelsat / GEE / CDSE）

**额外要求**:
```
- 需要下载QA60波段（云掩码）用于云筛选
- 建议筛选云覆盖率 < 50% 的影像
- 优先保留生长季影像（4-10月）
```

**预计耗时**: 下载103景 × 600MB ≈ **8-16小时**

---

#### 3. Landsat-8/9 L2 — 光学补充数据 🟡 P1

| 属性 | 详情 |
|------|------|
| 来源 | USGS Earth Explorer / GEE |
| 产品等级 | Level-2 (Surface Reflectance) |
| 分辨率 | 30m |
| 波段 | 7个光学波段 + QA |
| 时间范围 | 2025-01-01 ~ 2026-05-31 |
| 重访周期 | 16天 |

**数据量估算**:
```
时间跨度: 17个月
总景数: 515 / 16 ≈ 32 景
单景大小: ~1.2 GB (包含QA和辅助波段)
总大小: 32 × 1.2 GB ≈ 38 GB

存储路径: /workspace/raw/haidian/landsat/
```

**下载方式**:
```
方案A: landsatxplore (Python库)
  - 安装: pip install landsatxplore
  或 pip install landsat-utils

方案B: GEE直接导出
  代码: ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

方案C: USGS M2M API
  - 需USGS账号
  - 支持批量下载
```

**预计耗时**: 下载32景 × 1.2GB ≈ **4-8小时**

---

#### 4. Copernicus DEM — 地形数据 🟡 P1

| 属性 | 详情 |
|------|------|
| 来源 | Copernicus DEM (ECMWF) |
| 分辨率 | 30m |
| 格式 | GeoTIFF |
| 时间 | 静态（2010-2015年间采集）|

**数据量估算**:
```
海淀区面积: 430 km²
30m分辨率: 每像元900 m²
总像元数: 430,000,000 / 900 ≈ 478,000
单波段GeoTIFF: ~2 MB
总大小: < 10 MB

存储路径: /workspace/raw/haidian/dem/
```

**下载方式**:
```
方案A: AWS Open Data (推荐)
  - s3://copernicus-dem-30m/
  - 使用boto3或aws cli按tile下载

方案B: GEE
  代码: ee.Image("NASA/NASADEM_HGT/001")

方案C: 官网
  https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
```

**预计耗时**: **< 5分钟**

---

#### 5. ESA WorldCover 2021 — 粗标签基线 🟡 P1

| 属性 | 详情 |
|------|------|
| 来源 | ESA WorldCover |
| 分辨率 | 10m |
| 类别 | 11类土地覆盖 |
| 时间 | 2021年（静态）|

**数据量估算**:
```
单波段GeoTIFF（430km² @10m）: ~50 MB

存储路径: /workspace/raw/haidian/worldcover/
```

**下载方式**:
```
方案A: 官网直接下载
  https://worldcover2021.esa.int/

方案B: GEE
  代码: ee.ImageCollection("ESA/WorldCover/v200")

方案C: AWS
  s3://esa-worldcover/
```

**预计耗时**: **< 5分钟**

---

#### 6. Microsoft TEMPO — 季度建筑验证 🟡 P1

| 属性 | 详情 |
|------|------|
| 来源 | Microsoft AI for Good |
| 时间范围 | Q1 2020 ~ Q2 2025（季度）|
| 分辨率 | 37.6m |
| 内容 | 建筑密度 + 建筑高度 |
| 格式 | Cloud-Optimized GeoTIFF (COG) |

**数据量估算**:
```
海淀区覆盖范围: 需要约 2-4 个quad tiles
每个tile: ~50-100 MB
季度数: 2025Q1-Q4 + 2026Q1-Q2 = 6个季度
总大小: 6 × 4 × 75 MB ≈ 1.8 GB

存储路径: /workspace/raw/haidian/tempo/
```

**下载方式**:
```
GitHub: https://github.com/microsoft/buildings
Azure Blob Storage直接下载

按quad索引下载海淀区范围内的tiles
```

**预计耗时**: **1-2小时**

---

#### 7. CLCD (China Land Cover Dataset) — 中国年度土地覆盖 🟢 P2

| 属性 | 详情 |
|------|------|
| 来源 | 武汉大学 |
| 分辨率 | 30m |
| 时间 | 1990-2022年（年度，如有更新版可用）|
| 类别 | 9类 |

**数据量估算**:
```
单年GeoTIFF（海淀区范围）: ~10 MB
下载3年: 2020, 2021, 2022 ≈ 30 MB

存储路径: /workspace/raw/haidian/clcd/
```

**下载方式**:
```
官网: https://zenodo.org/records/5205676
或 GEE: 用户上传的CLCD资产
```

**预计耗时**: **< 10分钟**

---

#### 8. 吉林一号耕地变化检测数据集 — 预训练数据 🟢 P2

| 属性 | 详情 |
|------|------|
| 来源 | 长光卫星 / 吉林一号网 |
| 数据量 | 8000余组（训练6000 + 测试2000）|
| 分辨率 | < 0.75m |
| 图像尺寸 | 256×256像素 |
| 标签 | 8类耕地变化 + 1类非变化 |

**数据量估算**:
```
每组: 2时相 × 3通道TIF + 标签PNG ≈ 3 MB
8000组: 8000 × 3 MB ≈ 24 GB

存储路径: /workspace/datasets/jilin1_farmland_cd/
```

**下载方式**:
```
步骤:
  1. 访问 https://www.jl1mall.com/
  2. 注册教育用户（需.edu邮箱或学校认证）
  3. 进入【遥感商城】→【资源中心】→【大赛数据集】
  4. 下载"耕地变化检测数据集"

替代方式:
  联系吉林一号网客服获取FTP/网盘链接
```

**预计耗时**: **6-12小时**（取决于网盘速度）

---

### 哈尔滨新区补充数据集

#### 9. 哈尔滨 2025年补充数据 🔴 P0（如有缺失）

**现状检查**:
```
当前harbin数据:
  - S2: 2023-01 ~ 2025-10（云筛选后）
  - S1: 2023-01 ~ 2025-10
  - Landsat: 2023-01 ~ 2025-10

是否需要补充:
  - 2025-11 ~ 2026-05: 约7个月
  - S1: ~35景 × 1GB = ~35 GB
  - S2: ~42景 × 600MB = ~25 GB（原始）
  - Landsat: ~13景 × 1.2GB = ~16 GB
```

**下载方式**: 同海淀区Sentinel/Landsat下载流程

---

## 三、下载工具安装计划

### 需要安装的工具

```bash
# 基础环境（conda xuannv）
conda activate xuannv

# 1. sentinelsat — Sentinel-1/2下载
pip install sentinelsat

# 2. earthengine-api — Google Earth Engine
pip install earthengine-api
# 需额外运行: earthengine authenticate

# 3. landsatxplore — Landsat下载
pip install landsatxplore

# 4. boto3 — AWS S3数据访问（Copernicus DEM）
pip install boto3

# 5. requests / wget — 通用下载
pip install requests tqdm

# 6. geopandas / shapely — AOI处理
pip install geopandas shapely
```

### 账号注册需求

| 平台 | 用途 | 注册难度 | 预计时间 |
|------|------|---------|---------|
| Copernicus Open Access Hub | Sentinel-1/2下载 | 简单（邮箱注册）| 5分钟 |
| Google Earth Engine | 云端处理+下载 | 中等（需Gmail + 项目申请）| 1-3天 |
| USGS Earth Explorer | Landsat下载 | 简单（邮箱注册）| 5分钟 |
| 吉林一号网 | 耕地变化数据集 | 中等（需教育用户认证）| 1-3天 |

---

## 四、下载执行计划

### 阶段1: 环境准备（1天）

| 任务 | 内容 | 预计时间 |
|------|------|---------|
| 1.1 | 安装所有Python下载库 | 30分钟 |
| 1.2 | 注册Copernicus账号 | 10分钟 |
| 1.3 | 注册USGS账号 | 10分钟 |
| 1.4 | 申请GEE项目授权（如需要）| 1-3天 |
| 1.5 | 注册吉林一号网教育用户 | 1-3天 |
| 1.6 | 创建海淀区AOI GeoJSON | 30分钟 |
| 1.7 | 创建目录结构 | 10分钟 |

### 阶段2: 高优先级数据下载（3-5天）

```
Day 1: Sentinel-1 GRD（海淀区，2025-2026）
  - 启动后台下载脚本
  - 预计6-12小时完成

Day 2: Sentinel-2 L2A（海淀区，2025-2026）
  - 启动后台下载脚本
  - 预计8-16小时完成
  - 同步开始云筛选预处理

Day 3: Landsat-8/9 L2（海淀区，2025-2026）
  - 启动后台下载
  - 预计4-8小时完成

Day 4: DEM + WorldCover + CLCD
  - 小文件快速下载
  - 预计1小时内完成

Day 5: Microsoft TEMPO
  - 按quad索引下载海淀区范围
  - 预计1-2小时完成
```

### 阶段3: 中优先级数据下载（1-3天）

```
Day 6-7: 吉林一号耕地变化检测数据集
  - 教育用户审核通过后下载
  - 预计6-12小时

Day 8: 哈尔滨新区补充数据（如需要）
  - 2025-11 ~ 2026-05的S1/S2/Landsat
  - 预计1-2天
```

### 阶段4: 数据校验（1天）

```
Day 9: 完整性检查
  - 核对下载文件数量 vs 预期
  - 检查文件完整性（zip/unzip测试）
  - 统计每类数据的总大小

Day 10: 质量初检
  - Sentinel-1: 检查覆盖范围是否完整
  - Sentinel-2: 云量统计，筛选可用影像
  - 生成数据清单报告
```

---

## 五、目录结构设计

```
/workspace/raw/
├── harbin_scenes/                    # 现有（保持不变）
│   ├── harbin/
│   ├── harbin_scenes/
│   └── harbin_scenes_cloud_filtered/
│
└── haidian/                          # ★ 新增
    ├── s1/                           # Sentinel-1 GRD
    │   ├── 2025/
    │   │   ├── 01/
    │   │   ├── 02/
    │   │   └── ...
    │   └── 2026/
    │       ├── 01/
    │       └── ...
    │
    ├── s2/                           # Sentinel-2 L2A（原始）
    │   ├── 2025/
    │   └── 2026/
    │
    ├── s2_cloud_filtered/            # Sentinel-2（云筛选后）
    │   └── ...
    │
    ├── landsat/                      # Landsat-8/9 L2
    │   └── ...
    │
    ├── dem/                          # Copernicus DEM
    │   └── haidian_dem_30m.tif
    │
    ├── worldcover/                   # ESA WorldCover 2021
    │   └── haidian_worldcover_2021.tif
    │
    ├── tempo/                        # Microsoft TEMPO
    │   ├── 2025Q1/
    │   ├── 2025Q2/
    │   └── ...
    │
    ├── clcd/                         # CLCD年度土地覆盖
    │   └── ...
    │
    ├── aoi/                          # AOI边界
    │   └── haidian_aoi.geojson
    │
    └── manifest.json                 # 数据清单

/workspace/datasets/
└── jilin1_farmland_cd/               # 吉林一号耕地变化数据集
    ├── train/
    ├── test/
    └── README.md

/workspace/statistics/
└── haidian/                          # 海淀区统计数据（下载后计算）
    ├── s1_stats.json
    ├── s2_stats.json
    └── landsat_stats.json
```

---

## 六、时间线总览

| 阶段 | 内容 | 耗时 | 并行度 |
|------|------|------|--------|
| 环境准备 | 安装工具 + 注册账号 | 1天 | 串行 |
| P0数据下载 | S1/S2/Landsat（海淀+哈尔滨）| 3-5天 | 可并行 |
| P1数据下载 | DEM/WorldCover/TEMPO | 1天 | 与P0并行 |
| P2数据下载 | CLCD/吉林一号 | 1-3天 | 可并行 |
| 数据校验 | 完整性+质量检查 | 1天 | 串行 |
| **总计** | | **7-11天** | |

**关键路径**: 环境准备(1天) → S1/S2下载(3-5天) → 校验(1天) = **5-7天**

---

## 七、风险与应对

| 风险 | 影响 | 概率 | 应对方案 |
|------|------|------|---------|
| Copernicus账号下载限速 | 下载时间延长 | 中 | 改用CDSE新平台或GEE导出 |
| S2云量过高导致可用影像少 | 光学数据不足 | 高 | 增加Landsat权重，S1为主 |
| 吉林一号教育用户审核不通过 | 缺少预训练数据 | 低 | 可用OSCD/LEVIR-CD替代 |
| GEE项目申请被拒 | 无法使用GEE下载 | 低 | fallback到sentinelsat直接下载 |
| 网络中断导致下载失败 | 数据不完整 | 中 | 所有脚本加入断点续传和重试机制 |
| 磁盘空间不足 | 无法继续下载 | 低 | 已确认1.5TB可用，监控df |

---

## 八、待确认事项

确认后我立即开始执行：

1. **是否需要补充哈尔滨2025-11~2026-05的数据？**
   - 当前哈尔滨数据到2025-10，是否需要补齐到2026-05？

2. **吉林一号数据集是否必须下载？**
   - 需要教育用户认证（1-3天），是否有.edu邮箱可用？

3. **是否使用Google Earth Engine？**
   - GEE需要Google账号+项目授权，是否愿意申请？
   - 如果不申请，全部走sentinelsat/USGS直接下载

4. **Sentinel-2云量阈值设定？**
   - 建议云覆盖率 < 50%，是否接受？
   - 北京冬季可能几乎没有<50%的影像，是否放宽到<80%？

5. **Landsat是否必须下载？**
   - 如果S2数据充足，可省略Landsat节省空间和时间
   - 但建议保留作为S2云覆盖时的备份

---

*文档版本: V1*  
*生成时间: 2026-05-10*  
*状态: 计划草案，待用户确认后执行*  
*确认前不安装任何工具、不下载任何数据*
