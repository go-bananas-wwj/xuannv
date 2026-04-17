"""常量定义 — 时间窗口、颜色表、路径等."""
from pathlib import Path

# ── 基础路径 ──
PROJECT_ROOT = Path("/workspace/xuannv")
RAW_DIR = Path("/workspace/raw/harbin_scenes")
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")
ANNOT_DIR = Path("/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件")

# ── 模型注册表 ──
MODEL_REGISTRY = {
    "v1": {
        "display_name": "V1 基线 (反坍缩)",
        "config": PROJECT_ROOT / "configs" / "qwen_v1_scenes.yaml",
        "checkpoint": Path("/workspace/outputs/aef_qwen_v1/epoch_399.pt"),
        "embeddings_dir": Path("/workspace/outputs/aef_qwen_v1/embeddings"),
        "epochs": 400,
        "desc": "skip-L2 + raw uniformity 反坍缩基线",
    },
    "v2": {
        "display_name": "V2 时序对比",
        "config": PROJECT_ROOT / "configs" / "qwen_v2_temporal.yaml",
        "checkpoint": Path("/workspace/outputs/aef_qwen_v2/best.pt"),
        "embeddings_dir": Path("/workspace/outputs/aef_qwen_v2/embeddings"),
        "epochs": 500,
        "desc": "引入 temporal_contrastive_loss，增强时序敏感性",
    },
    "v3": {
        "display_name": "V3 双窗口增强",
        "config": PROJECT_ROOT / "configs" / "qwen_v3_temporal.yaml",
        "checkpoint": Path("/workspace/outputs/aef_qwen_v3/epoch_599.pt"),
        "embeddings_dir": Path("/workspace/outputs/aef_qwen_v3/embeddings"),
        "epochs": 600,
        "desc": "非重叠双窗口 + 时序对比权重增强",
    },
    "v2_hr_finetune": {
        "display_name": "V2-HR 安全微调",
        "config": PROJECT_ROOT / "configs" / "qwen_v2_hr_finetune.yaml",
        "checkpoint": Path("/workspace/outputs/aef_qwen_v2_hr_finetune/best.pt"),
        "embeddings_dir": Path("/workspace/outputs/aef_qwen_v2_hr_finetune/embeddings"),
        "epochs": 100,
        "desc": "V2 基线 + 高分影像安全微调 (lr=1e-5)",
    },
    "v2_hr_from_scratch": {
        "display_name": "V2-HR 五源从头训练",
        "config": PROJECT_ROOT / "configs" / "qwen_v2_hr_from_scratch.yaml",
        "checkpoint": Path("/workspace/outputs/aef_qwen_v2_hr_from_scratch/best.pt"),
        "embeddings_dir": Path("/workspace/outputs/aef_qwen_v2_hr_from_scratch/embeddings"),
        "epochs": 400,
        "desc": "5 源独立编码器 + 2025 月度数据从头训练",
    },
}

# ── MLP 下游结果根目录 ──
MLP_OUTPUT_BASE = Path("/workspace/outputs")


# ── 时间窗口预设 ──
TIME_WINDOWS = {
    # 月度
    "2023-01": (1672531200000.0, 1675209600000.0),
    "2023-02": (1675209600000.0, 1677628800000.0),
    "2023-03": (1677628800000.0, 1680307200000.0),
    "2023-04": (1680307200000.0, 1682899200000.0),
    "2023-05": (1682899200000.0, 1685577600000.0),
    "2023-06": (1685577600000.0, 1688169600000.0),
    "2023-07": (1688169600000.0, 1690848000000.0),
    "2023-08": (1690848000000.0, 1693526400000.0),
    "2023-09": (1693526400000.0, 1696118400000.0),
    "2023-10": (1696118400000.0, 1698796800000.0),
    "2024-01": (1704067200000.0, 1706745600000.0),
    "2024-02": (1706745600000.0, 1709251200000.0),
    "2024-03": (1709251200000.0, 1711929600000.0),
    "2024-04": (1711929600000.0, 1714521600000.0),
    "2024-05": (1714521600000.0, 1717200000000.0),
    "2024-06": (1717200000000.0, 1719792000000.0),
    "2024-07": (1719792000000.0, 1722470400000.0),
    "2024-08": (1722470400000.0, 1725148800000.0),
    "2024-09": (1725148800000.0, 1727740800000.0),
    "2024-10": (1727740800000.0, 1730419200000.0),
    "2025-01": (1735689600000.0, 1738368000000.0),
    "2025-02": (1738368000000.0, 1740787200000.0),
    "2025-03": (1740787200000.0, 1743465600000.0),
    "2025-04": (1743465600000.0, 1746057600000.0),
    "2025-05": (1746057600000.0, 1748736000000.0),
    "2025-06": (1748736000000.0, 1751328000000.0),
    "2025-07": (1751328000000.0, 1754006400000.0),
    "2025-08": (1754006400000.0, 1756684800000.0),
    "2025-09": (1756684800000.0, 1759276800000.0),
    "2025-10": (1759276800000.0, 1761955200000.0),
    # 季度
    "2023 Q1-Q2": (1672531200000.0, 1688169600000.0),
    "2023 Q3-Q4": (1688169600000.0, 1703980800000.0),
    "2024 Q1-Q2": (1704067200000.0, 1719792000000.0),
    "2024 Q3-Q4": (1719792000000.0, 1735603200000.0),
    "2025 Q1-Q2": (1735689600000.0, 1751328000000.0),
    # 年度
    "2023 全年": (1672531200000.0, 1703980800000.0),
    "2024 全年": (1704067200000.0, 1735603200000.0),
    "2025 全年": (1735689600000.0, 1767225600000.0),
}

# ── WorldCover 颜色表 ──
WORLDCOVER_COLORS = {
    10: (65, 155, 223),   # Tree cover
    20: (57, 125, 73),    # Shrubland
    30: (136, 176, 83),   # Grassland
    40: (255, 187, 34),   # Cropland
    50: (255, 255, 76),   # Built-up
    60: (187, 85, 29),    # Bare / sparse vegetation
    70: (222, 222, 222),  # Snow and ice
    80: (170, 170, 170),  # Permanent water bodies
    90: (120, 80, 20),    # Herbaceous wetland
    95: (140, 140, 140),  # Mangroves
    100: (100, 100, 100), # Moss and lichen
}

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

SOURCE_DISPLAY_NAMES = {
    "s2": "Sentinel-2",
    "s1": "Sentinel-1",
    "landsat": "Landsat",
    "s2_hr": "高分光学 (S2-HR)",
    "s1_hr": "高分雷达 (S1-HR)",
    "dem": "DEM",
    "worldcover": "WorldCover",
    "dynamic_world": "Dynamic World",
    "jrc_water": "JRC Water",
}
