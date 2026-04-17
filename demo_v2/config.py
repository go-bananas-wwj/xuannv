"""Demo 全局配置与模型注册."""
from pathlib import Path
from typing import Dict, Any
import json

from demo_v2.utils.constants import MODEL_REGISTRY, GRID_PATH


def get_model_info(version: str) -> Dict[str, Any]:
    """获取指定版本的模型信息."""
    info = MODEL_REGISTRY.get(version)
    if info is None:
        raise ValueError(f"Unknown model version: {version}")
    return {
        **info,
        "has_embeddings": _check_embeddings(info["embeddings_dir"]),
        "has_checkpoint": info["checkpoint"].exists(),
    }


def _check_embeddings(embeddings_dir: Path) -> bool:
    """检查是否存在预计算 embedding maps."""
    if not embeddings_dir.exists():
        return False
    required = ["embedding_maps.npy", "patch_ids.json"]
    return all((embeddings_dir / f).exists() for f in required)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """列出所有可用模型及其状态."""
    result = {}
    for ver, info in MODEL_REGISTRY.items():
        result[ver] = get_model_info(ver)
    return result


def get_grid_data() -> dict:
    """加载 grid GeoJSON 数据."""
    with open(GRID_PATH) as f:
        return json.load(f)
