#!/usr/bin/env python3
"""V7 Level 1: Backbone bare AUC on annotated patches."""
import sys, json, time, argparse
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu  # 必须先导入以注册 NPU 后端
import torch.nn.functional as F
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
CONFIG_PATH = "/workspace/xuannv/configs/xuannv_v7.yaml"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"

# 时间窗口
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)   # 2024Q3-Q4


def load_model(ckpt_path, device):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset

    cfg = load_config(CONFIG_PATH)
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cfg.data.preload = False
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    return model, dataset


def extract_embedding(model, dataset, patch_id, window, device):
    """提取单个 patch 在指定时间窗口的 embedding."""
    try:
        item = dataset[dataset.patches.index(patch_id)]
    except ValueError:
        return None

    # 设置时间窗口
    item["valid_start_ms"] = torch.tensor([window[0]], dtype=torch.float64)
    item["valid_end_ms"] = torch.tensor([window[1]], dtype=torch.float64)

    # 只保留窗口内的帧
    mask = (item["source_timestamps_ms"] >= window[0]) & (item["source_timestamps_ms"] <= window[1])
    item["source_frame_mask"] = mask

    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

    with torch.no_grad():
        out = model(
            source_frames=batch["source_frames"],
            source_timestamps_ms=batch["source_timestamps_ms"],
            source_frame_mask=batch["source_frame_mask"],
            source_input_mask=batch["source_input_mask"],
            source_type_ids=batch["source_type_ids"],
            valid_start_ms=batch["valid_start_ms"],
            valid_end_ms=batch["valid_end_ms"],
            target_relative_time=batch["target_relative_time"],
            target_metadata=batch["target_metadata"],
        )
        emb = out.embedding[0]  # [D]
        emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--device", type=str, default="npu:7", help="设备")
    args = parser.parse_args()

    device = args.device
    ckpt_path = args.checkpoint

    print("="*60)
    print(f"  V7 Level 1: Backbone Bare AUC")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {device}")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    model, dataset = load_model(ckpt_path, device)
    print(f"模型加载完成，Dataset: {len(dataset)} patches")

    # 加载 Grid
    print("\n加载 Grid...")
    with open(GRID_PATH) as f:
        grid_data = json.load(f)

    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    # 加载标注
    print("加载标注数据...")
    all_changes = []
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "MultiPolygon":
                    geom = list(geom.geoms)[0]
                all_changes.append({
                    "geometry": geom,
                    "patch_id": row.get("patch_id", None),
                })
        except Exception as e:
            print(f"  跳过 {shp_name}: {e}")

    print(f"总标注变化图斑: {len(all_changes)}")

    # 统计每个 patch 的标注
    patch_changes = {}
    for change in all_changes:
        if change["patch_id"]:
            pid = change["patch_id"]
            if pid not in patch_changes:
                patch_changes[pid] = []
            patch_changes[pid].append(change["geometry"])
        else:
            # 通过空间匹配找到所属 patch
            pt = change["geometry"].centroid
            for pid, bounds in patch_bounds.items():
                if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                    if pid not in patch_changes:
                        patch_changes[pid] = []
                    patch_changes[pid].append(change["geometry"])
                    break

    # 只保留有标注的 patch
    annotated_patches = [pid for pid in patch_changes if len(patch_changes[pid]) > 0]
    print(f"有标注的 patch 数: {len(annotated_patches)}")

    # 提取 embedding
    print("\n提取 before/after embedding...")
    results = []
    for pid in annotated_patches:
        if pid not in dataset.patches:
            continue

        emb_before = extract_embedding(model, dataset, pid, BEFORE_WINDOW, device)
        emb_after = extract_embedding(model, dataset, pid, AFTER_WINDOW, device)

        if emb_before is None or emb_after is None:
            continue

        # Cosine distance as change score
        cos_sim = np.dot(emb_before, emb_after)
        change_score = 1.0 - cos_sim  # higher = more change

        # 判断是否有变化（任意标注与该 patch 相交）
        has_change = len(patch_changes.get(pid, [])) > 0

        results.append({
            "patch_id": pid,
            "change_score": change_score,
            "has_change": has_change,
            "n_annotations": len(patch_changes.get(pid, [])),
        })

    print(f"成功提取 {len(results)} 个 patch")

    if len(results) < 10:
        print("样本太少，无法计算 AUC")
        return

    # 计算 AUC
    scores = np.array([r["change_score"] for r in results])
    labels = np.array([r["has_change"] for r in results], dtype=int)

    auc = roc_auc_score(labels, scores)

    print("\n" + "="*60)
    print(f"  AUC: {auc:.4f}")
    print(f"  样本数: {len(results)} (正例={labels.sum()}, 负例={len(labels)-labels.sum()})")
    print("="*60)

    # 输出 top 变化 patch
    print("\nTop 10 变化概率:")
    sorted_results = sorted(results, key=lambda x: x["change_score"], reverse=True)
    for r in sorted_results[:10]:
        flag = "✅" if r["has_change"] else "❌"
        print(f"  {r['patch_id']}: score={r['change_score']:.4f} {flag} ({r['n_annotations']} annotations)")


if __name__ == "__main__":
    main()
