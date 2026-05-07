#!/usr/bin/env python3
"""
变化检测可行性验证 - 使用真实标注数据验证模型能力
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.insert(0, "/workspace/xuannv")
sys.path.insert(0, "/workspace/AEF")

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = "/workspace/raw/harbin_scenes"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
CKPT_PATH = "/workspace/outputs/aef_qwen_v1/epoch_399.pt"
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"

# 时间窗口
TIME_WINDOWS = {
    "2023Q3-2024Q2": (1688169600000.0, 1719792000000.0),
    "2024Q3-2025Q2": (1719792000000.0, 1751328000000.0),
    "2023全年": (1672531200000.0, 1703980800000.0),
    "2025全年": (1735689600000.0, 1767225600000.0),
}

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
print("="*60)
print("加载模型和数据...")
print("="*60)

from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

cfg = load_config(CONFIG_PATH)
device = torch.device("npu:0")
model = AEFModel(cfg).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

print(f"  模型: {sum(p.numel() for p in model.parameters())/1e6:.1f}M 参数")
print(f"  数据: {len(dataset.patches)} patches")

# ──────────────────────────────────────────
# 加载标注 (需要转换 CRS 到 grid 的 EPSG:32652)
# ──────────────────────────────────────────
print("\n加载标注数据...")
all_changes = []
for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
    shp_path = f"{ANNOT_DIR}/{shp_name}"
    try:
        gdf = gpd.read_file(shp_path)
        # 转换 CRS 到 UTM Zone 52N (EPSG:32652)
        if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
            print(f"  {shp_name}: {len(gdf)} 个多边形 (CRS 已转换: EPSG:{gdf.crs.to_epsg()})")
        else:
            print(f"  {shp_name}: {len(gdf)} 个多边形")
        for _, row in gdf.iterrows():
            all_changes.append({
                "geometry": row.geometry,
                "period": shp_name.replace(".shp", ""),
            })
    except Exception as e:
        print(f"  {shp_name}: 错误 - {e}")

print(f"  总计: {len(all_changes)} 个变化多边形")

# ──────────────────────────────────────────
# 加载 Grid 并建立映射
# ──────────────────────────────────────────
print("\n加载 Grid...")
with open(GRID_PATH) as f:
    grid_data = json.load(f)

patch_to_rect = {}
for feat in grid_data["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_to_rect[pid] = {
        "minx": min(xs), "maxx": max(xs),
        "miny": min(ys), "maxy": max(ys),
    }

# ──────────────────────────────────────────
# 将标注光栅化到每个 patch
# ──────────────────────────────────────────
print("\n光栅化标注到 patch 级别...")

def rasterize_change_to_patch(change_polygons, patch_id, grid_size=64):
    """将变化多边形光栅化到指定 patch 的 64x64 网格."""
    rect = patch_to_rect.get(patch_id)
    if not rect:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    H, W = grid_size, grid_size
    resolution = (rect["maxx"] - rect["minx"]) / H  # 每像素多少米

    change_mask = np.zeros((H, W), dtype=np.float32)

    for poly_info in change_polygons:
        geom = poly_info["geometry"]
        if geom is None:
            continue
        # 确保在同一 CRS
        try:
            # 检查是否与 patch 有交集
            patch_box = box(rect["minx"], rect["miny"], rect["maxx"], rect["maxy"])
            if not geom.intersects(patch_box):
                continue

            # 光栅化
            from shapely.ops import transform
            # 将坐标转为像素坐标
            def to_pixel(x, y):
                px = int((x - rect["minx"]) / resolution)
                py = int((rect["maxy"] - y) / resolution)  # Y 翻转
                return (px, py)

            # 采样多边形内部点
            bounds = geom.bounds
            for px in range(max(0, int((bounds[0] - rect["minx"]) / resolution) - 1),
                           min(H, int((bounds[2] - rect["minx"]) / resolution) + 2)):
                for py in range(max(0, int((rect["maxy"] - bounds[3]) / resolution) - 1),
                               min(W, int((rect["maxy"] - bounds[1]) / resolution) + 2)):
                    # 像素中心坐标 (世界坐标)
                    wx = rect["minx"] + (px + 0.5) * resolution
                    wy = rect["maxy"] - (py + 0.5) * resolution
                    from shapely.geometry import Point
                    if geom.contains(Point(wx, wy)):
                        change_mask[px, py] = 1.0
        except:
            pass

    return change_mask


# ──────────────────────────────────────────
# 提取 embedding
# ──────────────────────────────────────────
def extract_emb(patch_idx, valid_start_ms, valid_end_ms):
    batch = dataset[patch_idx]
    batch["valid_start_ms"] = torch.tensor(valid_start_ms, dtype=torch.float64)
    batch["valid_end_ms"] = torch.tensor(valid_end_ms, dtype=torch.float64)

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)
        else:
            batch_dev[k] = v

    with torch.no_grad():
        output = model(
            source_frames=batch_dev["source_frames"],
            source_timestamps_ms=batch_dev["source_timestamps_ms"],
            source_frame_mask=batch_dev["source_frame_mask"],
            source_input_mask=batch_dev["source_input_mask"],
            source_type_ids=batch_dev["source_type_ids"],
            valid_start_ms=batch_dev["valid_start_ms"],
            valid_end_ms=batch_dev["valid_end_ms"],
            target_relative_time=batch_dev["target_relative_time"],
            target_metadata=batch_dev["target_metadata"],
        )
    emb_map = output.embedding_map  # [1, D, H, W]
    emb_map = F.normalize(emb_map, p=2, dim=1)
    return emb_map[0].cpu().numpy()


def cosine_distance_map(eb, ea):
    D, H, W = eb.shape
    fb = eb.reshape(D, -1); fa = ea.reshape(D, -1)
    nb = np.linalg.norm(fb, axis=0, keepdims=True)
    na = np.linalg.norm(fa, axis=0, keepdims=True)
    fb = fb / np.maximum(nb, 1e-8); fa = fa / np.maximum(na, 1e-8)
    cos_sim = np.sum(fb * fa, axis=0)
    return ((1.0 - cos_sim) / 2.0).reshape(H, W)


# ──────────────────────────────────────────
# 主实验: 在有标注的 patch 上验证变化检测
# ──────────────────────────────────────────
print("\n" + "="*60)
print("实验: 在有光学标注的 patch 上验证变化检测")
print("="*60)

# 找有标注的 patches
patch_changes = {}  # patch_id -> list of change polygons
for ch in all_changes:
    geom = ch["geometry"]
    if geom is None:
        continue
    for pid, rect in patch_to_rect.items():
        patch_box = box(rect["minx"], rect["miny"], rect["maxx"], rect["maxy"])
        if geom.intersects(patch_box):
            if pid not in patch_changes:
                patch_changes[pid] = []
            patch_changes[pid].append(ch)

# 按变化面积排序，选前10个
patch_change_area = []
for pid, changes in patch_changes.items():
    total_area = sum(c["geometry"].area for c in changes if c["geometry"])
    patch_change_area.append((pid, total_area, changes))

patch_change_area.sort(key=lambda x: x[1], reverse=True)
test_patches = patch_change_area[:15]  # 测试前15个

print(f"\n找到 {len(patch_changes)} 个有标注的 patch")
print(f"测试前 {len(test_patches)} 个 (按变化面积排序):")
for pid, area, changes in test_patches:
    print(f"  {pid}: {area/1e4:.1f} 公顷, {len(changes)} 个多边形")

# 提取 before/after embedding
print("\n提取 embedding...")
results = []
for i, (pid, area, changes) in enumerate(test_patches):
    pidx = dataset.patches.index(pid) if pid in dataset.patches else -1
    if pidx < 0:
        print(f"  {pid}: 不在数据集中，跳过")
        continue

    print(f"  [{i+1}/{len(test_patches)}] {pid}... ", end="", flush=True)

    t0 = time.time()
    try:
        # Before: 2023Q3-2024Q2
        emb_before = extract_emb(pidx, *TIME_WINDOWS["2023Q3-2024Q2"])
        # After: 2024Q3-2025Q2
        emb_after = extract_emb(pidx, *TIME_WINDOWS["2024Q3-2025Q2"])

        cd = cosine_distance_map(emb_before, emb_after)

        # 光栅化标注
        change_mask = rasterize_change_to_patch(changes, pid)

        # 计算 AUC
        flat_cd = cd.flatten()
        flat_mask = change_mask.flatten()

        # 只统计有有效标注的区域
        has_label = flat_mask > 0
        no_label = flat_mask == 0

        if has_label.sum() > 10 and no_label.sum() > 10:
            auc = roc_auc_score(flat_mask, flat_cd)
            ba = balanced_accuracy_score(flat_mask > 0.5, flat_cd > np.median(flat_cd))
            elapsed = time.time() - t0
            print(f"AUC={auc:.3f}, BA={ba:.3f}, mean_dist={np.mean(cd):.4f}, {elapsed:.1f}s")
            results.append({
                "patch_id": pid,
                "auc": auc,
                "balanced_acc": ba,
                "mean_dist": float(np.mean(cd)),
                "max_dist": float(np.max(cd)),
                "change_area_ha": area / 1e4,
                "n_changes": len(changes),
            })
        else:
            elapsed = time.time() - t0
            print(f"标注太少, mean_dist={np.mean(cd):.4f}, {elapsed:.1f}s")
            results.append({
                "patch_id": pid,
                "auc": None,
                "balanced_acc": None,
                "mean_dist": float(np.mean(cd)),
                "max_dist": float(np.max(cd)),
                "change_area_ha": area / 1e4,
                "n_changes": len(changes),
            })
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

# ──────────────────────────────────────────
# 结果汇总
# ──────────────────────────────────────────
print("\n" + "="*60)
print("结果汇总")
print("="*60)

valid_results = [r for r in results if r["auc"] is not None]
if valid_results:
    aucs = [r["auc"] for r in valid_results]
    bas = [r["balanced_acc"] for r in valid_results]
    mean_dists = [r["mean_dist"] for r in valid_results]

    print(f"\n有效 patch 数: {len(valid_results)}/{len(results)}")
    print(f"AUC 均值: {np.mean(aucs):.3f} (std={np.std(aucs):.3f})")
    print(f"AUC 中位数: {np.median(aucs):.3f}")
    print(f"BA 均值: {np.mean(bas):.3f}")
    print(f"Mean Dist: {np.mean(mean_dists):.4f}")

    # 分类统计
    good = [r for r in valid_results if r["auc"] > 0.7]
    ok = [r for r in valid_results if 0.55 < r["auc"] <= 0.7]
    bad = [r for r in valid_results if r["auc"] <= 0.55]

    print(f"\nAUC > 0.7: {len(good)} 个")
    for r in good:
        print(f"  {r['patch_id']}: AUC={r['auc']:.3f}, dist={r['mean_dist']:.4f}")

    print(f"\n0.55 < AUC <= 0.7: {len(ok)} 个")
    for r in ok:
        print(f"  {r['patch_id']}: AUC={r['auc']:.3f}, dist={r['mean_dist']:.4f}")

    print(f"\nAUC <= 0.55: {len(bad)} 个")
    for r in bad:
        print(f"  {r['patch_id']}: AUC={r['auc']:.3f}, dist={r['mean_dist']:.4f}")

    # 与 V6 baseline 对比
    print(f"\n与原版 V6 对比:")
    print(f"  V6 无监督 AUC: ~0.50 (随机)")
    print(f"  aef_qwen AUC:  {np.mean(aucs):.3f}")

    if np.mean(aucs) > 0.6:
        print(f"\n✅ 模型能做变化检测! (AUC > 0.6)")
    elif np.mean(aucs) > 0.55:
        print(f"\n⚠️  有一定变化检测能力，但需要改进 (AUC ~0.55)")
    else:
        print(f"\n❌ 变化检测能力不足 (AUC ~0.50)")
else:
    print("\n❌ 没有有效结果")

print("\n" + "="*60)
print("实验完成")
print("="*60)
