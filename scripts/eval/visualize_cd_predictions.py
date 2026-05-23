#!/usr/bin/env python3
"""可视化 CD Head 变化检测预测结果.

用法:
    python scripts/eval/visualize_cd_predictions.py \
        --config configs/exp_v2_A_skipL2_50ep_0520.yaml \
        --backbone /workspace/outputs/exp_v2_A_skipL2_50ep_0520/epoch_best_epoch48.pt \
        --cd-head /workspace/outputs/exp_v2_A_skipL2_50ep_0520/cd_head_v12_best.pt \
        --device npu:0
"""
import sys, os, argparse
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box, Point

ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
OUTPUT_DIR = "/workspace/outputs/exp_v2_A_skipL2_50ep_0520/visualization"

# 2025年period窗口
PERIOD_MONTHS = {
    "june":        (4, 6),
    "aug":         (6, 8),
    "September":   (8, 9),
    "October":     (9, 10),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--cd-head", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--max-patches", type=int, default=20)
    p.add_argument("--use-pre-norm", action="store_true")
    return p.parse_args()


def load_models(args):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.models.heads import ChangeDetectionHeadV3
    from src.data.dataset import HarbinPatchDataset
    from src.inference.engine import extract_embedding_for_month

    cfg = load_config(args.config)
    device = args.device

    # Backbone
    model = AEFModel(cfg).to(device)
    ckpt = torch.load(args.backbone, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # CD Head
    head_ckpt = torch.load(args.cd_head, map_location=device, weights_only=False)
    head_cfg = head_ckpt["config"]
    cd_head = ChangeDetectionHeadV3(
        embedding_dim=head_cfg["embedding_dim"],
        hidden_dim=head_cfg["hidden_dim"],
        dropout=head_cfg.get("dropout", 0.3),
    )
    cd_head.load_state_dict(head_ckpt["cd_head"])
    cd_head.to(device)
    cd_head.eval()

    # Dataset
    cfg.data.preload = True
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False

    return model, cd_head, dataset, cfg, extract_embedding_for_month


def load_annotations():
    with open(GRID_PATH) as f:
        grid_data = __import__('json').load(f)
    patch_bounds = {}
    for feat in grid_data["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

    patch_changes = {}
    for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
        period = shp_name.replace(".shp", "")
        try:
            gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 32652:
                gdf = gdf.to_crs(epsg=32652)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                pid = row.get("patch_id", None)
                if not pid:
                    # 空间匹配找到所属patch
                    pt = geom.centroid
                    for candidate_pid, bounds in patch_bounds.items():
                        if bounds[0] <= pt.x <= bounds[2] and bounds[1] <= pt.y <= bounds[3]:
                            pid = candidate_pid
                            break
                if pid:
                    patch_changes.setdefault(pid, {}).setdefault(period, []).append(geom)
            print(f"  {shp_name}: {len(gdf)} polygons")
        except Exception as e:
            print(f"  Skip {shp_name}: {e}")
    return patch_changes, patch_bounds


def rasterize_annotations(changes, bounds, H=64, W=64):
    resolution = (bounds[2] - bounds[0]) / W
    resolution_y = (bounds[3] - bounds[1]) / H
    mask = np.zeros((H, W), dtype=np.float32)
    for geom in changes:
        for row in range(H):
            for col in range(W):
                wx = bounds[0] + (col + 0.5) * resolution
                wy = bounds[3] - (row + 0.5) * resolution_y
                if geom.contains(Point(wx, wy)):
                    mask[row, col] = 1.0
    return mask


def visualize_patch(model, cd_head, dataset, extract_fn, device, pid, period, before_m, after_m, bounds, changes, output_dir, use_pre_norm=False):
    """为单个patch和period生成可视化图."""
    try:
        eb = extract_fn(model, dataset, pid, 2025, before_m, device, normalize=True, use_pre_norm=use_pre_norm)
        ea = extract_fn(model, dataset, pid, 2025, after_m, device, normalize=True, use_pre_norm=use_pre_norm)
    except Exception as e:
        print(f"  Skip {pid} {period}: {e}")
        return None

    # CD Head 推理
    with torch.no_grad():
        eb_t = torch.from_numpy(eb).unsqueeze(0).float().to(device)
        ea_t = torch.from_numpy(ea).unsqueeze(0).float().to(device)
        pred = torch.sigmoid(cd_head(eb_t, ea_t)).cpu().numpy()[0, 0]

    # Ground truth mask
    mask = rasterize_annotations(changes, bounds) if changes else np.zeros((64, 64))

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    im = ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'CD Prediction\n{pid} {period}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im = ax.imshow(mask, cmap='Greens', vmin=0, vmax=1)
    ax.set_title(f'Ground Truth\nn_changed={int(mask.sum())}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    overlay = np.zeros((*pred.shape, 3))
    overlay[..., 0] = pred  # Red = prediction
    overlay[..., 1] = mask  # Green = GT
    ax.imshow(overlay)
    ax.set_title('Overlay (R=Pred, G=GT)')
    ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{pid}_{period}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "pid": pid,
        "period": period,
        "pred_mean": float(pred.mean()),
        "pred_max": float(pred.max()),
        "mask_sum": int(mask.sum()),
        "auc": None,  # 单patch需要足够正负样本才能算AUC
    }


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  CD Head 变化检测可视化")
    print("=" * 60)

    model, cd_head, dataset, cfg, extract_fn = load_models(args)
    print(f"  Backbone loaded: {args.backbone}")
    print(f"  CD Head loaded: {args.cd_head}")

    patch_changes, patch_bounds = load_annotations()
    valid_patches = [pid for pid in patch_changes if pid in dataset.patches]
    print(f"  有效标注 patch: {len(valid_patches)}")

    # 限制数量
    valid_patches = valid_patches[:args.max_patches]

    results = []
    for pid in valid_patches:
        for period, changes in patch_changes[pid].items():
            before_m, after_m = PERIOD_MONTHS[period]
            bounds = patch_bounds.get(pid)
            if bounds is None:
                continue
            res = visualize_patch(model, cd_head, dataset, extract_fn, args.device, pid, period, before_m, after_m, bounds, changes, OUTPUT_DIR, args.use_pre_norm)
            if res:
                results.append(res)
                print(f"  {pid} {period}: pred_mean={res['pred_mean']:.3f}, pred_max={res['pred_max']:.3f}, mask={res['mask_sum']}")

    # 保存汇总
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        __import__('json').dump(results, f, indent=2)

    print(f"\n  可视化完成: {len(results)} 张图")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  汇总: {summary_path}")


if __name__ == "__main__":
    main()
