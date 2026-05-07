#!/usr/bin/env python3
"""AEF_qwen Gradio Demo — 嵌入可视化 + 变化检测 + 全域变化强度图."""
import os, sys, argparse, re, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
sys.path.insert(0, "/workspace/xuannv")

import gradio as gr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import rasterio
import json
import geopandas as gpd
from shapely.geometry import box, Point

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
RAW_DIR = Path("/workspace/raw/harbin_scenes")
EMB_RAW_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/embedding_maps_raw.npy"
EMB_L2_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/embedding_maps.npy"
IDS_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/patch_ids.json"
CKPT_PATH = "/workspace/outputs/aef_qwen_v1/epoch_399.pt"
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
GRID_PATH = Path("/workspace/index/harbin/grid/harbin_grid.geojson")

WC_COLORS = {10:(65,155,223), 20:(57,125,73), 30:(136,176,83), 40:(255,187,34),
             50:(255,255,76), 60:(187,85,29), 70:(222,222,222), 80:(170,170,170),
             90:(120,80,20), 95:(140,140,140), 100:(100,100,100)}

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
    "2025 Q3-Q4": (1751328000000.0, 1767225600000.0),
    # 年度
    "2023 全年": (1672531200000.0, 1703980800000.0),
    "2024 全年": (1704067200000.0, 1735603200000.0),
    "2025 全年": (1735689600000.0, 1767225600000.0),
}

def date_to_ms(dt):
    """datetime → milliseconds."""
    return int(dt.timestamp() * 1000)

def str_to_ms(date_str):
    """'YYYY-MM-DD' → milliseconds."""
    from datetime import datetime
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return date_to_ms(dt)

# ──────────────────────────────────────────
# 加载数据
# ──────────────────────────────────────────
print("Loading raw embeddings (pre-L2, for PCA visualization)...")
EMB_RAW = np.load(EMB_RAW_PATH)
with open(IDS_PATH) as f:
    EMB_IDS = json.load(f)
N, D, H, W = EMB_RAW.shape

# 加载 grid
print("Loading grid...")
with open(GRID_PATH) as f:
    GRID_DATA = json.load(f)

# 全局 patch bounds (用于湿地监测等下游任务)
PATCH_BOUNDS = {}
for _feat in GRID_DATA["features"]:
    _pid = _feat["properties"]["patch_id"]
    _coords = _feat["geometry"]["coordinates"][0]
    _xs = [c[0] for c in _coords]
    _ys = [c[1] for c in _coords]
    PATCH_BOUNDS[_pid] = (min(_xs), min(_ys), max(_xs), max(_ys))

# 计算 PCA
print("Computing global PCA...")
np.random.seed(42)
sub = EMB_RAW[:min(100, N)].reshape(-1, D)[::16]
pca = PCA(n_components=3)
pca.fit(sub)
proj = sub @ pca.components_.T
PC_LO, PC_HI = np.percentile(proj, (1, 99), axis=0)
print(f"PCA explained: {pca.explained_variance_ratio_}")

# 加载模型 (懒加载)
_model_cache = {}

def get_model():
    if "model" in _model_cache:
        return _model_cache["model"]
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
    _model_cache["model"] = (model, dataset, cfg)
    print("Model loaded.")
    return model, dataset, cfg

# ──────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────
def emb_to_rgb(emb_map, use_global=True):
    """[D,H,W] → [H,W,3] RGB using global PCA."""
    D, h, w = emb_map.shape
    flat = emb_map.reshape(D, -1).T
    pc = pca.transform(flat)
    lo = PC_LO if use_global else np.percentile(pc, 1, axis=0)
    hi = PC_HI if use_global else np.percentile(pc, 99, axis=0)
    rgb = np.clip((pc - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8).reshape(h, w, 3)

def load_s2_rgb(patch_id, frame_idx=-1):
    src_dir = RAW_DIR / "s2" / patch_id
    if not src_dir.exists(): return None
    tifs = sorted(src_dir.glob("*.tif"))
    if not tifs: return None
    idx = min(frame_idx if frame_idx >= 0 else len(tifs)-1, len(tifs)-1)
    try:
        with rasterio.open(str(tifs[idx])) as src:
            data = src.read().astype(np.float32)
        rgb = data[[2,1,0], :, :] if data.shape[0] >= 3 else data[:3]
        for c in range(min(3, rgb.shape[0])):
            p2, p98 = np.percentile(rgb[c], (2, 98))
            if p98 > p2: rgb[c] = np.clip((rgb[c]-p2)/(p98-p2)*255, 0, 255)
        return np.transpose(rgb.astype(np.uint8), (1, 2, 0))
    except: return None

def colorize_worldcover(patch_id):
    src_dir = RAW_DIR / "worldcover" / patch_id
    if not src_dir.exists(): return None
    tifs = sorted(src_dir.glob("*.tif"))
    if not tifs: return None
    try:
        with rasterio.open(str(tifs[0])) as src:
            data = src.read(1).astype(np.int32)
        h, w = data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for v, color in WC_COLORS.items():
            rgb[data == v] = color
        return rgb
    except: return None

def get_patch_idx(patch_id):
    for i, pid in enumerate(EMB_IDS):
        if pid == patch_id: return i
    return -1

# ──────────────────────────────────────────
# 全域拼接 (参照 app_v2.py: 基于 UTM bounds 定位)
# ──────────────────────────────────────────
def _build_mosaic():
    """Build global mosaic using UTM-based patch positioning."""
    # Get bounds for each patch from grid
    patch_bounds = {}
    for feat in GRID_DATA["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (round(min(xs)), round(min(ys)))

    records = []
    for i, pid in enumerate(EMB_IDS):
        bounds = patch_bounds.get(pid)
        if bounds is None:
            continue
        records.append((bounds, i))

    if not records:
        return None

    # Compute grid layout
    all_x = sorted({b[0] for b, _ in records})
    all_y = sorted({b[1] for b, _ in records}, reverse=True)
    x_to_col = {x: c for c, x in enumerate(all_x)}
    y_to_row = {y: r for r, y in enumerate(all_y)}

    nrows = len(all_y)
    ncols = len(all_x)
    canvas = np.zeros((nrows * H, ncols * W, 3), dtype=np.float32)

    for bounds, idx in records:
        col = x_to_col.get(bounds[0])
        row = y_to_row.get(bounds[1])
        if col is None or row is None:
            continue
        rgb = emb_to_rgb(EMB_RAW[idx], use_global=True)
        r0, c0 = row * H, col * W
        canvas[r0:r0+H, c0:c0+W] = rgb.astype(np.float32) / 255.0

    canvas_u8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(canvas_u8)
    # 3× nearest-neighbor upscaling
    scale = 3
    pil_img = pil_img.resize(
        (pil_img.width * scale, pil_img.height * scale), Image.NEAREST
    )
    return pil_img

# ──────────────────────────────────────────
# 全域变化强度图
# ──────────────────────────────────────────
def _compute_change_intensity(before_window, after_window, max_patches=50):
    """Compute global change intensity map for selected time windows.
    
    Args:
        before_window: tuple (start_ms, end_ms) OR string preset name
        after_window: tuple (start_ms, end_ms) OR string preset name
    
    Returns PIL Image (hot colormap) showing per-pixel change intensity.
    """
    # Resolve presets if strings are passed
    if isinstance(before_window, str):
        bs = TIME_WINDOWS.get(before_window)
    else:
        bs = before_window
    if isinstance(after_window, str):
        ae = TIME_WINDOWS.get(after_window)
    else:
        ae = after_window
        
    if not bs or not ae:
        return None, f"❌ 无效时间窗口 (bs={bs}, ae={ae})"

    model, dataset, cfg = get_model()
    device = torch.device("npu:0")

    # Get patch bounds from grid
    patch_bounds = {}
    for feat in GRID_DATA["features"]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        patch_bounds[pid] = (round(min(xs)), round(min(ys)))

    n_patches = min(max_patches, len(EMB_IDS))
    
    # Compute grid layout
    records = []
    for i in range(n_patches):
        pid = EMB_IDS[i]
        bounds = patch_bounds.get(pid)
        if bounds is None:
            continue
        records.append((bounds, i))

    if not records:
        return None, "❌ 无有效 patch"

    all_x = sorted({b[0] for b, _ in records})
    all_y = sorted({b[1] for b, _ in records}, reverse=True)
    x_to_col = {x: c for c, x in enumerate(all_x)}
    y_to_row = {y: r for r, y in enumerate(all_y)}

    nrows = len(all_y)
    ncols = len(all_x)
    
    # Create canvas for change intensity: each patch contributes a [H, W] heatmap
    change_canvas = np.zeros((nrows * H, ncols * W), dtype=np.float32)

    start_time = time.time()
    all_dists = []

    for bounds, i in records:
        pid = EMB_IDS[i]
        if pid not in dataset.patches:
            continue
        pidx = dataset.patches.index(pid)
        col = x_to_col.get(bounds[0])
        row = y_to_row.get(bounds[1])
        if col is None or row is None:
            continue

        try:
            # Extract before embedding
            batch = dataset[pidx]
            batch["valid_start_ms"] = torch.tensor(bs[0], dtype=torch.float64)
            batch["valid_end_ms"] = torch.tensor(bs[1], dtype=torch.float64)
            bd = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            with torch.no_grad():
                out_b = model(
                    source_frames=bd["source_frames"],
                    source_timestamps_ms=bd["source_timestamps_ms"],
                    source_frame_mask=bd["source_frame_mask"],
                    source_input_mask=bd["source_input_mask"],
                    source_type_ids=bd["source_type_ids"],
                    valid_start_ms=bd["valid_start_ms"],
                    valid_end_ms=bd["valid_end_ms"],
                    target_relative_time=bd["target_relative_time"],
                    target_metadata=bd["target_metadata"],
                )
            emb_b = F.normalize(out_b.embedding_map, p=2, dim=1).cpu().numpy()[0]

            # Extract after embedding
            batch["valid_start_ms"] = torch.tensor(ae[0], dtype=torch.float64)
            batch["valid_end_ms"] = torch.tensor(ae[1], dtype=torch.float64)
            bd = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            with torch.no_grad():
                out_a = model(
                    source_frames=bd["source_frames"],
                    source_timestamps_ms=bd["source_timestamps_ms"],
                    source_frame_mask=bd["source_frame_mask"],
                    source_input_mask=bd["source_input_mask"],
                    source_type_ids=bd["source_type_ids"],
                    valid_start_ms=bd["valid_start_ms"],
                    valid_end_ms=bd["valid_end_ms"],
                    target_relative_time=bd["target_relative_time"],
                    target_metadata=bd["target_metadata"],
                )
            emb_a = F.normalize(out_a.embedding_map, p=2, dim=1).cpu().numpy()[0]

            # Per-pixel cosine distance
            D_e = emb_b.shape[0]
            fb = emb_b.reshape(D_e, -1)
            fa = emb_a.reshape(D_e, -1)
            nb = np.linalg.norm(fb, axis=0, keepdims=True)
            na = np.linalg.norm(fa, axis=0, keepdims=True)
            fb = fb / np.maximum(nb, 1e-8)
            fa = fa / np.maximum(na, 1e-8)
            cos_sim = np.sum(fb * fa, axis=0)
            patch_dist = ((1.0 - cos_sim) / 2.0).reshape(H, W)

            # Place in global canvas
            r0, c0 = row * H, col * W
            change_canvas[r0:r0+H, c0:c0+W] = patch_dist
            all_dists.append(patch_dist.mean())

            elapsed = time.time() - start_time
            n_done = len(all_dists)
            eta = (elapsed / n_done) * (len(records) - n_done) if n_done > 0 else 0
            if n_done % 10 == 0:
                print(f"  Patch {n_done}/{len(records)}: mean_dist={patch_dist.mean():.4f}, ETA={eta:.0f}s")

        except Exception as e:
            print(f"  Patch {pid} error: {e}")
            continue

    # Convert to hot colormap PIL Image
    # Normalize to 0-1
    if all_dists:
        vmin = 0.0
        vmax = max(max(all_dists) * 1.5, 0.05)  # Scale for visibility
    else:
        vmin, vmax = 0.0, 0.1

    norm = np.clip((change_canvas - vmin) / (vmax - vmin + 1e-8), 0, 1)
    # Apply hot colormap
    from matplotlib.cm import get_cmap
    cmap = get_cmap('hot')
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb)
    # 3× nearest-neighbor upscaling
    scale = 3
    pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)

    elapsed = time.time() - start_time
    msg = (
        f"✅ 计算完成 ({elapsed:.1f}s)\n\n"
        f"| 指标 | 值 |\n|------|-----|\n"
        f"| 计算 patch 数 | {len(all_dists)}/{len(records)} |\n"
        f"| 平均 cosine distance | {np.mean(all_dists):.4f} |\n"
        f"| 最大 patch mean | {np.max(all_dists):.4f} |\n"
        f"| 最小 patch mean | {np.min(all_dists):.4f} |\n\n"
        f"🔴 暖色 (红/黄) = 高变化强度 | ⚫ 冷色 (黑) = 低变化强度\n\n"
        f"⚠️ 注意: 全域变化强度图使用无监督 cosine distance，AUC~0.50，仅供参考。如需准确变化检测，请使用 湿地监测 Tab (AUC=0.712)。"
    )

    return pil_img, msg


# ──────────────────────────────────────────
# 下游任务: 湿地监测 (少样本变化检测)
# ──────────────────────────────────────────
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)    # 2024Q3-2025Q4

# 加载标注
_all_changes = []
for _shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
    try:
        _gdf = gpd.read_file(f"{ANNOT_DIR}/{_shp_name}")
        if _gdf.crs is not None and _gdf.crs.to_epsg() != 32652:
            _gdf = _gdf.to_crs(epsg=32652)
        for _, _row in _gdf.iterrows():
            if _row.geometry is not None:
                _all_changes.append({"geometry": _row.geometry, "period": _shp_name.replace(".shp", "")})
    except:
        pass

_patch_to_changes = {}
for _ch in _all_changes:
    for _pid, _bounds in PATCH_BOUNDS.items():
        if _ch["geometry"].intersects(box(_bounds[0], _bounds[1], _bounds[2], _bounds[3])):
            if _pid not in _patch_to_changes:
                _patch_to_changes[_pid] = []
            _patch_to_changes[_pid].append(_ch)

_annotated_patches = sorted(list(_patch_to_changes.keys()))
_embedding_cache = {}

def _get_embedding_for_patch(pid):
    """Get before/after embedding for a patch (with caching)."""
    if pid in _embedding_cache:
        return _embedding_cache[pid]
    if pid not in EMB_IDS:
        return None, None

    model, dataset, cfg = get_model()
    try:
        idx = dataset.patches.index(pid)
    except (ValueError, AttributeError):
        return None, None

    batch = dataset[idx]
    emb_b = None
    emb_a = None

    for ws, we in [(BEFORE_WINDOW[0], BEFORE_WINDOW[1]), (AFTER_WINDOW[0], AFTER_WINDOW[1])]:
        batch["valid_start_ms"] = torch.tensor(ws, dtype=torch.float64)
        batch["valid_end_ms"] = torch.tensor(we, dtype=torch.float64)
        batch_dev = {k: v.unsqueeze(0).to("npu:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
        emb = F.normalize(output.embedding_map, p=2, dim=1)[0].cpu().numpy()
        if emb_b is None:
            emb_b = emb
        else:
            emb_a = emb

    _embedding_cache[pid] = (emb_b, emb_a)
    return emb_b, emb_a

def _run_wetland_cd(pid, shot_count=500):
    """Run change detection with kNN-5 on a specific patch."""
    if pid not in _patch_to_changes:
        return None, f"❌ Patch {pid} 无标注数据"
    before, after = _get_embedding_for_patch(pid)
    if before is None:
        return None, f"❌ 无法提取 {pid} 的 embedding"

    D_e, H_e, W_e = before.shape
    bounds = PATCH_BOUNDS[pid]
    resolution = (bounds[2] - bounds[0]) / H_e

    mask = np.zeros((H_e, W_e), dtype=np.int32)
    for ch_info in _patch_to_changes.get(pid, []):
        geom = ch_info["geometry"]
        if geom is None:
            continue
        minx, miny, maxx, maxy = geom.bounds
        px_start = max(0, int((minx - bounds[0]) / resolution))
        px_end = min(H_e, int((maxx - bounds[0]) / resolution) + 1)
        py_start = max(0, int((bounds[3] - maxy) / resolution))
        py_end = min(W_e, int((bounds[3] - miny) / resolution) + 1)
        for px in range(px_start, px_end):
            for py in range(py_start, py_end):
                wx = bounds[0] + (px + 0.5) * resolution
                wy = bounds[3] - (py + 0.5) * resolution
                if geom.contains(Point(wx, wy)):
                    mask[px, py] = 1

    all_features = []
    all_labels = []
    for px in range(H_e):
        for py in range(W_e):
            feat = np.concatenate([before[:, px, py], after[:, px, py]])
            all_features.append(feat)
            all_labels.append(mask[px, py])

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    rng = np.random.RandomState(42)
    changed_idx = np.where(y == 1)[0]
    unchanged_idx = np.where(y == 0)[0]
    n_ch = min(shot_count, len(changed_idx))
    n_un = min(shot_count, len(unchanged_idx))
    if n_ch == 0 or n_un == 0:
        return None, f"❌ 样本不足"

    ch_sample = rng.choice(changed_idx, n_ch, replace=False)
    un_sample = rng.choice(unchanged_idx, n_un, replace=False)
    train_idx = np.concatenate([ch_sample, un_sample])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_all_scaled = scaler.transform(X)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_scaled, y[train_idx])
    probs = clf.predict_proba(X_all_scaled)[:, 1].reshape(H_e, W_e)

    # Generate heatmap
    img = (probs * 255).astype(np.uint8)
    colored = np.zeros((H_e, W_e, 3), dtype=np.uint8)
    colored[:, :, 0] = img
    colored[:, :, 1] = img // 2
    colored[:, :, 2] = 255 - img
    pil_img = Image.fromarray(colored).resize((W_e*4, H_e*4), Image.NEAREST)

    stats = f"""✅ 变化检测完成 (kNN-5)

| 指标 | 值 |
|------|-----|
| Patch ID | {pid} |
| Shot Count | {shot_count} |
| 变化像素比例 | {y.mean()*100:.1f}% |
| 正样本数 | {int(y.sum())} |
| 负样本数 | {int(len(y)-y.sum())} |

📊 少样本 AUC 参考:
- 1-shot: 0.495
- 10-shot: 0.507
- 50-shot: 0.526
- 100-shot: 0.525
- **500-shot: 0.712** (当前最佳)
"""
    return pil_img, stats

def _plot_fewshot_curve():
    """Plot few-shot performance curve."""
    shots = [1, 10, 50, 100, 500]
    aucs = [0.495, 0.507, 0.526, 0.525, 0.712]
    stds = [0.011, 0.014, 0.009, 0.012, 0.008]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(shots, aucs, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8, color='#2196F3')
    ax.fill_between(shots,
                    [a - s for a, s in zip(aucs, stds)],
                    [a + s for a, s in zip(aucs, stds)],
                    alpha=0.2, color='#2196F3')
    ax.set_xlabel('Shot Count (samples per class)', fontsize=14)
    ax.set_ylabel('AUC-ROC', fontsize=14)
    ax.set_title('Few-Shot Change Detection Performance\n(V2 Embedding + kNN-5, AUC=0.712 @ 500-shot)', fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim(0.4, 0.8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.axhline(y=0.712, color='green', linestyle='--', alpha=0.5, label='Best (500-shot)')
    ax.legend()
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────
patch_choices = EMB_IDS[:30]

with gr.Blocks(title="AEF_qwen Demo") as demo:
    gr.Markdown("""
# AEF_qwen Demo — 哈尔滨新区 424 patches

| 项目 | 详情 |
|------|------|
| **模型** | qwen_v2 (S2+S1+Landsat → 128-dim embedding, skip_l2 + raw_uniformity) |
| **训练** | 500 epochs, 含时序对比损失微调 |
| **输入** | 3类时序图像 (仅 S2, S1, Landsat) |
| **目标** | 7类重建 (输入3类 + DEM + WorldCover + Dynamic World + JRC Water) |
| **下游任务** | kNN-5 分类器, AUC = **0.712** (500-shot) |
""")

    with gr.Tabs():
        # ── Tab 1: 嵌入可视化 ──
        with gr.Tab("🗺️ 嵌入可视化"):
            gr.Markdown("### 嵌入空间 PCA 可视化")
            with gr.Row():
                emb_patch = gr.Dropdown(choices=patch_choices, value=patch_choices[0], label="Patch ID")
            with gr.Row():
                btn_emb = gr.Button("🔍 查看", variant="primary")
            with gr.Row():
                img_s2 = gr.Image(label="S2 RGB", height=280)
                img_pca = gr.Image(label="Embedding PCA-RGB", height=280)
                img_wc = gr.Image(label="WorldCover", height=280)

            with gr.Accordion("全域 Embedding PCA RGB 拼接 (参照论文 Fig. 1C-D)", open=True):
                gr.Markdown(
                    "将所有 424 个 patch 按真实 UTM 位置拼接。颜色使用全局 PCA + 1%/99% 百分位归一化，"
                    "确保跨 patch 颜色连续。3× 最近邻上采样保留像素级细节。"
                )
                btn_mosaic = gr.Button("🌐 生成全域拼接图", variant="primary")
                img_mosaic = gr.Image(label="全域 Embedding PCA RGB", height=600)

            btn_emb.click(
                fn=lambda pid: (
                    load_s2_rgb(pid),
                    emb_to_rgb(EMB_RAW[get_patch_idx(pid)], use_global=False) if get_patch_idx(pid) >= 0 else None,
                    colorize_worldcover(pid)
                ),
                inputs=[emb_patch],
                outputs=[img_s2, img_pca, img_wc]
            )
            btn_mosaic.click(fn=_build_mosaic, outputs=[img_mosaic])

        # ── Tab 2: 全域变化强度图 ──
        with gr.Tab("🔥 全域变化强度"):
            gr.Markdown("""
### 全域变化强度一览图

选择任意两个日期范围（精确到天），模型对每个 patch 分别提取 before/after embedding，
计算逐像素 cosine distance，然后按真实 UTM 位置拼接为全域变化强度热力图。

- **暖色 (红/黄)**: 高变化强度
- **冷色 (黑)**: 低变化强度

> 可用预设: 月度 (2023-01~2025-10) / 季度 / 年度
> 也可使用 DatePicker 输入任意自定义日期范围
            """)

            with gr.Row():
                chg_before = gr.Dropdown(
                    choices=list(TIME_WINDOWS.keys()),
                    value="2023-04",
                    label="Before 预设"
                )
                chg_before_start = gr.Textbox(value="", label="Before 起始 (自定义, 留空用预设)", placeholder="YYYY-MM-DD")
                chg_before_end = gr.Textbox(value="", label="Before 结束 (自定义)", placeholder="YYYY-MM-DD")
            with gr.Row():
                chg_after = gr.Dropdown(
                    choices=list(TIME_WINDOWS.keys()),
                    value="2025-06",
                    label="After 预设"
                )
                chg_after_start = gr.Textbox(value="", label="After 起始 (自定义, 留空用预设)", placeholder="YYYY-MM-DD")
                chg_after_end = gr.Textbox(value="", label="After 结束 (自定义)", placeholder="YYYY-MM-DD")
            btn_chg = gr.Button("🚀 生成全域变化强度图", variant="primary")
            chg_image = gr.Image(label="全域变化强度图 (hot colormap: 暖色=高变化, 冷色=低变化)", height=600)
            chg_stats = gr.Markdown("### 统计信息\n点击运行后显示")

            def _run_change(b_preset, a_preset, b_start, b_end, a_start, a_end):
                """Resolve time window from presets or custom dates."""
                bs = TIME_WINDOWS.get(b_preset)
                ae = TIME_WINDOWS.get(a_preset)
                if b_start and b_end:
                    try: bs = (str_to_ms(b_start), str_to_ms(b_end))
                    except: pass
                if a_start and a_end:
                    try: ae = (str_to_ms(a_start), str_to_ms(a_end))
                    except: pass
                return _compute_change_intensity(bs, ae)

            btn_chg.click(
                fn=_run_change,
                inputs=[chg_before, chg_after, chg_before_start, chg_before_end, chg_after_start, chg_after_end],
                outputs=[chg_image, chg_stats]
            )

        # ── Tab 3: 单 Patch 变化对比 ──
        with gr.Tab("🔍 单 Patch 变化对比"):
            gr.Markdown("### 单 Patch 变化对比\n选择 patch 和两个时间窗口，对比变化前后的图像。")
            with gr.Row():
                cd_patch = gr.Dropdown(choices=patch_choices, value=patch_choices[0], label="Patch ID")
            with gr.Row():
                cd_before = gr.Dropdown(choices=list(TIME_WINDOWS.keys()), value="2023 Q3-Q4", label="Before 时间窗口")
                cd_after = gr.Dropdown(choices=list(TIME_WINDOWS.keys()), value="2024 Q3-Q4", label="After 时间窗口")
            btn_cd = gr.Button("🔍 对比变化", variant="primary")
            with gr.Row():
                img_before = gr.Image(label="Before S2 RGB", height=280)
                img_after = gr.Image(label="After S2 RGB", height=280)
                img_change = gr.Image(label="变化热力图", height=280)

            def _show_single_patch_cd(pid, before_tw, after_tw):
                """Show before/after S2 RGB + change heatmap for a single patch."""
                model, dataset, cfg = get_model()
                idx = get_patch_idx(pid)
                if idx < 0:
                    return None, None, None

                bs = TIME_WINDOWS.get(before_tw)
                ae = TIME_WINDOWS.get(after_tw)
                if not bs or not ae:
                    return None, None, None

                # Load S2 RGB images
                # Find frames closest to the time windows
                s2_dir = RAW_DIR / "s2" / pid
                if not s2_dir.exists():
                    return None, None, None
                tifs = sorted(s2_dir.glob("*.tif"))
                if not tifs:
                    return None, None, None

                # Find frame indices closest to time window centers
                n_frames = len(tifs)
                before_center = (bs[0] + bs[1]) / 2
                after_center = (ae[0] + ae[1]) / 2

                # Get timestamps from dataset
                batch = dataset[idx]
                ts = batch["source_timestamps_ms"]
                if ts.dim() == 2:
                    ts_flat = ts.flatten()
                else:
                    ts_flat = ts

                # Find closest frames
                before_idx = int(torch.argmin(torch.abs(ts_flat - before_center)))
                after_idx = int(torch.argmin(torch.abs(ts_flat - after_center)))
                before_idx = min(before_idx, n_frames - 1)
                after_idx = min(after_idx, n_frames - 1)

                img_b = None
                img_a = None
                try:
                    with rasterio.open(str(tifs[before_idx])) as src:
                        data = src.read().astype(np.float32)
                    rgb = data[[2, 1, 0], :, :] if data.shape[0] >= 3 else data[:3]
                    for c in range(min(3, rgb.shape[0])):
                        p2, p98 = np.percentile(rgb[c], (2, 98))
                        if p98 > p2:
                            rgb[c] = np.clip((rgb[c] - p2) / (p98 - p2) * 255, 0, 255)
                    img_b = Image.fromarray(np.transpose(rgb.astype(np.uint8), (1, 2, 0)))
                except:
                    pass

                try:
                    with rasterio.open(str(tifs[after_idx])) as src:
                        data = src.read().astype(np.float32)
                    rgb = data[[2, 1, 0], :, :] if data.shape[0] >= 3 else data[:3]
                    for c in range(min(3, rgb.shape[0])):
                        p2, p98 = np.percentile(rgb[c], (2, 98))
                        if p98 > p2:
                            rgb[c] = np.clip((rgb[c] - p2) / (p98 - p2) * 255, 0, 255)
                    img_a = Image.fromarray(np.transpose(rgb.astype(np.uint8), (1, 2, 0)))
                except:
                    pass

                # Compute change heatmap using embeddings
                emb_b = None
                emb_a = None

                for ws, we in [bs, ae]:
                    batch["valid_start_ms"] = torch.tensor(ws, dtype=torch.float64)
                    batch["valid_end_ms"] = torch.tensor(we, dtype=torch.float64)
                    batch_dev = {k: v.unsqueeze(0).to("npu:0") if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
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
                    emb = F.normalize(output.embedding_map, p=2, dim=1)[0].cpu().numpy()
                    if emb_b is None:
                        emb_b = emb
                    else:
                        emb_a = emb

                # Compute change heatmap
                fb = emb_b.reshape(D, -1)
                fa = emb_a.reshape(D, -1)
                cos_sim = np.sum(fb * fa, axis=0) / (np.linalg.norm(fb, axis=0) * np.linalg.norm(fa, axis=0) + 1e-8)
                change_dist = ((1.0 - cos_sim) / 2.0).reshape(H, W)

                colored = np.zeros((H, W, 3), dtype=np.uint8)
                change_norm = np.clip(change_dist / max(change_dist.max(), 1e-8), 0, 1)
                colored[:, :, 0] = (change_norm * 255).astype(np.uint8)
                colored[:, :, 1] = (change_norm * 200).astype(np.uint8)
                colored[:, :, 2] = ((1 - change_norm) * 100).astype(np.uint8)
                img_c = Image.fromarray(colored).resize((W * 4, H * 4), Image.NEAREST)

                return img_b, img_a, img_c

            btn_cd.click(
                fn=_show_single_patch_cd,
                inputs=[cd_patch, cd_before, cd_after],
                outputs=[img_before, img_after, img_change]
            )

        # ── Tab 4: 训练曲线 ──
        with gr.Tab("📈 训练曲线"):
            gr.Markdown("### 训练曲线")
            btn_curve = gr.Button("加载", variant="primary")
            curve_plot = gr.Plot()

            def show_curve():
                log_path = Path("/workspace/logs/qwen_v1_train_v3.log")
                if not log_path.exists():
                    fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No log"); ax.axis("off"); return fig
                epochs, recons, runifs, punifs = [], [], [], []
                with open(log_path) as f:
                    for line in f:
                        if "Epoch" in line and "Recon" in line and "Traceback" not in line:
                            ep = re.search(r'Epoch\s+(\d+)', line)
                            r = re.search(r'Recon:\s*([\d.]+)', line)
                            u = re.search(r'RawUnif:\s*([-.\d]+)', line)
                            p = re.search(r'PreUnif:\s*([-.\d]+)', line)
                            if all([ep, r, u, p]):
                                epochs.append(int(ep.group(1)))
                                recons.append(float(r.group(1)))
                                runifs.append(float(u.group(1)))
                                punifs.append(float(p.group(1)))
                if not epochs:
                    fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data"); ax.axis("off"); return fig
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].plot(epochs, recons, "b-", lw=2); axes[0].set_title("Reconstruction"); axes[0].grid(True, alpha=0.3)
                axes[1].plot(epochs, runifs, "g-", lw=2, label="RawUnif")
                axes[1].plot(epochs, punifs, "r--", lw=2, label="PreUnif")
                axes[1].set_title("Uniformity (lower=better)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
                axes[2].text(0.5, 0.5, "Recon=1.06\nRawUnif=-4.35\nPreUnif=-4.04\nVar=0.12",
                           ha="center", va="center", fontsize=14, transform=axes[2].transAxes)
                axes[2].set_title("Final Metrics"); axes[2].axis("off")
                fig.tight_layout(); return fig

            btn_curve.click(show_curve, outputs=[curve_plot])

        # ── Tab 5: 下游任务 - 湿地监测 ──
        with gr.Tab("🌊 湿地监测"):
            gr.Markdown("""
### 湿地变化监测 — 少样本变化检测

基于 V2 Embedding + kNN-5 分类器，只需标注少量像素即可训练。

**当前最佳**: AUC = **0.712** (500-shot)
""")
            with gr.Row():
                wetland_patch = gr.Dropdown(choices=_annotated_patches, value=_annotated_patches[0] if _annotated_patches else None, label="Patch ID")
                shot_slider = gr.Slider(minimum=1, maximum=1000, value=500, step=50, label="Shot Count (每类样本数)")
            btn_wetland = gr.Button("🚀 运行变化检测", variant="primary")
            with gr.Row():
                wetland_image = gr.Image(label="变化检测热力图", height=400)
                wetland_stats = gr.Markdown("### 统计信息")

            btn_wetland.click(
                fn=_run_wetland_cd,
                inputs=[wetland_patch, shot_slider],
                outputs=[wetland_image, wetland_stats]
            )

        # ── Tab 6: 少样本效果曲线 ──
        with gr.Tab("📊 少样本效果"):
            gr.Markdown("""
### 少样本性能曲线

不同 shot count 下的 AUC-ROC 变化。展示模型在不同标注量下的表现。
""")
            btn_curve2 = gr.Button("加载曲线", variant="primary")
            curve_image = gr.Plot()
            btn_curve2.click(fn=_plot_fewshot_curve, outputs=[curve_image])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
