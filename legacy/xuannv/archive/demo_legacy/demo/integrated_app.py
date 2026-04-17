#!/usr/bin/env python3
"""
AEF_qwen 整合版 Demo — 所有功能在一个页面

包含:
1. 🗺️ 嵌入可视化 (单 patch + 全域拼接)
2. 🔥 全域变化强度图 (before/after 对比)
3. 🔍 空间异常检测
4. 📈 训练曲线
5. 🌊 下游任务: 湿地监测 (少样本变化检测)
6. 📊 少样本效果曲线

基于: V2 checkpoint (epoch_499.pt), AUC 0.712 (500-shot)
"""
import os, sys, json, time, re, argparse
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.insert(0, "/workspace/xuannv")

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
CONFIG_PATH = "/workspace/xuannv/configs/qwen_v1_scenes.yaml"
CKPT_PATH = "/workspace/outputs/aef_qwen_v2/epoch_499.pt"
GRID_PATH = "/workspace/index/harbin/grid/harbin_grid.geojson"
ANNOT_DIR = "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件"
EMB_L2_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/embedding_maps_normalized.npy"
EMB_RAW_PATH = "/workspace/outputs/aef_qwen_v1/embeddings/embedding_maps_raw.npy"
PCA_RGB_PATH = None
GLOBAL_PCA_PATH = None
BEFORE_WINDOW = (1688169600000.0, 1703980800000.0)  # 2023Q3-Q4
AFTER_WINDOW = (1719792000000.0, 1735603200000.0)    # 2024Q3-2025Q4
H, W, D = 64, 64, 128

# ──────────────────────────────────────────
# 加载模型
# ──────────────────────────────────────────
from src.config import load_config
from src.models.model import AEFModel
from src.data.dataset import HarbinPatchDataset

cfg = load_config(CONFIG_PATH)
model = AEFModel(cfg).to("cuda:0")
ckpt = torch.load(CKPT_PATH, map_location="cuda:0", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dataset = HarbinPatchDataset(cfg)
dataset.training = False
dataset._spatial_augmentation = False

# ──────────────────────────────────────────
# 加载嵌入
# ──────────────────────────────────────────
EMB_L2 = np.load(EMB_L2_PATH)
EMB_RAW = np.load(EMB_RAW_PATH)
# PCA_RGB = np.load(PCA_RGB_PATH)
with open(GRID_PATH) as f:
    GRID_DATA = json.load(f)
EMB_IDS = sorted([f["properties"]["patch_id"] for f in GRID_DATA["features"]])

# Global PCA
# with open(GLOBAL_PCA_PATH, "rb") as f:
    # _gdata = json.load(f)
    # GLOBAL_PCA = np.array(_gdata["pca_components"])
    # GLOBAL_MEAN = np.array(_gdata["global_mean"])
    # GLOBAL_MIN = np.array(_gdata["global_min"])
    # GLOBAL_MAX = np.array(_gdata["global_max"])

# ──────────────────────────────────────────
# 时间窗口
# ──────────────────────────────────────────
TIME_WINDOWS = {}
MONTHS = ["01","02","03","04","05","06","07","08","09","10","11","12"]
for y in [2023,2024,2025]:
    for m in MONTHS[:10]:
        ts = int(time.mktime(time.strptime(f"{y}-{m}-01", "%Y-%m-%d")))*1000
        next_m = int(m)+1 if int(m)<12 else 1; next_y = y if int(m)<12 else y+1
        ts_end = int(time.mktime(time.strptime(f"{next_y}-{next_m:02d}-01", "%Y-%m-%d")))*1000
        TIME_WINDOWS[f"{y}-{m}"] = (float(ts), float(ts_end))

# 季度/年度预设
for y in [2023,2024,2025]:
    q1 = int(time.mktime(time.strptime(f"{y}-01-01", "%Y-%m-%d")))*1000
    q2 = int(time.mktime(time.strptime(f"{y}-04-01", "%Y-%m-%d")))*1000
    q3 = int(time.mktime(time.strptime(f"{y}-07-01", "%Y-%m-%d")))*1000
    q4 = int(time.mktime(time.strptime(f"{y}-10-01", "%Y-%m-%d")))*1000
    ny = int(time.mktime(time.strptime(f"{y+1}-01-01", "%Y-%m-%d")))*1000
    TIME_WINDOWS[f"{y} Q1-Q2"] = (float(q1), float(q3))
    TIME_WINDOWS[f"{y} Q3-Q4"] = (float(q3), float(ny))
    TIME_WINDOWS[f"{y} 全年"] = (float(q1), float(ny))

def str_to_ms(s):
    return int(time.mktime(time.strptime(s.strip(), "%Y-%m-%d"))) * 1000

def get_patch_idx(pid):
    try: return EMB_IDS.index(pid)
    except: return -1

def load_s2_rgb(pid):
    """Load S2 RGB image."""
    import rasterio
    s2_dir = Path("/workspace/raw/harbin_scenes/sentinel2_10m") / pid
    if not s2_dir.exists():
        return None
    files = sorted(s2_dir.glob("*.tif"))
    if not files:
        return None
    # Use median frame
    frames = []
    for f in files[:5]:  # Use first 5 frames for speed
        try:
            with rasterio.open(f) as src:
                data = src.read([4,3,2])  # RGB bands
                frames.append(data)
        except:
            pass
    if not frames:
        return None
    img = np.median(frames, axis=0)
    img = np.clip(img / 3000.0 * 255, 0, 255).astype(np.uint8)
    img = img.transpose(1,2,0)
    return Image.fromarray(img)

def emb_to_rgb(emb, use_global=True):
    """Convert embedding to RGB using PCA."""
    D_e = emb.shape[0]
    flat = emb.reshape(D_e, -1).T  # [N, D]
    if use_global:
        pca = GLOBAL_PCA[:3, :D_e]  # [3, D]
        rgb = (flat - GLOBAL_MEAN[:D_e]) @ pca.T  # [N, 3]
        vmin = GLOBAL_MIN
        vmax = GLOBAL_MAX
    else:
        from sklearn.decomposition import PCA
        pca_local = PCA(n_components=3)
        rgb = pca_local.fit_transform(flat)
        vmin = np.percentile(rgb, 1, axis=0)
        vmax = np.percentile(rgb, 99, axis=0)
    rgb = (rgb - vmin) / (vmax - vmin + 1e-8)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb.reshape(H, W, 3))

def colorize_worldcover(pid):
    """Colorize WorldCover."""
    wc_path = Path("/workspace/raw/harbin_scenes/worldcover") / pid / "2020.tif"
    if not wc_path.exists():
        return None
    import rasterio
    with rasterio.open(wc_path) as src:
        wc = src.read(1)
    colors = {
        10: (0,100,0), 20: (0,160,0), 30: (255,187,34), 40: (255,255,76),
        50: (195,23,5), 60: (179,179,179), 70: (250,250,250), 80: (0,0,160),
        90: (0,0,0), 95: (0,51,51), 100: (200,200,200)
    }
    img = np.zeros((wc.shape[0], wc.shape[1], 3), dtype=np.uint8)
    for code, color in colors.items():
        img[wc == code] = color
    return Image.fromarray(img)

# ──────────────────────────────────────────
# Tab 1: 嵌入可视化
# ──────────────────────────────────────────
def _build_mosaic():
    """Build global mosaic from all patches using global PCA."""
    # Collect all embeddings
    all_embeddings = EMB_L2.reshape(len(EMB_IDS), D, -1).transpose(0, 2, 1)
    all_flat = all_embeddings.reshape(-1, D)
    
    # Global PCA
    pca_global = PCA(n_components=3)
    rgb_global = pca_global.fit_transform(all_flat)
    vmin = np.percentile(rgb_global, 1, axis=0)
    vmax = np.percentile(rgb_global, 99, axis=0)
    rgb_norm = (rgb_global - vmin) / (vmax - vmin + 1e-8)
    rgb_norm = np.clip(rgb_norm * 255, 0, 255).astype(np.uint8)
    
    bounds_data = []
    for feat in GRID_DATA["features"]:
        pid = feat["properties"]["patch_id"]
        idx = EMB_IDS.index(pid)
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bounds_data.append((pid, min(xs), min(ys), idx))

    min_x = min(b[1] for b in bounds_data)
    min_y = min(b[2] for b in bounds_data)
    global_w = int((max(b[1]+W for b in bounds_data) - min_x))
    global_h = int((max(b[2]+H for b in bounds_data) - min_y))
    mosaic = np.zeros((global_h, global_w, 3), dtype=np.uint8)

    for pid, bx, by, idx in bounds_data:
        px = int(bx - min_x)
        py = int(by - min_y)
        rgb_patch = rgb_norm[idx*H*W:(idx+1)*H*W].reshape(H, W, 3)
        mosaic[py:py+H, px:px+W] = rgb_patch

    # 3x upscale
    mosaic = Image.fromarray(mosaic).resize((global_w*3, global_h*3), Image.NEAREST)
    return mosaic

# ──────────────────────────────────────────
# Tab 2: 全域变化强度图
# ──────────────────────────────────────────
def _compute_change_intensity(before_window, after_window, max_patches=50):
    """Compute global change intensity map."""
    records = []
    for feat in GRID_DATA["features"][:max_patches]:
        pid = feat["properties"]["patch_id"]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        records.append((pid, min(xs), min(ys)))

    records.sort(key=lambda r: (int(r[1]/1000), int(r[2]/1000)))
    cols = int(np.sqrt(len(records) * 1.5))
    rows = (len(records) + cols - 1) // cols

    min_x = min(r[1] for r in records)
    min_y = min(r[2] for r in records)

    change_canvas = np.zeros((rows * H, cols * W), dtype=np.float32)
    all_dists = []

    for i, (pid, bx, by) in enumerate(records):
        if i >= max_patches:
            break
        try:
            idx = dataset.patches.index(pid) if pid in dataset.patches else -1
            if idx < 0:
                continue

            batch = dataset[idx]
            emb_b = None
            emb_a = None

            for ws, we in [(before_window[0], before_window[1]), (after_window[0], after_window[1])]:
                batch["valid_start_ms"] = torch.tensor(ws, dtype=torch.float64)
                batch["valid_end_ms"] = torch.tensor(we, dtype=torch.float64)
                batch_dev = {k: v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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

            if emb_b is None or emb_a is None:
                continue

            D_e = emb_b.shape[0]
            fb = emb_b.reshape(D_e, -1)
            fa = emb_a.reshape(D_e, -1)
            nb = np.linalg.norm(fb, axis=0, keepdims=True)
            na = np.linalg.norm(fa, axis=0, keepdims=True)
            fb = fb / np.maximum(nb, 1e-8)
            fa = fa / np.maximum(na, 1e-8)
            cos_sim = np.sum(fb * fa, axis=0)
            patch_dist = ((1.0 - cos_sim) / 2.0).reshape(H, W)

            row, col = divmod(i, cols)
            r0, c0 = row * H, col * W
            change_canvas[r0:r0+H, c0:c0+W] = patch_dist
            all_dists.append(patch_dist.mean())

        except Exception as e:
            print(f"  Patch {pid} error: {e}")
            continue

    if not all_dists:
        return None, "❌ 无效时间窗口"

    img = np.clip(change_canvas / max(np.max(change_canvas), 1e-8), 0, 1)
    colored = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    colored[:, :, 0] = (img * 255).astype(np.uint8)
    colored[:, :, 1] = (img * 200).astype(np.uint8)
    colored[:, :, 2] = (255 - img * 255).astype(np.uint8)

    scale = 3
    pil_img = Image.fromarray(colored).resize((img.shape[1]*scale, img.shape[0]*scale), Image.NEAREST)

    msg = (
        f"✅ 计算完成\n\n"
        f"| 指标 | 值 |\n"
        f"|------|-----|\n"
        f"| 计算 patch 数 | {len(all_dists)}/{max_patches} |\n"
        f"| 平均 cosine distance | {np.mean(all_dists):.4f} |\n"
        f"| 最大 patch mean | {np.max(all_dists):.4f} |\n"
        f"| 最小 patch mean | {np.min(all_dists):.4f} |\n\n"
        f"🔴 暖色 (红/黄) = 高变化强度 | ⚫ 冷色 (黑) = 低变化强度\n\n"
        f"⚠️ 注意: 当前模型 AUC≈0.50，变化强度仅供参考，可能与真实变化不符。"
    )

    return pil_img, msg

# ──────────────────────────────────────────
# Tab 3: 空间异常检测
# ──────────────────────────────────────────
def show_anomaly(pid):
    """Show spatial anomaly detection."""
    idx = get_patch_idx(pid)
    if idx < 0:
        return None, None, None

    emb_l2 = EMB_L2[idx]
    mean_emb = emb_l2.mean(axis=(1, 2))
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    flat = emb_l2.reshape(D, -1)
    cos_sim = np.sum(flat * mean_emb[:, None], axis=0)
    anomaly = ((1.0 - cos_sim) / 2.0).reshape(H, W)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(anomaly, cmap="hot", vmin=0, vmax=0.5)
    ax.set_title(f"Spatial Anomaly", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")
    fig.tight_layout()

    return load_s2_rgb(pid), fig, colorize_worldcover(pid)

# ──────────────────────────────────────────
# Tab 4: 训练曲线
# ──────────────────────────────────────────
def show_curve():
    """Show training curves."""
    log_path = Path("/workspace/logs/qwen_v1_train_v3.log")
    if not log_path.exists():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No log"); ax.axis("off")
        return fig

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
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data"); ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, recons, "b-", lw=2)
    axes[0].set_title("Reconstruction")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, runifs, "g-", lw=2, label="RawUnif")
    axes[1].plot(epochs, punifs, "r--", lw=2, label="PreUnif")
    axes[1].set_title("Uniformity (lower=better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].text(0.5, 0.5,
                f"Recon={recons[-1]:.2f}\nRawUnif={runifs[-1]:.2f}\nPreUnif={punifs[-1]:.2f}",
                ha="center", va="center", fontsize=14, transform=axes[2].transAxes)
    axes[2].set_title("Final Metrics")
    axes[2].axis("off")
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────
# Tab 5: 下游任务 - 湿地监测
# ──────────────────────────────────────────
# 加载标注
all_changes = []
for shp_name in ["june.shp", "aug.shp", "September.shp", "October.shp"]:
    try:
        gdf = gpd.read_file(f"{ANNOT_DIR}/{shp_name}")
        if gdf.crs is not None and gdf.crs.to_epsg() != 32652:
            gdf = gdf.to_crs(epsg=32652)
        for _, row in gdf.iterrows():
            if row.geometry is not None:
                all_changes.append({"geometry": row.geometry, "period": shp_name.replace(".shp", "")})
    except:
        pass

patch_bounds = {}
for feat in GRID_DATA["features"]:
    pid = feat["properties"]["patch_id"]
    coords = feat["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    patch_bounds[pid] = (min(xs), min(ys), max(xs), max(ys))

patch_to_changes = {}
for ch in all_changes:
    for pid, bounds in patch_bounds.items():
        if ch["geometry"].intersects(box(bounds[0], bounds[1], bounds[2], bounds[3])):
            if pid not in patch_to_changes:
                patch_to_changes[pid] = []
            patch_to_changes[pid].append(ch)

# 缓存 embedding
embedding_cache = {}

def get_embedding_for_patch(pid):
    """Get before/after embedding for a patch."""
    if pid in embedding_cache:
        return embedding_cache[pid]

    if pid not in dataset.patches:
        return None, None

    pidx = dataset.patches.index(pid)
    batch = dataset[pidx]
    before_map = None
    after_map = None

    for ws, we in [(BEFORE_WINDOW[0], BEFORE_WINDOW[1]), (AFTER_WINDOW[0], AFTER_WINDOW[1])]:
        batch["valid_start_ms"] = torch.tensor(ws, dtype=torch.float64)
        batch["valid_end_ms"] = torch.tensor(we, dtype=torch.float64)
        batch_dev = {k: v.unsqueeze(0).to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
        if before_map is None:
            before_map = emb
        else:
            after_map = emb

    embedding_cache[pid] = (before_map, after_map)
    return before_map, after_map

def run_change_detection_with_knn(pid, shot_count=500):
    """Run change detection with kNN-5."""
    if pid not in patch_to_changes or pid not in dataset.patches:
        return None, f"❌ Patch {pid} 无标注数据或不存在"

    before, after = get_embedding_for_patch(pid)
    if before is None:
        return None, f"❌ 无法提取 {pid} 的 embedding"

    D_e, H_e, W_e = before.shape
    bounds = patch_bounds[pid]
    resolution = (bounds[2] - bounds[0]) / H_e

    mask = np.zeros((H_e, W_e), dtype=np.int32)
    for ch_info in patch_to_changes.get(pid, []):
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

    ch_sample = rng.choice(changed_idx, n_ch, replace=False)
    un_sample = rng.choice(unchanged_idx, n_un, replace=False)
    train_idx = np.concatenate([ch_sample, un_sample])

    X_train = X[train_idx]
    y_train = y[train_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_scaled, y_train)
    probs = clf.predict_proba(X_all_scaled)[:, 1].reshape(H_e, W_e)

    # 生成热力图
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
- **500-shot: 0.712**
"""

    return pil_img, stats

# ──────────────────────────────────────────
# Tab 6: 少样本效果曲线
# ──────────────────────────────────────────
def plot_fewshot_curve():
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
# Gradio UI - 整合版
# ──────────────────────────────────────────
patch_choices = EMB_IDS[:30]
wetland_patches = sorted(list(patch_to_changes.keys()))

with gr.Blocks(title="AEF_qwen 整合版 Demo") as demo:
    gr.Markdown("""
# 🌍 AEF_qwen 整合版 Demo — 哈尔滨新区 424 patches

| 项目 | 详情 |
|------|------|
| **模型** | V2 (S2+S1+Landsat → 128-dim embedding, skip L2 + raw_uniformity) |
| **训练** | 500 epochs, 时序对比损失 (已修复) |
| **输入** | 3类时序图像 (仅 S2, S1, Landsat) |
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

            with gr.Accordion("全域 Embedding PCA RGB 拼接", open=True):
                gr.Markdown("将所有 424 个 patch 按真实 UTM 位置拼接。3× 最近邻上采样。")
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
### 全域变化强度图
选择两个时间窗口，生成全域变化强度热力图。
- **暖色 (红/黄)**: 高变化强度
- **冷色 (黑)**: 低变化强度
""")
            with gr.Row():
                chg_before = gr.Dropdown(choices=list(TIME_WINDOWS.keys()), value="2023-04", label="Before 预设")
                chg_after = gr.Dropdown(choices=list(TIME_WINDOWS.keys()), value="2025-06", label="After 预设")
            btn_chg = gr.Button("🚀 生成全域变化强度图", variant="primary")
            chg_image = gr.Image(label="全域变化强度图", height=600)
            chg_stats = gr.Markdown("### 统计信息")

            def _run_change_simple(b_preset, a_preset):
                bs = TIME_WINDOWS.get(b_preset)
                ae = TIME_WINDOWS.get(a_preset)
                if not bs or not ae:
                    return None, "❌ 无效预设"
                return _compute_change_intensity(bs, ae)

            btn_chg.click(fn=_run_change_simple, inputs=[chg_before, chg_after], outputs=[chg_image, chg_stats])

        # ── Tab 3: 空间异常检测 ──
        with gr.Tab("🔍 空间异常检测"):
            gr.Markdown("### 空间异常检测\n每个像素与 patch 内均值的 cosine distance。")
            with gr.Row():
                anom_patch = gr.Dropdown(choices=patch_choices, value=patch_choices[0], label="Patch ID")
            btn_anom = gr.Button("🔍 检测", variant="primary")
            with gr.Row():
                img_s2_anom = gr.Image(label="S2 RGB", height=280)
                img_anom = gr.Plot(label="异常热力图")
                img_wc_anom = gr.Image(label="WorldCover", height=280)

            btn_anom.click(show_anomaly, [anom_patch], [img_s2_anom, img_anom, img_wc_anom])

        # ── Tab 4: 训练曲线 ──
        with gr.Tab("📈 训练曲线"):
            gr.Markdown("### V1 训练曲线 (400 epochs)")
            btn_curve = gr.Button("加载", variant="primary")
            curve_plot = gr.Plot()
            btn_curve.click(show_curve, outputs=[curve_plot])

        # ── Tab 5: 下游任务 - 湿地监测 ──
        with gr.Tab("🌊 湿地监测"):
            gr.Markdown("""
### 湿地变化监测 — 少样本变化检测
基于 V2 Embedding + kNN-5 分类器，只需标注少量像素即可训练。

**当前最佳**: AUC = **0.712** (500-shot)
""")
            with gr.Row():
                wetland_patch = gr.Dropdown(choices=wetland_patches, value=wetland_patches[0] if wetland_patches else None, label="Patch ID")
                shot_slider = gr.Slider(minimum=1, maximum=1000, value=500, step=50, label="Shot Count")
            btn_wetland = gr.Button("🚀 运行变化检测", variant="primary")
            with gr.Row():
                wetland_image = gr.Image(label="变化检测热力图", height=400)
                wetland_stats = gr.Markdown("### 统计信息")

            btn_wetland.click(
                fn=run_change_detection_with_knn,
                inputs=[wetland_patch, shot_slider],
                outputs=[wetland_image, wetland_stats]
            )

        # ── Tab 6: 少样本效果曲线 ──
        with gr.Tab("📊 少样本效果"):
            gr.Markdown("### 少样本性能曲线")
            gr.Markdown("不同 shot count 下的 AUC-ROC 变化。")
            btn_curve2 = gr.Button("加载曲线", variant="primary")
            curve_image = gr.Plot()
            btn_curve2.click(fn=plot_fewshot_curve, outputs=[curve_image])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
