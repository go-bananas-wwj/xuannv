"""验证 Planet/Landsat resize 修复效果."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.aef.data.transforms import read_tif

PATCH = "patch_000036"
IMAGE_SIZE = 128
OUTPUT_DIR = Path("outputs/viz_preview")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Planet: 验证下采样后保留完整区域 ----
planet_path = Path(f"/workspace/xuannv/data_raw/beijing/planetscene/{PATCH}/20251209.tif")
planet_old = read_tif(planet_path, IMAGE_SIZE, resize_mode="crop")  # 模拟旧行为
planet_new = read_tif(planet_path, IMAGE_SIZE, resize_mode="area")  # 新行为

print(f"Planet old (crop): shape={planet_old.shape}, range=[{planet_old.min():.1f}, {planet_old.max():.1f}]")
print(f"Planet new (area): shape={planet_new.shape}, range=[{planet_new.min():.1f}, {planet_new.max():.1f}]")

# 旧 crop 只保留中心区域，新 area 是全局下采样
# 通过计算非零像素的比例或边缘梯度来验证
def edge_energy(img):
    """计算图像边缘变化强度（area resize 应全局有变化，crop 也是全局）."""
    gy, gx = np.gradient(img.astype(np.float32))
    return np.sqrt(gy**2 + gx**2).mean()

print(f"  Planet old edge energy: {edge_energy(planet_old[0]):.2f}")
print(f"  Planet new edge energy: {edge_energy(planet_new[0]):.2f}")

# ---- Landsat: 验证上采样后无大面积 pad ----
landsat_path = Path(f"/workspace/xuannv/data_raw/haidian/scenes/{PATCH}/landsat/20250214.tif")
# 模拟旧行为：先 read raw 再 pad
import rasterio
with rasterio.open(landsat_path) as src:
    raw = src.read()
C, H, W = raw.shape
pad_h = max(0, IMAGE_SIZE - H)
pad_w = max(0, IMAGE_SIZE - W)
landsat_old = np.pad(raw, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')[:, :IMAGE_SIZE, :IMAGE_SIZE]
landsat_new = read_tif(landsat_path, IMAGE_SIZE, resize_mode="bilinear")

print(f"\nLandsat old (pad): shape={landsat_old.shape}, range=[{landsat_old.min():.4f}, {landsat_old.max():.4f}]")
print(f"Landsat new (bilinear): shape={landsat_new.shape}, range=[{landsat_new.min():.4f}, {landsat_new.max():.4f}]")

# 旧 pad：右下角大面积为同一值（边缘复制）
def unique_ratio(img):
    """计算唯一值占比，pad 图像该值极低."""
    flat = img.astype(np.float32).ravel()
    return len(np.unique(flat)) / flat.size

print(f"  Landsat old unique ratio: {unique_ratio(landsat_old[0]):.4f}")
print(f"  Landsat new unique ratio: {unique_ratio(landsat_new[0]):.4f}")

# ---- S2: 验证保持 center crop ----
s2_dir = Path(f"/workspace/xuannv/data_raw/haidian/scenes/{PATCH}/s2")
s2_file = sorted(s2_dir.glob("*.tif"))[0]
s2_new = read_tif(s2_file, IMAGE_SIZE)
print(f"\nS2 (auto crop): shape={s2_new.shape}, range=[{s2_new.min():.2f}, {s2_new.max():.2f}]")

# ---- 可视化对比 ----
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

def to_rgb(img_chw):
    """取前3通道，min-max归一化到0-1."""
    rgb = img_chw[:3].astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return np.transpose(rgb, (1, 2, 0))

# Planet
axes[0, 0].imshow(to_rgb(planet_old))
axes[0, 0].set_title("Planet OLD (center crop)")
axes[0, 1].imshow(to_rgb(planet_new))
axes[0, 1].set_title("Planet NEW (area resize)")
axes[0, 2].imshow(np.abs(planet_new.astype(float) - planet_old.astype(float)).mean(axis=0), cmap="hot")
axes[0, 2].set_title("|NEW - OLD|")

# Landsat
axes[1, 0].imshow(to_rgb(landsat_old))
axes[1, 0].set_title("Landsat OLD (edge pad)")
axes[1, 1].imshow(to_rgb(landsat_new))
axes[1, 1].set_title("Landsat NEW (bilinear)")
axes[1, 2].imshow(np.abs(landsat_new.astype(float) - landsat_old.astype(float)).mean(axis=0), cmap="hot")
axes[1, 2].set_title("|NEW - OLD|")

# S2
with rasterio.open(s2_file) as src:
    s2_raw = src.read()
C2, H2, W2 = s2_raw.shape
start_h = (H2 - IMAGE_SIZE) // 2
start_w = (W2 - IMAGE_SIZE) // 2
s2_old = s2_raw[:, start_h:start_h+IMAGE_SIZE, start_w:start_w+IMAGE_SIZE]
axes[2, 0].imshow(to_rgb(s2_old))
axes[2, 0].set_title("S2 (unchanged crop)")
axes[2, 1].imshow(to_rgb(s2_new))
axes[2, 1].set_title("S2 (auto crop)")
axes[2, 2].imshow(np.abs(s2_new.astype(float) - s2_old.astype(float)).mean(axis=0), cmap="hot")
axes[2, 2].set_title("|auto - manual|")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
out_path = OUTPUT_DIR / "resize_fix_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"\n可视化已保存至: {out_path}")

# ---- 地理范围验证 ----
print("\n--- 地理范围一致性检查 ---")
from rasterio.coords import BoundingBox

def get_bounds(path):
    with rasterio.open(path) as src:
        return src.bounds

bounds_planet = get_bounds(planet_path)
bounds_s2 = get_bounds(s2_file)
print(f"Planet bounds: {bounds_planet}")
print(f"S2 bounds:     {bounds_s2}")
print(f"Bounds match:  {np.allclose(bounds_planet, bounds_s2, atol=0.1)}")
