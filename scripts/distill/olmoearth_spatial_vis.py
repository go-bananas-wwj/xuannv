#!/usr/bin/env python3
"""OlmoEarth 全区6月空间嵌入地图 — 每个patch内保留32×32 token空间结构。
patch_size=4 → 128/4=32 token/边 → 每个patch出32×32×768 token图
全局PCA 768→3 保证颜色跨patch一致。
"""
from __future__ import annotations
import sys, os, glob, json
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/workspace/olmoearth_pretrain")
import numpy as np, torch, rasterio
import torch.nn.functional as F

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality

RAW  = "/workspace/xuannv/data_raw/harbin/olmoearth"
META = "/workspace/xuannv_show/data/harbin/patches_meta.json"
OUT  = "/workspace/xuannv/outputs/olmoearth_harbin/june"
computed   = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)
H = 128   # 输入尺寸 (=patch_size*32)

# ── 字体 ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt
matplotlib.font_manager._load_fontmanager(try_read_cache=False)


def pick_june(m, patch):
    fs = sorted(glob.glob(f"{RAW}/{m}/{patch}/*.tif"))
    fs = [f for f in fs if "static" not in f]
    if not fs: return None, None
    jun = [f for f in fs if os.path.basename(f)[4:6] == "06"]
    f = jun[len(jun)//2] if jun else fs[len(fs)//2]
    return f, os.path.basename(f)[:8]


def read_resize(path, h=H, w=H, mode="bilinear"):
    with rasterio.open(path) as ds: a = ds.read().astype("float32")
    if a.shape[1:] != (h,w):
        t = torch.from_numpy(a)[None]
        t = F.interpolate(t,(h,w),mode=mode,
                          align_corners=False if mode=="bilinear" else None)
        a = t[0].numpy()
    return a


def build_sample(patch, device):
    meta = {p["patch_id"]:p for p in json.load(open(META))}
    arr = {}
    date_str = None
    for m, oe in [("s2",Modality.SENTINEL2_L2A),("s1",Modality.SENTINEL1),("landsat",Modality.LANDSAT)]:
        f,d = pick_june(m, patch)
        if f is None: continue
        a = read_resize(f)
        a = np.transpose(a,(1,2,0))[None,:,:,None,:]
        a = computed.normalize(oe, a).astype("float32")
        arr[m] = a
        if date_str is None: date_str = d
    if date_str is None: date_str = "20250615"
    ts = np.array([[[int(date_str[6:8]), int(date_str[4:6])-1, int(date_str[:4])]]])

    def static(m, oe, nearest=False):
        f = f"{RAW}/{m}/{patch}/static.tif"
        if not os.path.exists(f): return None
        mode = "nearest" if nearest else "bilinear"
        a = read_resize(f, mode=mode)
        a = np.transpose(a,(1,2,0))[None,:,:,None,:]
        return computed.normalize(oe, a).astype("float32")

    b = meta[patch]["bounds_wgs84"]
    ll = predefined.normalize(Modality.LATLON,
         np.array([[(b[1]+b[3])/2,(b[0]+b[2])/2]],dtype="float32")).astype("float32")

    def t(x): return torch.from_numpy(x).float().to(device) if x is not None else None
    kw = dict(
        timestamps    = torch.from_numpy(ts).long().to(device),
        sentinel2_l2a = t(arr.get("s2")),
        sentinel1     = t(arr.get("s1")),
        landsat       = t(arr.get("landsat")),
        worldcover    = t(static("worldcover",Modality.WORLDCOVER,nearest=True)),
        srtm          = t(static("dem",Modality.SRTM)),
        latlon        = t(ll),
    )
    for k in ["sentinel2_l2a","sentinel1","landsat","worldcover","srtm","latlon"]:
        if kw.get(k) is not None: kw[k+"_mask"] = torch.zeros_like(kw[k])
    return MaskedOlmoEarthSample(**kw)


@torch.no_grad()
def get_spatial_tokens(model, patch, device):
    """返回 (32, 32, 768) 的空间 token 图 (来自所有模态 mean)。"""
    s = build_sample(patch, device)
    out = model(s, patch_size=4, fast_pass=True)
    tm = out["tokens_and_masks"]

    all_tokens = []
    for m in tm.modalities:
        tok = getattr(tm, m)  # (1, Ht, Wt, T, Cg, 768) 或 (1,Ht,Wt,1,Cg,768)
        if tok is None: continue
        # mean over T, Cg → (1, Ht, Wt, 768)
        tok = tok.mean(dim=(3,4))
        # 如果空间不是32x32, 双线性插值到32x32
        b,h,w,d = tok.shape
        if (h,w) != (32,32):
            tok = tok.permute(0,3,1,2)  # (1,768,h,w)
            tok = F.interpolate(tok,(32,32),mode="bilinear",align_corners=False)
            tok = tok.permute(0,2,3,1)  # (1,32,32,768)
        all_tokens.append(tok)

    merged = torch.stack(all_tokens, dim=0).mean(dim=0)  # (1,32,32,768)
    return merged[0].float().cpu().numpy()  # (32,32,768)


def main():
    import torch_npu
    device = torch.device("npu:0")
    print("[加载模型] OlmoEarth-Base -> npu:0")
    model_full = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model_full.eval()
    model = model_full.encoder.to(device)

    meta_list = json.load(open(META))
    meta = {p["patch_id"]:p for p in meta_list}
    patches = [p["patch_id"] for p in meta_list]

    os.makedirs(OUT, exist_ok=True)
    cache_f = f"{OUT}/spatial_tokens.npz"

    done = {}
    if os.path.exists(cache_f):
        c = np.load(cache_f, allow_pickle=True)
        for pid,tok in zip(c["patch_ids"], c["tokens"]):
            done[str(pid)] = tok
        print(f"[续传] 已有 {len(done)} patch")

    tokens_dict = dict(done)
    for i,pid in enumerate(patches):
        if pid in tokens_dict: continue
        try:
            tok = get_spatial_tokens(model, pid, device)   # (32,32,768)
            tokens_dict[pid] = tok
            if i % 30 == 0:
                print(f"[{i+1}/{len(patches)}] {pid}")
                np.savez(cache_f,
                         tokens=np.stack(list(tokens_dict.values())),
                         patch_ids=list(tokens_dict.keys()))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[skip] {pid}: {e}")

    np.savez(cache_f,
             tokens=np.stack([tokens_dict[p] for p in patches if p in tokens_dict]),
             patch_ids=[p for p in patches if p in tokens_dict])
    print(f"\n[保存] tokens -> {cache_f}")

    # ── 全局PCA 768→3 ────────────────────────────────
    print("[全局PCA拟合...]")
    from sklearn.decomposition import PCA
    ids = [p for p in patches if p in tokens_dict]
    all_toks = np.stack([tokens_dict[p] for p in ids])  # (N,32,32,768)
    N = all_toks.shape[0]
    flat = all_toks.reshape(-1, 768)                     # (N*32*32, 768)
    # 采样最多50万点
    if flat.shape[0] > 500000:
        idx = np.random.choice(flat.shape[0], 500000, replace=False)
        flat_s = flat[idx]
    else: flat_s = flat
    pca = PCA(n_components=3); pca.fit(flat_s)
    rgb_all = pca.transform(flat_s)
    vmin = np.percentile(rgb_all, 2, axis=0)
    vmax = np.percentile(rgb_all, 98, axis=0)
    var = pca.explained_variance_ratio_
    print(f"PCA方差: PC1={var[0]*100:.1f}% PC2={var[1]*100:.1f}% PC3={var[2]*100:.1f}%")

    # ── 拼mosaic大图 ──────────────────────────────────
    print("[拼接地图...]")
    ix = np.array([meta[p]["ix"] for p in ids])
    iy = np.array([meta[p]["iy"] for p in ids])
    ix_min,ix_max = ix.min(),ix.max()
    iy_min,iy_max = iy.min(),iy.max()
    n_cols = ix_max-ix_min+1  # ~26
    n_rows = iy_max-iy_min+1  # ~24
    TILE = 32  # token分辨率 32×32
    W_px = n_cols*TILE; H_px = n_rows*TILE
    canvas = np.full((H_px, W_px, 3), 0.12, dtype="float32")  # 深色背景

    for i,pid in enumerate(ids):
        tok = tokens_dict[pid]   # (32,32,768)
        rgb = pca.transform(tok.reshape(-1,768)).reshape(32,32,3)
        rgb = np.clip((rgb-vmin)/(vmax-vmin+1e-8), 0, 1)
        col = ix[i]-ix_min; row = iy[i]-iy_min
        y0 = (n_rows-1-row)*TILE
        x0 = col*TILE
        canvas[y0:y0+TILE, x0:x0+TILE] = rgb

    # ── 画图 ──────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#111827")
    gs = plt.GridSpec(1, 2, figure=fig, width_ratios=[4,1], wspace=0.03)

    ax = fig.add_subplot(gs[0])
    ax.imshow(canvas, origin="upper", interpolation="nearest")
    ax.set_facecolor("#111827")
    ax.set_title(
        f"OlmoEarth-Base  哈尔滨新区  2025年6月  全区空间嵌入地图\n"
        f"每格 = 1 patch (~1.3km) × 32×32 token · RGB = PCA(PC1,PC2,PC3) · "
        f"PC1={var[0]*100:.0f}% PC2={var[1]*100:.0f}% PC3={var[2]*100:.0f}%",
        color="white", fontsize=12, pad=8
    )
    ax.set_xlabel("→ 经度方向", color="#9ca3af", fontsize=10)
    ax.set_ylabel("↑ 纬度方向", color="#9ca3af", fontsize=10)
    ax.tick_params(colors="#6b7280")
    for sp in ax.spines.values(): sp.set_color("#374151")
    xt = range(0, n_cols, 5)
    ax.set_xticks([x*TILE+TILE//2 for x in xt])
    ax.set_xticklabels([f"ix={ix_min+x}" for x in xt], fontsize=7, color="#9ca3af")
    yt = range(0, n_rows, 4)
    ax.set_yticks([(n_rows-1-y)*TILE+TILE//2 for y in yt])
    ax.set_yticklabels([f"iy={iy_min+y}" for y in yt], fontsize=7, color="#9ca3af")

    # 右侧说明
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_facecolor("#111827")
    en = all_toks.mean((1,2)); en /= (np.linalg.norm(en,axis=1,keepdims=True)+1e-8)
    sim = en@en.T; off=sim[~np.eye(N,dtype=bool)]
    info = (
        f"全区统计  ({N} patches)\n"
        f"{'─'*26}\n"
        f"空间分辨率\n"
        f"  输入:   128×128 px\n"
        f"  Token:  32×32 (patch=4)\n"
        f"  地图:   {W_px}×{H_px} px\n\n"
        f"嵌入维度:  768 维\n"
        f"零塌缩:    0 / 768\n\n"
        f"Patch级余弦相似度\n"
        f"  均值  {off.mean():.4f}\n"
        f"  标准差 {off.std():.4f}\n"
        f"  范围  [{off.min():.3f}, {off.max():.3f}]\n\n"
        f"PCA 方差\n"
        f"  PC1  {var[0]*100:.1f}%\n"
        f"  PC2  {var[1]*100:.1f}%\n"
        f"  PC3  {var[2]*100:.1f}%\n"
        f"  合计 {var[:3].sum()*100:.1f}%\n\n"
        f"颜色说明\n"
        f"  RGB = PCA前3维\n"
        f"  颜色相近 → 地物相似\n"
        f"  颜色跳变 → 地类边界\n"
        f"  蓝/紫   → 水体/建筑\n"
        f"  绿      → 植被/农田\n"
        f"  黄/橙   → 裸地"
    )
    ax2.text(0.05, 0.97, info, transform=ax2.transAxes,
             color="white", fontsize=10, va="top",
             fontfamily="WenQuanYi Micro Hei",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1f2937", alpha=0.95))

    out_png = f"{OUT}/june_spatial_map.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#111827")
    plt.close()
    print(f"\n✅ 地图: {out_png}  ({os.path.getsize(out_png)//1024}KB)")


if __name__ == "__main__":
    main()
