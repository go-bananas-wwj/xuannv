#!/usr/bin/env python3
"""全区6月份嵌入生成 + 地图可视化。
策略: 每个patch取6月份(2025-06-xx)景, 若缺则取最近日期。
输出: /workspace/xuannv/outputs/olmoearth_harbin/june/emb_all.npz + june_map.png
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/workspace/olmoearth_pretrain")
import glob, json
import numpy as np
import torch
import rasterio
from datetime import datetime

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality

RAW   = "/workspace/xuannv/data_raw/olmoearth_harbin"
META  = "/workspace/xuannv_show/data/harbin/patches_meta.json"
OUT   = "/workspace/xuannv/outputs/olmoearth_harbin/june"
MOD2OE = {"s2": Modality.SENTINEL2_L2A, "s1": Modality.SENTINEL1,
           "landsat": Modality.LANDSAT}
computed  = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)


def pick_june(m, patch):
    """优先取6月份景, 否则取最近日期景; 返回 (path, date_str) 或 None。"""
    fs = sorted(glob.glob(f"{RAW}/{m}/{patch}/*.tif"))
    fs = [f for f in fs if "static" not in f]
    if not fs: return None, None
    jun = [f for f in fs if os.path.basename(f)[4:6] == "06"]
    if jun: f = jun[len(jun)//2]   # 取6月中间景
    else:   f = fs[len(fs)//2]     # fallback: 中间景
    d = os.path.basename(f)[:8]
    return f, d


def read_tif_resize(path, H=128, W=128):
    with rasterio.open(path) as ds:
        a = ds.read().astype(np.float32)
    if a.shape[1:] != (H, W):
        t = torch.from_numpy(a)[None]
        mode = "nearest" if a.shape[0]==1 else "bilinear"
        t = torch.nn.functional.interpolate(t, (H,W), mode=mode, align_corners=False if mode=="bilinear" else None)
        a = t[0].numpy()
    return a


def build_sample(patch, device, H=128):
    meta_list = json.load(open(META))
    meta = {p["patch_id"]: p for p in meta_list}

    # 时序各模态取1景(T=1), timestamps对齐到s2日期
    arr = {}
    date_str = None
    for m in ["s2","s1","landsat"]:
        f, d = pick_june(m, patch)
        if f is None: continue
        a = read_tif_resize(f, H, H)          # (C,H,W)
        a = np.transpose(a, (1,2,0))           # (H,W,C)
        a = a[None,:,:,None,:]                 # (1,H,W,1,C) T=1
        a = computed.normalize(MOD2OE[m], a).astype(np.float32)
        arr[m] = a
        if date_str is None: date_str = d

    if date_str is None: date_str = "20250615"
    ts = np.array([[int(date_str[6:8]), int(date_str[4:6])-1, int(date_str[:4])]])  # (1,3)

    # 静态模态
    def static(m, oe, nearest=False):
        f = f"{RAW}/{m}/{patch}/static.tif"
        if not os.path.exists(f): return None
        a = read_tif_resize(f, H, H)
        a = np.transpose(a,(1,2,0))[None,:,:,None,:]
        return computed.normalize(oe, a).astype(np.float32)

    wc   = static("worldcover", Modality.WORLDCOVER, nearest=True)
    srtm = static("dem", Modality.SRTM)

    b = meta[patch]["bounds_wgs84"]
    ll = predefined.normalize(Modality.LATLON,
         np.array([[(b[1]+b[3])/2, (b[0]+b[2])/2]], dtype=np.float32)).astype(np.float32)

    def t(x): return torch.from_numpy(x).float().to(device) if x is not None else None
    kw = dict(
        timestamps   = torch.from_numpy(ts)[None].long().to(device),  # (1,1,3)
        sentinel2_l2a= t(arr.get("s2")),
        sentinel1    = t(arr.get("s1")),
        landsat      = t(arr.get("landsat")),
        worldcover   = t(wc),
        srtm         = t(srtm),
        latlon       = t(ll),
    )
    for k in ["sentinel2_l2a","sentinel1","landsat","worldcover","srtm","latlon"]:
        if kw.get(k) is not None:
            kw[k+"_mask"] = torch.zeros_like(kw[k])
    return MaskedOlmoEarthSample(**kw)


@torch.no_grad()
def embed_patch(model, patch, device):
    s = build_sample(patch, device)
    out = model(s, patch_size=4, fast_pass=False)
    return out["project_aggregated"][0].float().cpu().numpy()  # (768,)


def make_map(embs, ids, meta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    ix = np.array([meta[p]["ix"] for p in ids])
    iy = np.array([meta[p]["iy"] for p in ids])
    ix_min, ix_max = ix.min(), ix.max()
    iy_min, iy_max = iy.min(), iy.max()
    n_cols = ix_max - ix_min + 1   # ~26
    n_rows = iy_max - iy_min + 1   # ~24

    # 全局PCA 768->3 -> RGB
    pca = PCA(n_components=3)
    rgb3 = pca.fit_transform(embs)             # (N,3)
    vmin = np.percentile(rgb3, 2, axis=0)
    vmax = np.percentile(rgb3, 98, axis=0)
    rgb3 = (rgb3 - vmin) / (vmax - vmin + 1e-8)
    rgb3 = np.clip(rgb3, 0, 1)
    TILE = 64  # 每个patch格子像素

    canvas = np.ones((n_rows*TILE, n_cols*TILE, 3), dtype=np.float32) * 0.15  # 深灰背景

    # 填色块
    for i, pid in enumerate(ids):
        col = ix[i] - ix_min
        row = iy[i] - iy_min
        y0 = (n_rows - 1 - row) * TILE  # iy越大在越上方
        x0 = col * TILE
        canvas[y0:y0+TILE, x0:x0+TILE] = rgb3[i]

    var = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                              gridspec_kw={"width_ratios":[3,1]})
    fig.patch.set_facecolor("#111827")

    ax = axes[0]
    ax.imshow(canvas, origin="upper", interpolation="nearest")
    ax.set_facecolor("#111827")
    ax.set_title(
        f"OlmoEarth-Base  哈尔滨新区  2025年6月  全区嵌入地图\n"
        f"PCA前3维累计方差: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%",
        color="white", fontsize=13, pad=10
    )
    ax.set_xlabel("→ 经度 (ix)", color="#aaa", fontsize=10)
    ax.set_ylabel("↑ 纬度 (iy)", color="#aaa", fontsize=10)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_color("#333")
    # x轴刻度(经度大约) - 每5格标一次
    xticks = range(0, n_cols, 5)
    ax.set_xticks([x*TILE+TILE//2 for x in xticks])
    ax.set_xticklabels([f"ix={ix_min+x}" for x in xticks], fontsize=7, color="#aaa")
    yticks = range(0, n_rows, 4)
    ax.set_yticks([(n_rows-1-y)*TILE+TILE//2 for y in yticks])
    ax.set_yticklabels([f"iy={iy_min+y}" for y in yticks], fontsize=7, color="#aaa")

    # 右侧: 统计面板
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ax2.axis("off")
    en = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-8)
    sim = en @ en.T
    off = sim[~np.eye(len(sim), dtype=bool)]
    std_d = embs.std(0)
    dead = (std_d<1e-4).sum()

    stats_text = (
        f"全区嵌入统计  ({len(ids)} patches)\n"
        f"{'─'*30}\n"
        f"嵌入维度:    768 维\n"
        f"零塌缩维度:  {dead} / 768\n\n"
        f"两两余弦相似度\n"
        f"  均值  {off.mean():.4f}\n"
        f"  标准差 {off.std():.4f}\n"
        f"  最小值 {off.min():.4f}\n"
        f"  最大值 {off.max():.4f}\n\n"
        f"PCA 方差解释\n"
        f"  PC1  {var[0]*100:.1f}%\n"
        f"  PC2  {var[1]*100:.1f}%\n"
        f"  PC3  {var[2]*100:.1f}%\n"
        f"  合计 {var[:3].sum()*100:.1f}%\n\n"
        f"颜色含义:\n"
        f"  RGB = PCA(PC1,PC2,PC3)\n"
        f"  相近颜色 ≈ 相似地物\n"
        f"  颜色跳变 ≈ 土地利用\n"
        f"  类型不同"
    )
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             color="white", fontsize=11, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#1f2937", alpha=0.9))

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ 地图保存: {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu; torch.device("npu:0")
        device = torch.device("npu:0")
    else:
        device = torch.device(args.device)

    print(f"[加载模型] OlmoEarth-Base -> {device}")
    model_full = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model_full.eval()
    model = model_full.encoder.to(device)

    meta_list = json.load(open(META))
    meta = {p["patch_id"]: p for p in meta_list}
    all_patches = [p["patch_id"] for p in meta_list]

    os.makedirs(OUT, exist_ok=True)
    cache = f"{OUT}/emb_all.npz"

    # 断点续传
    done = {}
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        for pid, e in zip(c["patch_ids"], c["embeddings"]):
            done[str(pid)] = e
        print(f"[续传] 已有 {len(done)} patch")

    embs, ids = [], []
    for i, pid in enumerate(all_patches):
        if str(pid) in done:
            embs.append(done[str(pid)]); ids.append(str(pid)); continue
        try:
            e = embed_patch(model, pid, device)
            embs.append(e); ids.append(pid)
            done[str(pid)] = e
            if i % 20 == 0:
                print(f"[{i+1}/{len(all_patches)}] {pid} norm={np.linalg.norm(e):.3f}")
                np.savez(cache, embeddings=np.stack(list(done.values())),
                         patch_ids=list(done.keys()))
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[skip] {pid}: {ex}")

    embs = np.stack(embs)
    np.savez(cache, embeddings=embs, patch_ids=ids)
    print(f"\n[保存] {embs.shape} -> {cache}")

    print("\n[生成地图...]")
    make_map(embs, ids, meta, f"{OUT}/june_map.png")

    en = embs/(np.linalg.norm(embs,axis=1,keepdims=True)+1e-8)
    sim = en@en.T; off=sim[~np.eye(len(sim),dtype=bool)]
    print(f"\n===== 全区嵌入质量 =====")
    print(f"patch数: {len(ids)}, 维度: {embs.shape[1]}")
    print(f"余弦相似度: 均={off.mean():.4f} std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print(f"塌缩维度(std<1e-4): {(embs.std(0)<1e-4).sum()}/768")

if __name__ == "__main__":
    main()
