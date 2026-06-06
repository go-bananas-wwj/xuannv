#!/usr/bin/env python3
"""预计算所有月份的 OlmoEarth spatial tokens (fast_pass=True).
输出: /workspace/xuannv/outputs/olmoearth_harbin/{MM}/spatial_tokens.npz  shape=(424, 32, 32, 768)
已有: 06 月 → 跳过 (已在 june/ 目录下)
"""
from __future__ import annotations
import sys, os, glob, json
sys.stdout.reconfigure(line_buffering=True)
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"]:
    os.environ.pop(k, None)
sys.path.insert(0, "/workspace/olmoearth_pretrain")

import numpy as np, torch
import torch.nn.functional as F

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality
import rasterio

RAW  = "/workspace/xuannv/data_raw/olmoearth_harbin"
META = "/workspace/xuannv_show/data/harbin/patches_meta.json"
OUT  = "/workspace/xuannv/outputs/olmoearth_harbin"

# June spatial tokens 已存在于旧路径，复制到标准路径
JUNE_SRC = "/workspace/xuannv/outputs/olmoearth_harbin/june/spatial_tokens.npz"
JUNE_DST = "/workspace/xuannv/outputs/olmoearth_harbin/06/spatial_tokens.npz"

computed   = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)
H = 128

TARGET_MONTHS = [4, 8, 9, 10]   # 06 已有，跳过


def pick_month(modality, patch, target_month):
    fs = sorted(glob.glob(f"{RAW}/{modality}/{patch}/*.tif"))
    fs = [f for f in fs if "static" not in f]
    if not fs: return None
    def month_of(f): return int(os.path.basename(f)[4:6])
    fs.sort(key=lambda f: abs(month_of(f) - target_month))
    return fs[0]


def read_resize(path, h=H, w=H, mode="bilinear"):
    with rasterio.open(path) as ds: a = ds.read().astype("float32")
    if a.shape[1:] != (h, w):
        t = F.interpolate(torch.from_numpy(a)[None], (h, w),
                          mode=mode, align_corners=False if mode=="bilinear" else None)
        a = t[0].numpy()
    return a


def build_sample(patch, target_month, meta_dict, device):
    arr = {}
    date_str = None
    for m, oe in [("s2", Modality.SENTINEL2_L2A),
                  ("s1", Modality.SENTINEL1),
                  ("landsat", Modality.LANDSAT)]:
        f = pick_month(m, patch, target_month)
        if f is None: continue
        a = read_resize(f)
        a = np.transpose(a,(1,2,0))[None,:,:,None,:]
        a = computed.normalize(oe, a).astype("float32")
        arr[m] = a
        if date_str is None:
            date_str = os.path.basename(f)[:8]
    if date_str is None: date_str = f"2025{target_month:02d}15"
    ts = np.array([[[int(date_str[6:8]), int(date_str[4:6])-1, int(date_str[:4])]]])

    def static(m, oe, nearest=False):
        f2 = f"{RAW}/{m}/{patch}/static.tif"
        if not os.path.exists(f2): return None
        a = read_resize(f2, mode="nearest" if nearest else "bilinear")
        a = np.transpose(a,(1,2,0))[None,:,:,None,:]
        return computed.normalize(oe, a).astype("float32")

    b = meta_dict[patch]["bounds_wgs84"]
    ll = predefined.normalize(Modality.LATLON,
         np.array([[(b[1]+b[3])/2,(b[0]+b[2])/2]], dtype="float32")).astype("float32")

    def t(x): return torch.from_numpy(x).float().to(device) if x is not None else None
    kw = dict(
        timestamps    = torch.from_numpy(ts).long().to(device),
        sentinel2_l2a = t(arr.get("s2")),
        sentinel1     = t(arr.get("s1")),
        landsat       = t(arr.get("landsat")),
        worldcover    = t(static("worldcover", Modality.WORLDCOVER, nearest=True)),
        srtm          = t(static("dem", Modality.SRTM)),
        latlon        = t(ll),
    )
    for k in ["sentinel2_l2a","sentinel1","landsat","worldcover","srtm","latlon"]:
        if kw.get(k) is not None: kw[k+"_mask"] = torch.zeros_like(kw[k])
    return MaskedOlmoEarthSample(**kw)


@torch.no_grad()
def extract_spatial_tokens(model, patch, target_month, meta_dict, device):
    """返回 (32, 32, 768) spatial tokens，取 S2 L2A modality 均值。"""
    s = build_sample(patch, target_month, meta_dict, device)
    out = model(s, patch_size=4, fast_pass=True)
    # tokens_and_masks.sentinel2_l2a shape: (1, 32, 32, 1, N_views, 768)
    tokens_attr = out["tokens_and_masks"]
    s2_tokens = tokens_attr.sentinel2_l2a  # (1, 32, 32, 1, Nv, 768) or (1, H, W, T, Nv, 768)
    # mean over all view / modality dims except spatial and feature
    # s2_tokens: (1, 32, 32, T, Nv, 768) → mean T,Nv → (1, 32, 32, 768)
    tokens = s2_tokens[0]  # (32, 32, T, Nv, 768)
    tokens = tokens.mean(dim=(2, 3))  # (32, 32, 768)
    return tokens.float().cpu().numpy()


def main():
    import torch_npu  # noqa
    device = torch.device("npu:0")

    # 若 june 目标路径不存在，拷贝
    if not os.path.exists(JUNE_DST) and os.path.exists(JUNE_SRC):
        import shutil
        os.makedirs(os.path.dirname(JUNE_DST), exist_ok=True)
        shutil.copy2(JUNE_SRC, JUNE_DST)
        print(f"[复制] June spatial tokens: {JUNE_SRC} → {JUNE_DST}")

    print("[加载 OlmoEarth 模型]")
    model_full = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model_full.eval()
    model = model_full.encoder.to(device)

    meta_list = json.load(open(META))
    meta_dict = {p["patch_id"]: p for p in meta_list}
    patches   = [p["patch_id"] for p in meta_list]
    print(f"[共 {len(patches)} patches]")

    for month in TARGET_MONTHS:
        out_dir = f"{OUT}/{month:02d}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/spatial_tokens.npz"

        if os.path.exists(out_path):
            print(f"[月份{month:02d}] 已有 spatial_tokens.npz，跳过")
            continue

        print(f"\n===== 月份 {month:02d} ({len(patches)} patches) =====")
        tokens_list = {}

        # 续传支持
        tmp_path = f"{out_dir}/spatial_tokens_partial.npy"

        for i, pid in enumerate(patches):
            if pid in tokens_list: continue
            try:
                tok = extract_spatial_tokens(model, pid, month, meta_dict, device)
                tokens_list[pid] = tok
                if (i+1) % 50 == 0 or (i+1) == len(patches):
                    print(f"  [{i+1}/{len(patches)}] {pid}  shape={tok.shape}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [skip] {pid}: {e}")

        valid_patches = [p for p in patches if p in tokens_list]
        arr = np.stack([tokens_list[p] for p in valid_patches]).astype(np.float16)
        ids = np.array(valid_patches)
        np.savez(out_path, tokens=arr, patch_ids=ids)
        print(f"  ✅ 保存 {len(valid_patches)} patches → {out_path}  shape={arr.shape}")

    print("\n✅ 所有月份 spatial tokens 提取完成")
    print(f"   June (已有): {JUNE_DST}")
    for m in TARGET_MONTHS:
        p = f"{OUT}/{m:02d}/spatial_tokens.npz"
        if os.path.exists(p):
            d = np.load(p)
            print(f"   {m:02d}: {d['tokens'].shape}")


if __name__ == "__main__":
    main()
