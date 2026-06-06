#!/usr/bin/env python3
"""海淀区多时相全局嵌入提取 — 用于蒸馏训练。
对 4 / 6 / 8 / 9 / 10 月各选最近一景，提取 project_and_aggregate 全局 768D 向量。
输出: /workspace/xuannv/outputs/olmoearth_haidian/{MM}/emb_all.npz  (320, 768)
"""
from __future__ import annotations
import sys, os, glob, json, argparse
sys.stdout.reconfigure(line_buffering=True)
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"]:
    os.environ.pop(k, None)
sys.path.insert(0, "/workspace/olmoearth_pretrain")

import numpy as np, torch, rasterio
import torch.nn.functional as F

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality

RAW  = "/workspace/xuannv/data_raw/olmoearth_haidian"
META = "/workspace/xuannv/data_raw/olmoearth_haidian/patches_meta.json"
OUT  = "/workspace/xuannv/outputs/olmoearth_haidian"
computed   = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)
H = 128

TARGET_MONTHS = [4, 6, 8, 9, 10]


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


def build_sample(patch, target_month, device):
    meta = {p["patch_id"]:p for p in json.load(open(META))}
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
        f = f"{RAW}/{m}/{patch}/static.tif"
        if not os.path.exists(f): return None
        a = read_resize(f, mode="nearest" if nearest else "bilinear")
        a = np.transpose(a,(1,2,0))[None,:,:,None,:]
        return computed.normalize(oe, a).astype("float32")

    b = meta[patch]["bounds_wgs84"]
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
def embed_patch(model, patch, target_month, device):
    s = build_sample(patch, target_month, device)
    out = model(s, patch_size=4, fast_pass=False)
    return out["project_aggregated"][0].float().cpu().numpy()


def main():
    import torch_npu
    device = torch.device("npu:0")
    print("[加载模型]")
    model_full = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model_full.eval()
    model = model_full.encoder.to(device)

    meta_list = json.load(open(META))
    patches   = [p["patch_id"] for p in meta_list]

    for month in TARGET_MONTHS:
        out_dir = f"{OUT}/{month:02d}"
        os.makedirs(out_dir, exist_ok=True)
        cache_f = f"{out_dir}/emb_all.npz"

        if os.path.exists(cache_f):
            print(f"[月份{month:02d}] 已有，跳过")
            continue

        print(f"\n===== 月份 {month:02d} ({len(patches)} patches) =====")
        embs = {}

        for i, pid in enumerate(patches):
            if pid in embs: continue
            try:
                embs[pid] = embed_patch(model, pid, month, device)
                if i % 50 == 0:
                    print(f"  [{i+1}/{len(patches)}] {pid}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [skip] {pid}: {e}")

        arr = np.stack([embs[p] for p in patches if p in embs])
        ids = [p for p in patches if p in embs]
        np.savez(cache_f, embeddings=arr, patch_ids=ids)
        print(f"  保存 {len(ids)} patches → {cache_f}")

    print("\n✅ 所有月份完成")


if __name__ == "__main__":
    main()
