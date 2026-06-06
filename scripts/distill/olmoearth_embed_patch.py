#!/usr/bin/env python3
"""用 OlmoEarth-Base 对哈尔滨新区 patch 生成 patch 级 embedding, 测试嵌入判别力。

流程(对齐官方 generate_embeddings.ipynb):
  读5模态多时相tif -> 取最近T景对齐时间轴 -> 归一化(时序COMPUTED, latlon PREDEFINED)
  -> MaskedOlmoEarthSample(mask全0) -> encoder(patch_size=4, fast_pass) -> project_and_aggregate
  -> (B, 768) patch embedding
评估: 多patch生成embedding, 看 (1)空间邻近patch更相似 (2)embedding非塌缩(方差/有效维度)。
"""
from __future__ import annotations
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/workspace/olmoearth_pretrain")
import argparse, glob, json
import numpy as np
import torch
import rasterio
from datetime import datetime

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality

RAW = "/workspace/xuannv/data_raw/olmoearth_harbin"
META = "/workspace/xuannv_show/data/harbin/patches_meta.json"
# 时序模态: 取整年均匀采样 T 景
TEMPORAL = ["s2", "s1", "landsat"]
MOD2OE = {"s2": Modality.SENTINEL2_L2A, "s1": Modality.SENTINEL1, "landsat": Modality.LANDSAT}

computed = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)


def list_dates(patch, m):
    fs = sorted(glob.glob(f"{RAW}/{m}/{patch}/*.tif"))
    fs = [f for f in fs if "static" not in f]
    dates = [os.path.basename(f)[:8] for f in fs]
    return fs, dates


def pick_uniform(fs, dates, T):
    """整年均匀取T景。"""
    if len(fs) <= T:
        idx = list(range(len(fs)))
    else:
        idx = [round(i * (len(fs) - 1) / (T - 1)) for i in range(T)]
    return [fs[i] for i in idx], [dates[i] for i in idx]


def read_tif(path):
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)  # (C,H,W)


def build_sample(patch, T, device):
    """组装一个 patch 的 MaskedOlmoEarthSample。空间用 s2 的 129x129 作为基准。"""
    # 用 s2 时间轴作为统一时间戳；各时序模态各自均匀取T景对齐到同一时间槽
    s2_fs, s2_dates = list_dates(patch, "s2")
    s2_fs, s2_dates = pick_uniform(s2_fs, s2_dates, T)
    Tn = len(s2_fs)
    H = W = 128
    # timestamps: [day, month-1(0索引), year]
    ts = np.array([[int(d[6:8]), int(d[4:6]) - 1, int(d[:4])] for d in s2_dates], dtype=np.float32)

    sample_arrays = {}
    for m in TEMPORAL:
        fs, dates = list_dates(patch, m)
        if not fs:
            continue
        fs, dates = pick_uniform(fs, dates, Tn)
        # 读并堆叠 -> (T, C, H, W) -> 上采到129再 (H,W,T,C)
        frames = []
        for f in fs:
            a = read_tif(f)  # (C,h,w)
            C, h, w = a.shape
            if (h, w) != (H, W):
                t = torch.from_numpy(a)[None]
                t = torch.nn.functional.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
                a = t[0].numpy()
            frames.append(a)
        while len(frames) < Tn:
            frames.append(frames[-1])
        arr = np.stack(frames[:Tn], axis=0)            # (T,C,H,W)
        arr = np.transpose(arr, (2, 3, 0, 1))          # (H,W,T,C)
        arr = computed.normalize(MOD2OE[m], arr).astype(np.float32)
        sample_arrays[m] = arr[None]                   # (1,H,W,T,C)

    # 静态模态 (1,H,W,1,C)
    def static(m, oe):
        f = f"{RAW}/{m}/{patch}/static.tif"
        if not os.path.exists(f):
            return None
        a = read_tif(f)  # (C,h,w)
        C, h, w = a.shape
        if (h, w) != (H, W):
            t = torch.from_numpy(a)[None]
            t = torch.nn.functional.interpolate(t, size=(H, W), mode="nearest")
            a = t[0].numpy()
        a = np.transpose(a, (1, 2, 0))[None, :, :, None, :]  # (1,H,W,1,C)
        a = computed.normalize(oe, a).astype(np.float32)
        return a
    wc = static("worldcover", Modality.WORLDCOVER)
    srtm = static("dem", Modality.SRTM)

    # latlon (1,2) 用patch中心
    meta = {p["patch_id"]: p for p in json.load(open(META))}
    b = meta[patch]["bounds_wgs84"]
    lon = (b[0] + b[2]) / 2; lat = (b[1] + b[3]) / 2
    ll = predefined.normalize(Modality.LATLON, np.array([[lat, lon]], dtype=np.float32)).astype(np.float32)

    def t(x):
        return torch.from_numpy(x).float().to(device) if x is not None else None
    kw = dict(
        timestamps=torch.from_numpy(ts)[None].long().to(device),
        sentinel2_l2a=t(sample_arrays.get("s2")),
        sentinel1=t(sample_arrays.get("s1")),
        landsat=t(sample_arrays.get("landsat")),
        worldcover=t(wc),
        srtm=t(srtm),
        latlon=t(ll),
    )
    # 加 mask 全0
    for k in ["sentinel2_l2a", "sentinel1", "landsat", "worldcover", "srtm", "latlon"]:
        if kw.get(k) is not None:
            kw[k + "_mask"] = torch.zeros_like(kw[k])
    return MaskedOlmoEarthSample(**kw)


@torch.no_grad()
def embed(model, patch, T, device):
    s = build_sample(patch, T, device)
    out = model(s, patch_size=4, fast_pass=False)
    emb = out["project_aggregated"]  # (1,D) 模型自洽计算
    return emb[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", nargs="+", default=None, help="指定patch_id, 默认前若干")
    ap.add_argument("--n", type=int, default=20, help="默认取前n个patch")
    ap.add_argument("--T", type=int, default=6, help="时间步数")
    ap.add_argument("--device", default="npu")
    ap.add_argument("--out", default="/workspace/xuannv/outputs/olmoearth_harbin/embeddings.npz")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa
        device = torch.device("npu:0")
    else:
        device = torch.device(args.device)

    print(f"[load] OlmoEarth-Base -> {device}")
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model.eval()
    model = model.encoder.to(device)
    D = model.project_and_aggregate.projection[-1].out_features if hasattr(model.project_and_aggregate.projection[-1], "out_features") else 768
    print(f"[load] embedding维度 ~ {D}")

    all_p = [p["patch_id"] for p in json.load(open(META))]
    patches = args.patches or all_p[: args.n]
    embs, ids = [], []
    for i, p in enumerate(patches):
        try:
            e = embed(model, p, args.T, device)
            embs.append(e); ids.append(p)
            if i < 3 or i % 10 == 0:
                print(f"[{i+1}/{len(patches)}] {p}: emb shape={e.shape} norm={np.linalg.norm(e):.3f} "
                      f"min/max={e.min():.3f}/{e.max():.3f}")
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"[skip] {p}: {ex}")
    embs = np.stack(embs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, embeddings=embs, patch_ids=ids)
    print(f"\n[save] {embs.shape} -> {args.out}")

    # ===== 判别力评估 =====
    print("\n===== 嵌入质量评估 =====")
    print(f"embedding维度: {embs.shape[1]}")
    std_per_dim = embs.std(0)
    print(f"逐维std: 均值={std_per_dim.mean():.4f} 最小={std_per_dim.min():.4f} 最大={std_per_dim.max():.4f}")
    dead = (std_per_dim < 1e-4).sum()
    print(f"近似塌缩维度(std<1e-4): {dead}/{embs.shape[1]}")
    # 余弦相似度矩阵
    en = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim = en @ en.T
    off = sim[~np.eye(len(sim), dtype=bool)]
    print(f"两两余弦相似度: 均值={off.mean():.4f} 标准差={off.std():.4f} min={off.min():.4f} max={off.max():.4f}")
    print("(均值远小于1且有方差 => 没有塌缩, embedding有判别力)")


if __name__ == "__main__":
    main()
