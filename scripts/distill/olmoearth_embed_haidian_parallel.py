#!/usr/bin/env python3
"""海淀区 OlmoEarth 嵌入并行提取（8卡NPU）。
同时提取 spatial_tokens (32,32,768) 和 global_emb (768)。
输出:
  /workspace/xuannv/outputs/olmoearth_haidian/{MM}/spatial_tokens.npz  (N,32,32,768)
  /workspace/xuannv/outputs/olmoearth_haidian/{MM}/emb_all.npz         (N,768)
"""
from __future__ import annotations
import sys, os, glob, json, argparse
sys.stdout.reconfigure(line_buffering=True)
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"]:
    os.environ.pop(k, None)
os.environ["HCCL_WHITELIST_DISABLE"] = "1"
sys.path.insert(0, "/workspace/olmoearth_pretrain")

import numpy as np, torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality
import rasterio

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
def extract_both(model, patch, target_month, meta_dict, device):
    """同时提取 spatial_tokens 和 global_emb。"""
    s = build_sample(patch, target_month, meta_dict, device)

    # fast_pass=True for spatial tokens
    out_fast = model(s, patch_size=4, fast_pass=True)
    tokens_attr = out_fast["tokens_and_masks"]
    s2_tokens = tokens_attr.sentinel2_l2a[0]  # (1, T, C, H, W)
    spatial_tok = s2_tokens.mean(dim=(2, 3)).float().cpu().numpy()  # (T, C) -> (32,32,768)?

    # fast_pass=False for global embedding
    out_full = model(s, patch_size=4, fast_pass=False)
    global_emb = out_full["project_aggregated"][0].float().cpu().numpy()

    return spatial_tok, global_emb


def worker(rank, world_size, month, patch_ids, meta_dict):
    import torch_npu
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    # 每卡加载独立模型副本
    model_full = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model_full.eval()
    model = model_full.encoder.to(device)

    # 分配 patches
    my_patches = patch_ids[rank::world_size]
    print(f"[NPU{rank}] 处理 {len(my_patches)} patches")

    spatial_dict = {}
    global_dict = {}

    for i, pid in enumerate(my_patches):
        try:
            sp, gl = extract_both(model, pid, month, meta_dict, device)
            spatial_dict[pid] = sp
            global_dict[pid] = gl
            if (i+1) % 20 == 0 or (i+1) == len(my_patches):
                print(f"  [NPU{rank}] [{i+1}/{len(my_patches)}] {pid}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [NPU{rank}] [skip] {pid}: {e}")

    return spatial_dict, global_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--month', type=int, default=None, help='指定月份，否则处理所有月份')
    args = parser.parse_args()

    meta_list = json.load(open(META))
    meta_dict = {p["patch_id"]: p for p in meta_list}
    patches = [p["patch_id"] for p in meta_list]

    months = [args.month] if args.month else TARGET_MONTHS

    for month in months:
        out_dir = f"{OUT}/{month:02d}"
        os.makedirs(out_dir, exist_ok=True)

        sp_path = f"{out_dir}/spatial_tokens.npz"
        gl_path = f"{out_dir}/emb_all.npz"

        if os.path.exists(sp_path) and os.path.exists(gl_path):
            print(f"[月份{month:02d}] 已有，跳过")
            continue

        print(f"\n===== 月份 {month:02d} ({len(patches)} patches, {args.world_size} 卡并行) =====")

        # 使用 spawn 启动多进程
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.world_size) as pool:
            results = []
            for rank in range(args.world_size):
                r = pool.apply_async(worker, (rank, args.world_size, month, patches, meta_dict))
                results.append(r)

            # 收集结果
            all_spatial = {}
            all_global = {}
            for r in results:
                sp, gl = r.get()
                all_spatial.update(sp)
                all_global.update(gl)

        valid = [p for p in patches if p in all_spatial]
        print(f"  成功: {len(valid)}/{len(patches)}")

        # 保存 spatial tokens
        sp_arr = np.stack([all_spatial[p] for p in valid]).astype(np.float16)
        np.savez(sp_path, tokens=sp_arr, patch_ids=np.array(valid))
        print(f"  ✅ spatial_tokens → {sp_path} shape={sp_arr.shape}")

        # 保存 global embeddings
        gl_arr = np.stack([all_global[p] for p in valid]).astype(np.float32)
        np.savez(gl_path, embeddings=gl_arr, patch_ids=np.array(valid))
        print(f"  ✅ emb_all → {gl_path} shape={gl_arr.shape}")

    print("\n✅ 所有月份完成")


if __name__ == "__main__":
    main()
