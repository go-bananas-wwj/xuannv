#!/usr/bin/env python3
"""DDP 多卡并行生成 OlmoEarth spatial tokens (fast_pass=True).

用法:
    torchrun --nproc_per_node=8 scripts/distill/generate_olmoearth_tokens_ddp.py \
        --region haidian --months 1-17
    torchrun --nproc_per_node=8 scripts/distill/generate_olmoearth_tokens_ddp.py \
        --region harbin --months 1-17

输出:
    {out_dir}/{MM}/spatial_tokens.npz  shape=(N_patches, 32, 32, 768)
"""
from __future__ import annotations
import sys, os, glob, json, argparse
sys.stdout.reconfigure(line_buffering=True)
for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"]:
    os.environ.pop(k, None)
sys.path.insert(0, "/workspace/olmoearth_pretrain")

import numpy as np, torch
import torch.nn.functional as F
import torch.distributed as dist

# PyTorch 2.1.0 缺少 DeviceMesh / distribute_tensor / FSDP 类，mock 之
import types
mock_dist_tensor = types.ModuleType('torch.distributed.tensor')
mock_dist_tensor.distribute_tensor = lambda tensor, device_mesh: tensor
sys.modules['torch.distributed.tensor'] = mock_dist_tensor

import torch.distributed
if not hasattr(torch.distributed, 'DeviceMesh'):
    class _MockDeviceMesh:
        pass
    torch.distributed.DeviceMesh = _MockDeviceMesh

import torch.distributed.fsdp
if not hasattr(torch.distributed.fsdp, 'fully_shard'):
    def _mock_fully_shard(module, **kwargs):
        return module
    torch.distributed.fsdp.fully_shard = _mock_fully_shard
if not hasattr(torch.distributed.fsdp, 'MixedPrecisionPolicy'):
    class _MockMixedPrecisionPolicy:
        def __init__(self, *args, **kwargs):
            pass
    torch.distributed.fsdp.MixedPrecisionPolicy = _MockMixedPrecisionPolicy
if not hasattr(torch.distributed.fsdp, 'register_fsdp_forward_method'):
    def _mock_register_fsdp_forward_method(module, method_name):
        pass
    torch.distributed.fsdp.register_fsdp_forward_method = _mock_register_fsdp_forward_method

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.data.constants import Modality
import rasterio

# ---------------------------------------------------------------------------
# 区域配置
# ---------------------------------------------------------------------------
REGIONS = {
    "haidian": {
        "raw": "/workspace/raw/haidian_olmoearth",
        "meta": "/workspace/raw/haidian_olmoearth/patches_meta.json",
        "out": "/workspace/outputs/olmoearth_haidian",
        "meta_is_list": True,
    },
    "harbin": {
        "raw": "/workspace/raw/harbin_newarea_olmoearth",
        "meta": "/workspace/xuannv_show/data/harbin/patches_meta.json",
        "out": "/workspace/outputs/olmoearth_harbin",
        "meta_is_list": True,
    },
}

computed   = Normalizer(Strategy.COMPUTED)
predefined = Normalizer(Strategy.PREDEFINED)
H = 128


def setup_ddp():
    if not dist.is_available():
        return 0, 1
    # 检查是否是 torchrun / mp.spawn 环境
    if os.environ.get("RANK") is None or os.environ.get("WORLD_SIZE") is None:
        return 0, 1
    dist.init_process_group(backend="hccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def pick_month(modality, patch, target_month, raw):
    fs = sorted(glob.glob(f"{raw}/{modality}/{patch}/*.tif"))
    fs = [f for f in fs if "static" not in f]
    if not fs:
        return None
    def month_of(f): return int(os.path.basename(f)[4:6])
    fs.sort(key=lambda f: abs(month_of(f) - target_month))
    return fs[0]


def read_resize(path, h=H, w=H, mode="bilinear"):
    with rasterio.open(path) as ds:
        a = ds.read().astype("float32")
    if a.shape[1:] != (h, w):
        t = F.interpolate(torch.from_numpy(a)[None], (h, w),
                          mode=mode, align_corners=False if mode == "bilinear" else None)
        a = t[0].numpy()
    return a


def build_sample(patch, target_month, meta_dict, device, raw):
    arr = {}
    date_str = None
    for m, oe in [("s2", Modality.SENTINEL2_L2A),
                  ("s1", Modality.SENTINEL1),
                  ("landsat", Modality.LANDSAT)]:
        f = pick_month(m, patch, target_month, raw)
        if f is None:
            continue
        a = read_resize(f)
        a = np.transpose(a, (1, 2, 0))[None, :, :, None, :]
        a = computed.normalize(oe, a).astype("float32")
        arr[m] = a
        if date_str is None:
            date_str = os.path.basename(f)[:8]
    if date_str is None:
        date_str = f"2025{target_month:02d}15"
    ts = np.array([[[int(date_str[6:8]), int(date_str[4:6]) - 1, int(date_str[:4])]]])

    def static(m, oe, nearest=False):
        f2 = f"{raw}/{m}/{patch}/static.tif"
        if not os.path.exists(f2):
            return None
        a = read_resize(f2, mode="nearest" if nearest else "bilinear")
        a = np.transpose(a, (1, 2, 0))[None, :, :, None, :]
        return computed.normalize(oe, a).astype("float32")

    b = meta_dict[patch]["bounds_wgs84"]
    ll = predefined.normalize(Modality.LATLON,
         np.array([[(b[1]+b[3])/2, (b[0]+b[2])/2]], dtype="float32")).astype("float32")

    def t(x):
        return torch.from_numpy(x).float().to(device) if x is not None else None

    kw = dict(
        timestamps    = torch.from_numpy(ts).long().to(device),
        sentinel2_l2a = t(arr.get("s2")),
        sentinel1     = t(arr.get("s1")),
        landsat       = t(arr.get("landsat")),
        worldcover    = t(static("worldcover", Modality.WORLDCOVER, nearest=True)),
        srtm          = t(static("dem", Modality.SRTM)),
        latlon        = t(ll),
    )
    for k in ["sentinel2_l2a", "sentinel1", "landsat", "worldcover", "srtm", "latlon"]:
        if kw.get(k) is not None:
            kw[k + "_mask"] = torch.zeros_like(kw[k])
    return MaskedOlmoEarthSample(**kw)


@torch.no_grad()
def extract_spatial_tokens(model, patch, target_month, meta_dict, device, raw):
    """返回 (32, 32, 768) spatial tokens，取 S2 L2A modality 均值。"""
    s = build_sample(patch, target_month, meta_dict, device, raw)
    out = model(s, patch_size=4, fast_pass=True)
    tokens_attr = out["tokens_and_masks"]
    s2_tokens = tokens_attr.sentinel2_l2a
    tokens = s2_tokens[0]  # (32, 32, T, Nv, 768)
    tokens = tokens.mean(dim=(2, 3))  # (32, 32, 768)
    return tokens.float().cpu().numpy()


def get_patches(region_cfg):
    """加载 patch 列表。"""
    meta = json.load(open(region_cfg["meta"]))
    return [p["patch_id"] for p in meta]


def get_meta_dict(region_cfg):
    """构建 patch_id -> meta 字典。"""
    meta = json.load(open(region_cfg["meta"]))
    return {p["patch_id"]: p for p in meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=["haidian", "harbin"])
    parser.add_argument("--months", default="1-17",
                        help="月份范围，如 1-17 表示 2025-01 到 2026-05")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    import torch_npu  # noqa

    rank, world_size = setup_ddp()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)

    region_cfg = REGIONS[args.region]
    raw = region_cfg["raw"]
    out = region_cfg["out"]

    # 解析月份
    if "-" in args.months:
        start, end = map(int, args.months.split("-"))
        all_months = list(range(start, end + 1))
    else:
        all_months = [int(m) for m in args.months.split(",")]

    # 月份映射：1-12 是 2025，13-17 是 2026
    def month_to_ym(m):
        if m <= 12:
            return 2025, m
        else:
            return 2026, m - 12

    # 加载模型
    if rank == 0:
        print(f"[Rank {rank}] 加载 OlmoEarth 模型到 {device}")
    model_full = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model_full.eval()
    model = model_full.encoder.to(device)

    # 加载 meta
    patches = get_patches(region_cfg)
    meta_dict = get_meta_dict(region_cfg)

    if rank == 0:
        print(f"[Rank {rank}] 区域: {args.region}, patches: {len(patches)}, months: {all_months}")

    # DDP 分片：每个 rank 处理一部分 patches
    patches_per_rank = (len(patches) + world_size - 1) // world_size
    start_idx = rank * patches_per_rank
    end_idx = min(start_idx + patches_per_rank, len(patches))
    my_patches = patches[start_idx:end_idx]

    if rank == 0:
        print(f"[Rank {rank}] 本 rank 处理 patches [{start_idx}:{end_idx}] = {len(my_patches)} patches")

    # 为每个缺失月份生成 tokens
    for month_idx in all_months:
        year, month = month_to_ym(month_idx)
        if year == 2025:
            out_dir = f"{out}/{month:02d}"
        else:
            out_dir = f"{out}/{year}/{month:02d}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/spatial_tokens.npz"

        # 检查是否已有（仅 rank 0 检查，避免冲突）
        if args.skip_existing and os.path.exists(out_path):
            if rank == 0:
                print(f"[{args.region}] 月份 {month:02d} ({year}) 已有，跳过")
            continue

        # 每个 rank 生成自己的 partial 文件
        partial_path = f"{out_dir}/spatial_tokens_rank{rank}.npz"
        if os.path.exists(partial_path):
            if rank == 0:
                print(f"[{args.region}] 月份 {month:02d} rank{rank} partial 已有，加载续传")
            try:
                loaded = np.load(partial_path)
                tokens_dict = {str(p): loaded["tokens"][i] for i, p in enumerate(loaded["patch_ids"])}
            except Exception:
                tokens_dict = {}
        else:
            tokens_dict = {}

        if rank == 0:
            print(f"\n===== [{args.region}] {year}-{month:02d} (rank {rank}: {len(my_patches)} patches) =====")

        for i, pid in enumerate(my_patches):
            if pid in tokens_dict:
                continue
            try:
                tok = extract_spatial_tokens(model, pid, month, meta_dict, device, raw)
                tokens_dict[pid] = tok
                if (i + 1) % 20 == 0 or (i + 1) == len(my_patches):
                    print(f"  [Rank {rank}] [{i+1}/{len(my_patches)}] {pid} shape={tok.shape}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [Rank {rank}] [skip] {pid}: {e}")

        # 保存 partial
        valid_patches = [p for p in my_patches if p in tokens_dict]
        if valid_patches:
            arr = np.stack([tokens_dict[p] for p in valid_patches]).astype(np.float16)
            ids = np.array(valid_patches)
            np.savez(partial_path, tokens=arr, patch_ids=ids)
            if rank == 0:
                print(f"  [Rank {rank}] 保存 partial: {len(valid_patches)} patches → {partial_path}")

    # 同步所有 rank
    if dist.is_initialized():
        dist.barrier()

    # Rank 0 合并所有 rank 的 partial 文件
    if rank == 0:
        for month_idx in all_months:
            year, month = month_to_ym(month_idx)
            if year == 2025:
                out_dir = f"{out}/{month:02d}"
            else:
                out_dir = f"{out}/{year}/{month:02d}"
            out_path = f"{out_dir}/spatial_tokens.npz"

            if args.skip_existing and os.path.exists(out_path):
                continue

            # 收集所有 rank 的 partial
            all_tokens = {}
            for r in range(world_size):
                partial_path = f"{out_dir}/spatial_tokens_rank{r}.npz"
                if os.path.exists(partial_path):
                    try:
                        d = np.load(partial_path)
                        for i, p in enumerate(d["patch_ids"]):
                            all_tokens[str(p)] = d["tokens"][i]
                    except Exception as e:
                        print(f"  [合并] 读取 rank{r} partial 失败: {e}")

            if not all_tokens:
                print(f"  [{args.region}] {year}-{month:02d} 无有效 tokens")
                continue

            # 按原始 patches 顺序合并
            valid_patches = [p for p in patches if p in all_tokens]
            arr = np.stack([all_tokens[p] for p in valid_patches]).astype(np.float16)
            ids = np.array(valid_patches)
            np.savez(out_path, tokens=arr, patch_ids=ids)
            print(f"  ✅ [{args.region}] {year}-{month:02d} 合并完成: {len(valid_patches)} patches → {out_path} shape={arr.shape}")

            # 清理 partial 文件
            for r in range(world_size):
                partial_path = f"{out_dir}/spatial_tokens_rank{r}.npz"
                if os.path.exists(partial_path):
                    os.remove(partial_path)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"\n✅ [{args.region}] 所有月份 spatial tokens 提取完成")


if __name__ == "__main__":
    main()
