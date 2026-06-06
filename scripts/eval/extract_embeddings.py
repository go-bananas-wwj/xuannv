#!/usr/bin/env python
"""Embedding 批量提取 — 统一入口。

为所有 patch × 所有月份提取 embedding，支持分片并行。

输出格式:
  npy   每个 patch×月份 输出一个 {patch_id}_{YYYY-MM}.npy
  npz   所有数据合并为一个 patch_embeddings.npz（供 knn_eval.py 使用）

用法:
    # 单设备提取（npz 格式，供下游评估）
    python extract_embeddings.py --config configs/config.yaml \\
        --checkpoint epoch_40.pt --output-dir evaluation/results/embeddings/

    # 7 卡并行（每张卡跑一段，npy 格式）
    ASCEND_RT_VISIBLE_DEVICES=0 python extract_embeddings.py \\
        --config configs/config.yaml --checkpoint epoch_40.pt \\
        --output-dir evaluation/results/embeddings/ --gpu-idx 0 --total-gpus 7 --format npy

    # 合并 npy 结果为 npz（所有分片完成后）
    python extract_embeddings.py --merge-npy-dir out/embeddings/ \\
        --output-dir evaluation/results/ --merge-only
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import calendar
import time as time_mod
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
from tqdm import tqdm

# 默认月份列表：2025 年目标月份（对应变化检测标注）
DEFAULT_MONTHS = [
    (2025, 4), (2025, 5), (2025, 6), (2025, 7),
    (2025, 8), (2025, 9), (2025, 10),
]


# ── 时间窗口 ─────────────────────────────────────────────────────────────────

def month_to_window(year: int, month: int) -> tuple[int, int]:
    """年月 → (valid_start_ms, valid_end_ms)。"""
    start_s = int(time_mod.mktime((year, month, 1, 0, 0, 0, 0, 0, 0)))
    last_d  = calendar.monthrange(year, month)[1]
    end_s   = int(time_mod.mktime((year, month, last_d, 23, 59, 59, 0, 0, 0)))
    return start_s * 1000, end_s * 1000


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def load_model_and_dataset(config_path, ckpt_path, device_str):
    from src.config import load_config
    from src.models.model import AEFModel
    from src.data.dataset import HarbinPatchDataset
    from src.data.multi_region_dataset import MultiRegionPatchDataset

    cfg = load_config(config_path)
    cfg.data.preload = False

    if getattr(cfg.data, 'multi_region_manifest', None):
        dataset = MultiRegionPatchDataset(cfg)
    else:
        dataset = HarbinPatchDataset(cfg=cfg)
    device  = torch.device(device_str)

    model = AEFModel(cfg=cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd    = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # patch × month → dataset index
    sample_map = {
        (pid, y, m): idx
        for idx, (pid, y, m) in enumerate(dataset.monthly_samples)
    }
    return model, dataset, cfg, device, sample_map


# ── 单张 embedding 提取 ────────────────────────────────────────────────────

def _center_crop_spatial(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """对空间特征图做中心裁剪. x: [..., H, W], ratio in (0, 1]."""
    if ratio >= 1.0:
        return x
    *leading, H, W = x.shape
    new_h = int(H * ratio)
    new_w = int(W * ratio)
    top = (H - new_h) // 2
    left = (W - new_w) // 2
    return x[..., top:top + new_h, left:left + new_w]


@torch.no_grad()
def extract_one(model, dataset, cfg, sample_map, device,
                patch_id, year, month, center_crop_ratio: float = 1.0) -> np.ndarray | None:
    key = (patch_id, year, month)
    if key not in sample_map:
        return None

    item   = dataset[sample_map[key]]
    vs, ve = month_to_window(year, month)

    def _to(x):
        return x.unsqueeze(0).to(device)

    use_bf16 = getattr(cfg.training, 'use_bf16', True)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=use_bf16):
        out = model(
            source_frames        = _to(item["source_frames"]),
            source_timestamps_ms = _to(item["source_timestamps_ms"]),
            source_frame_mask    = _to(item["source_frame_mask"]),
            source_input_mask    = _to(item["source_input_mask"]),
            source_type_ids      = _to(item["source_type_ids"]),
            valid_start_ms       = torch.tensor([vs], dtype=torch.int64, device=device),
            valid_end_ms         = torch.tensor([ve], dtype=torch.int64, device=device),
            target_relative_time = torch.zeros(1, cfg.data.num_target_sources, device=device),
            target_metadata      = torch.zeros(1, cfg.data.num_target_sources,
                                               cfg.data.metadata_dim, device=device),
            skip_decoder=True,
        )
    emb = F.normalize(out.embedding_map.float(), p=2, dim=1)  # [1, D, H, W]
    emb = _center_crop_spatial(emb, center_crop_ratio)
    return emb.squeeze(0).cpu().numpy()                 # [D, H, W]


# ── 主提取逻辑 ───────────────────────────────────────────────────────────────

def extract_all(args):
    model, dataset, cfg, device, sample_map = load_model_and_dataset(
        args.config, args.checkpoint, args.device
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_patches = dataset.patches
    total = len(all_patches)

    # 分片
    chunk   = (total + args.total_gpus - 1) // args.total_gpus
    start   = args.gpu_idx * chunk
    end     = min(start + chunk, total)
    patches = all_patches[start:end]

    months = DEFAULT_MONTHS
    crop_ratio = getattr(args, 'center_crop_ratio', 1.0)
    print(f"[提取] GPU 分片 {args.gpu_idx}/{args.total_gpus} → "
          f"patch {start}:{end} ({len(patches)} 个), {len(months)} 个月"
          f"{' (center_crop=' + str(crop_ratio) + ')' if crop_ratio < 1.0 else ''}")

    if args.format == "npy":
        _extract_npy(model, dataset, cfg, sample_map, device,
                     patches, months, output_dir, args.gpu_idx, crop_ratio)
    else:
        _extract_npz(model, dataset, cfg, sample_map, device,
                     patches, months, output_dir, args.gpu_idx, crop_ratio)


def _extract_npy(model, dataset, cfg, sample_map, device,
                 patches, months, output_dir, gpu_idx, center_crop_ratio: float = 1.0):
    extracted = skipped = errors = 0
    records = []

    for pid in tqdm(patches, desc=f"GPU{gpu_idx}"):
        for year, month in months:
            month_str = f"{year}-{month:02d}"
            out_fp = output_dir / f"{pid}_{month_str}.npy"
            if out_fp.exists():
                skipped += 1
                continue
            emb = extract_one(model, dataset, cfg, sample_map,
                               device, pid, year, month, center_crop_ratio)
            if emb is None:
                continue
            np.save(out_fp, emb)
            records.append({"patch_id": pid, "month": month_str, "shape": list(emb.shape)})
            extracted += 1

    meta_fp = output_dir / f"meta_shard{gpu_idx}.json"
    meta_fp.write_text(json.dumps(records, indent=2))
    print(f"\n[提取] 完成 skipped={skipped} extracted={extracted} errors={errors}")
    print(f"       元数据: {meta_fp}")


def _extract_npz(model, dataset, cfg, sample_map, device,
                 patches, months, output_dir, gpu_idx, center_crop_ratio: float = 1.0):
    """提取并合并为 npz，格式与 knn_eval.py --embedding-file 兼容。

    spatial_maps: [N_patches, N_months, D, H, W]
    patch_ids:    [N_patches] str
    month_labels: [N_months] str
    """
    n_patches = len(patches)
    n_months  = len(months)
    all_maps  = None

    for m_idx, (year, month) in enumerate(months):
        month_str = f"{year}-{month:02d}"
        print(f"  [{m_idx+1}/{n_months}] {month_str}...")
        for p_idx, pid in enumerate(tqdm(patches, desc=f"  {month_str}", leave=False)):
            emb = extract_one(model, dataset, cfg, sample_map, device, pid, year, month, center_crop_ratio)
            if emb is None:
                continue
            if all_maps is None:
                D, H, W = emb.shape
                all_maps = np.zeros((n_patches, n_months, D, H, W), dtype=np.float32)
            all_maps[p_idx, m_idx] = emb

    if all_maps is None:
        print("[警告] 没有提取到任何数据")
        return

    month_labels = [f"{y}-{m:02d}" for y, m in months]
    out_fp = output_dir / f"patch_embeddings_shard{gpu_idx}.npz"
    np.savez_compressed(out_fp, spatial_maps=all_maps,
                        patch_ids=np.array(patches), month_labels=np.array(month_labels))
    print(f"[提取] 保存 → {out_fp}  shape={all_maps.shape}")


# ── 合并多分片 npy → npz ──────────────────────────────────────────────────────

def merge_npy(args):
    """将并行分片生成的 .npy 文件合并为一个 patch_embeddings.npz。"""
    npy_dir = Path(args.merge_npy_dir)
    npy_files = sorted(npy_dir.glob("*.npy"))
    if not npy_files:
        print(f"[合并] 未找到 .npy 文件于 {npy_dir}")
        return

    # 解析 patch_id 与 month
    records: dict[str, dict[str, np.ndarray]] = {}
    for fp in tqdm(npy_files, desc="加载 npy"):
        name = fp.stem          # {patch_id}_{YYYY-MM}
        month_str = name[-7:]   # YYYY-MM
        patch_id  = name[:-8]   # everything before _{YYYY-MM}
        emb = np.load(fp)
        if patch_id not in records:
            records[patch_id] = {}
        records[patch_id][month_str] = emb

    patch_ids   = sorted(records.keys())
    all_months  = sorted({m for pm in records.values() for m in pm})
    D, H, W     = next(iter(next(iter(records.values())).values())).shape
    spatial_maps = np.zeros((len(patch_ids), len(all_months), D, H, W), dtype=np.float32)

    for p_idx, pid in enumerate(patch_ids):
        for m_idx, m in enumerate(all_months):
            if m in records[pid]:
                spatial_maps[p_idx, m_idx] = records[pid][m]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "patch_embeddings.npz"
    np.savez_compressed(out_fp, spatial_maps=spatial_maps,
                        patch_ids=np.array(patch_ids),
                        month_labels=np.array(all_months))
    print(f"[合并] 完成 → {out_fp}")
    print(f"       shape={spatial_maps.shape}  patches={len(patch_ids)}  months={all_months}")


# ── 参数解析 & 入口 ──────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(description="Embedding 批量提取")
    pa.add_argument("--config",       default="configs/config.yaml")
    pa.add_argument("--checkpoint",   default="")
    pa.add_argument("--output-dir",   default="outputs/embeddings/")
    pa.add_argument("--device",       default="npu:0")
    pa.add_argument("--format",       default="npz", choices=["npy", "npz"],
                    help="npy: 每个 patch×月份 独立文件；npz: 合并为单文件（供 knn_eval）")
    pa.add_argument("--gpu-idx",      type=int, default=0,  help="分片索引（0-based）")
    pa.add_argument("--total-gpus",   type=int, default=1,  help="总分片数")
    pa.add_argument("--merge-only",   action="store_true",  help="仅执行合并，不提取")
    pa.add_argument("--merge-npy-dir", default="",          help="--merge-only 时的 npy 目录")
    pa.add_argument("--center-crop-ratio", type=float, default=1.0,
                    help="中心裁剪比例（默认 1.0 不裁剪，0.75 保留中心 75% 区域）")
    args = pa.parse_args()

    if args.merge_only:
        if not args.merge_npy_dir:
            pa.error("--merge-only 需要指定 --merge-npy-dir")
        merge_npy(args)
    else:
        if not args.checkpoint:
            pa.error("--checkpoint 是必需参数")
        extract_all(args)


if __name__ == "__main__":
    main()
