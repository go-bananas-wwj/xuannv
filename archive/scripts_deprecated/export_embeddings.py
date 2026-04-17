#!/usr/bin/env python
"""AEF_qwen Embedding 导出脚本.

用法:
    python scripts/export_embeddings.py \
        --config configs/qwen_v1_scenes.yaml \
        --checkpoint /workspace/outputs/aef_qwen_v1/best.pt \
        --output-dir /workspace/outputs/aef_qwen_v1/embeddings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import load_config
    from src.data.dataset import HarbinPatchDataset
    from src.models.model import AEFModel

    cfg = load_config(args.config)
    device = torch.device(args.device)

    # 加载模型
    model = AEFModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 加载数据集
    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    dataset._spatial_augmentation = False
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.experiment.output_dir) / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_patch_ids = []

    print(f"Exporting embeddings for {len(dataset)} patches...")

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
            )

            # 导出 post-L2 embedding (推理模式, 用于变化检测等下游任务)
            emb_map = output.embedding_map  # [B, D, H, W] - L2 normalized
            emb_map_np = emb_map.cpu().numpy()
            all_embeddings.append(emb_map_np)

            # 导出 pre-L2 embedding (原始幅度, 用于 PCA 可视化)
            if output.pre_norm_map is not None:
                pre_norm_map = output.pre_norm_map.cpu().numpy()
                if not hasattr(main, 'all_pre_norm_maps'):
                    main.all_pre_norm_maps = []
                main.all_pre_norm_maps.append(pre_norm_map)

            for pid in batch.get("patch_id", []):
                all_patch_ids.append(pid)

    # 保存
    embeddings_np = np.concatenate(all_embeddings, axis=0)  # [N, D, H, W]
    output_path = output_dir / "embedding_maps.npy"
    np.save(output_path, embeddings_np)
    print(f"Saved embedding maps (L2-normalized): {output_path} shape={embeddings_np.shape}")

    # 保存 pre-L2 embedding maps (原始幅度, 用于 PCA 可视化)
    if hasattr(main, 'all_pre_norm_maps') and main.all_pre_norm_maps:
        pre_norm_maps = np.concatenate(main.all_pre_norm_maps, axis=0)
        output_path_raw = output_dir / "embedding_maps_raw.npy"
        np.save(output_path_raw, pre_norm_maps)
        print(f"Saved raw embedding maps (pre-L2): {output_path_raw} shape={pre_norm_maps.shape}")

    ids_path = output_dir / "patch_ids.json"
    with open(ids_path, "w") as f:
        json.dump(all_patch_ids, f)
    print(f"Saved patch IDs: {ids_path} count={len(all_patch_ids)}")

    # 同时导出推理时的 L2-normalized embedding
    emb_normalized = np.zeros_like(embeddings_np)
    for i in range(len(embeddings_np)):
        for j in range(embeddings_np.shape[2]):
            for k in range(embeddings_np.shape[3]):
                vec = embeddings_np[i, :, j, k]
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    emb_normalized[i, :, j, k] = vec / norm

    output_path_norm = output_dir / "embedding_maps_normalized.npy"
    np.save(output_path_norm, emb_normalized)
    print(f"Saved L2-normalized embeddings: {output_path_norm}")


if __name__ == "__main__":
    main()
