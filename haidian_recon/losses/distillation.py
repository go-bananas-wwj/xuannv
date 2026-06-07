"""AEF蒸馏损失 — 加载预训练AEF模型."""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class AEFDistiller(nn.Module):
    """
    加载预训练AEF模型，输出64维embedding供蒸馏使用。
    注意：AEF的输入格式与HRE不同，需要做适配。
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "npu") -> None:
        super().__init__()
        from src.config import load_config
        from src.models.model import AEFModel

        self.cfg = load_config(config_path)

        # 先加载raw checkpoint以检测实际的precision_dim
        ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt_raw.get("model_state_dict", ckpt_raw)

        # 自动修正配置维度以匹配checkpoint
        adjustments = {}

        # precision_dim: 从第一个 projection 层检测
        for k in state.keys():
            if "projection.0.weight" in k:
                actual = state[k].shape[0]
                cfg_val = getattr(self.cfg.model, "precision_dim", 128)
                if actual != cfg_val:
                    adjustments["precision_dim"] = (cfg_val, actual)
                    self.cfg.model.precision_dim = actual
                break

        # embedding_dim: 从 bottleneck.to_embedding 检测
        for k in state.keys():
            if "bottleneck.to_embedding.weight" in k:
                actual = state[k].shape[0]
                cfg_val = getattr(self.cfg.model, "embedding_dim", 64)
                if actual != cfg_val:
                    adjustments["embedding_dim"] = (cfg_val, actual)
                    self.cfg.model.embedding_dim = actual
                break

        # space_dim: 从 summary_query 检测
        for k in state.keys():
            if "summary_query.weight" in k:
                actual = state[k].shape[0]
                cfg_val = getattr(self.cfg.model, "space_dim", 256)
                if actual != cfg_val:
                    adjustments["space_dim"] = (cfg_val, actual)
                    self.cfg.model.space_dim = actual
                break

        if adjustments:
            for name, (old, new) in adjustments.items():
                print(f"[AEFDistiller] Auto-adjust {name}: {old} -> {new}")

        self.aef_model = AEFModel(self.cfg)
        self.aef_model.load_state_dict(state, strict=False)
        self.aef_model.eval()

        # 冻结
        for p in self.aef_model.parameters():
            p.requires_grad = False

        self.device = device
        self.aef_model.to(device)

        # AEF的输入源映射
        self.aef_input_sources = getattr(self.cfg.data, "input_sources", ["s2", "s1", "landsat"])

    def forward(self, batch: dict) -> torch.Tensor | None:
        """
        将HRE格式的batch转换为AEF格式，输出embedding。

        Args:
            batch: HRE格式的batch {source_name: [B, T, C, H, W]}

        Returns:
            embedding: [B, 64] 或 None（若无法适配）
        """
        B = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                B = v.shape[0]
                break
        if B is None:
            return None

        # 构建AEF输入
        source_frames = []
        source_timestamps = []
        source_type_ids = []
        T_total = 0

        for src_name in self.aef_input_sources:
            if src_name == "s1" and batch.get("tianyi_sar") is not None:
                x = batch["tianyi_sar"]
            else:
                x = batch.get(src_name)

            if x is None:
                continue

            B_src, T, C, H, W = x.shape
            source_frames.append(x)
            num_source_types = getattr(self.cfg.model, "num_source_types", 8)
            type_id = self._get_source_type_id(src_name, num_source_types)
            source_type_ids.extend([type_id] * T)
            # 简单时间戳：使用0
            timestamps = torch.zeros(B_src, T, device=x.device)
            source_timestamps.append(timestamps)
            T_total += T

        if len(source_frames) == 0:
            return None

        # 拼接
        max_c = max(f.shape[2] for f in source_frames)
        padded_frames = []
        for f in source_frames:
            if f.shape[2] < max_c:
                pad = torch.zeros(f.shape[0], f.shape[1], max_c - f.shape[2], f.shape[3], f.shape[4],
                                  device=f.device, dtype=f.dtype)
                f = torch.cat([f, pad], dim=2)
            padded_frames.append(f)

        aef_frames = torch.cat(padded_frames, dim=1)  # [B, T_total, C_max, H, W]
        aef_timestamps = torch.cat(source_timestamps, dim=1)  # [B, T_total]

        # 构建mask (NPU bitwise_and 只支持 bool/int，不能用 float)
        source_frame_mask = torch.ones(B, T_total, device=aef_frames.device, dtype=torch.bool)
        source_input_mask = torch.ones(B, T_total, device=aef_frames.device, dtype=torch.bool)
        source_type_ids_tensor = torch.tensor(source_type_ids, device=aef_frames.device).unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            output = self.aef_model(
                source_frames=aef_frames,
                source_timestamps_ms=aef_timestamps,
                source_frame_mask=source_frame_mask,
                source_input_mask=source_input_mask,
                source_type_ids=source_type_ids_tensor,
                valid_start_ms=torch.zeros(B, device=aef_frames.device),
                valid_end_ms=torch.ones(B, device=aef_frames.device) * 1e12,
                target_relative_time=torch.zeros(B, 1, device=aef_frames.device),
                target_metadata=torch.zeros(B, 4, device=aef_frames.device),
                skip_decoder=True,
            )

        return output.embedding  # [B, 64]

    def _get_source_type_id(self, src_name: str, num_source_types: int = 8) -> int:
        """获取AEF的source type id，避免越界."""
        type_map = {"s2": 0, "s1": 1, "landsat": 2, "tianyi_sar": 7}
        raw_id = type_map.get(src_name, 0)
        return min(raw_id, num_source_types - 1)


def aef_distillation_loss(hre_embedding: torch.Tensor, aef_embedding: torch.Tensor | None) -> torch.Tensor:
    """
    AEF蒸馏损失。

    Args:
        hre_embedding: [B, 64]
        aef_embedding: [B, 64] 或 None

    Returns:
        loss: scalar tensor
    """
    if aef_embedding is None:
        return hre_embedding.new_tensor(0.0)

    hre_norm = F.normalize(hre_embedding, p=2, dim=-1)
    aef_norm = F.normalize(aef_embedding, p=2, dim=-1)
    return F.mse_loss(hre_norm, aef_norm)
