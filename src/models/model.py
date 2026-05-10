"""AEF_qwen 主模型 — 3类输入, 7类重建目标.

输入 (Input): S2, S1, Landsat (仅带时间戳的图像帧)
目标 (Target): 输入3类 + DEM + WorldCover + Dynamic World + JRC Water = 7类
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as _checkpoint

from src.models.bottleneck import VMFBottleneck
from src.models.blocks import STPBlock
from src.models.decoders import ContinuousDecoder, CategoricalDecoder
from src.models.sensor_encoders import SensorEncoderBank
from src.models.time_encoding import TimeCodeEncoder, WindowCodeEncoder, RelativeTimeCodeEncoder


class CosineClassificationHead(nn.Module):
    """余弦相似度分类头."""

    def __init__(self, embedding_dim: int, num_classes: int, temperature: float = 10.0) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.temperature = temperature

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(embedding, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return normed @ proto.T * self.temperature


@dataclass(slots=True)
class AEFOutput:
    embedding_map: torch.Tensor          # [B, D, H, W]
    embedding: torch.Tensor              # [B, D]
    reconstructions: torch.Tensor        # [B, T_tgt, C, H, W]
    logits: torch.Tensor                 # [B, num_classes]
    pre_norm_embedding: torch.Tensor     # [B, D]  L2 norm 前
    pre_norm_map: torch.Tensor | None = None  # [B, D, H, W]  L2 norm 前空间 embedding
    aux_logits: torch.Tensor | None = None
    summary_pooled: torch.Tensor | None = None
    bottleneck_logits: torch.Tensor | None = None


class AEFModel(nn.Module):
    """AEF 主模型 — 3类输入, 7类目标."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = cfg.data

        # 传感器编码器 (仅 3 类输入)
        # 推导 input_sources (对齐 dataset.py 逻辑)
        input_sources = getattr(d, "input_sources", None)
        if input_sources is None:
            from src.data.transforms import INPUT_SOURCES
            input_sources = INPUT_SOURCES[:d.num_input_sources]

        self.sensor_encoder_bank = SensorEncoderBank(
            num_sensor_types=m.num_sensor_types,
            input_dim=m.input_dim,
            stem_dim=m.stem_dim,
            out_dim=m.precision_dim,
            source_channels=getattr(m, "source_channels", None),
            stem_channels=getattr(m, "stem_channels", None),
            input_sources=input_sources,
        )
        self.time_encoder = TimeCodeEncoder(m.time_code_dim)
        self.window_encoder = WindowCodeEncoder(m.window_code_dim)
        self.relative_time_encoder = RelativeTimeCodeEncoder(m.relative_time_code_dim)

        # STP blocks (前 N 层禁用 Space)
        n_space_disabled = getattr(m, 'num_blocks_disable_space', 0)
        self.stp_blocks = nn.ModuleList([
            STPBlock(
                channels=m.precision_dim,
                num_heads=m.num_heads,
                time_code_dim=m.time_code_dim,
                use_space=(i >= n_space_disabled),
            )
            for i in range(m.num_blocks)
        ])
        self.use_checkpoint = m.gradient_checkpointing

        # 时间条件摘要
        self.summary_query = nn.Linear(m.window_code_dim, m.precision_dim)

        # Bottleneck (训练 skip L2, 推理 L2)
        self.bottleneck = VMFBottleneck(
            channels=m.precision_dim,
            embedding_dim=m.embedding_dim,
            kappa=m.vmf_kappa,
            skip_l2_training=m.skip_l2_norm_training,
        )

        # 目标解码器
        from src.data.transforms import TARGET_SOURCES
        num_tgt = d.num_target_sources
        target_sources = getattr(d, "target_sources", None)
        if target_sources is not None:
            tgt_list = [(t["name"], t["loss_type"], t["sensor_src"]) for t in target_sources]
        else:
            tgt_list = TARGET_SOURCES[:num_tgt]
        self.per_source_decoders = nn.ModuleList()
        self._per_source_out_channels: list[int] = []
        for t_idx, (tgt_name, loss_type, sensor_src) in enumerate(tgt_list):
            if loss_type == 1:
                # 分类目标
                dec = CategoricalDecoder(
                    m.embedding_dim, m.window_code_dim, m.relative_time_code_dim,
                    m.metadata_dim, out_channels=d.num_classes,
                )
                self._per_source_out_channels.append(d.num_classes)
            else:
                # 连续目标
                dec = ContinuousDecoder(
                    m.embedding_dim, m.window_code_dim, m.relative_time_code_dim,
                    m.metadata_dim, out_channels=m.reconstruction_channels,
                )
                self._per_source_out_channels.append(m.reconstruction_channels)
            self.per_source_decoders.append(dec)

        # 分类头
        self.classification_head = CosineClassificationHead(m.embedding_dim, d.num_classes)
        self.aux_cls_head = CosineClassificationHead(m.precision_dim, d.num_classes)
        self.bottleneck_cls_head = CosineClassificationHead(m.embedding_dim, d.num_classes)

    def encode_frames(
        self,
        source_frames: torch.Tensor,
        source_timestamps_ms: torch.Tensor,
        source_frame_mask: torch.Tensor,
        source_input_mask: torch.Tensor,
        source_type_ids: torch.Tensor,
        valid_start_ms: torch.Tensor,
        valid_end_ms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码输入帧 → summary_map, window_code, attention."""
        # 兼容单源 [B,T,C,H,W] 或多源 [B,S,T,C,H,W]
        if source_frames.dim() == 5:
            source_frames = source_frames[:, None, ...]
            source_timestamps_ms = source_timestamps_ms[:, None, ...]
            source_frame_mask = source_frame_mask[:, None, ...]
            source_input_mask = source_input_mask[None, :] if source_input_mask.dim() == 1 else source_input_mask
            source_type_ids = source_type_ids[None, :] if source_type_ids.dim() == 1 else source_type_ids

        # 传感器编码
        encoded = self.sensor_encoder_bank(source_frames, source_type_ids)

        # 展平
        B, S, T, C, H, W = encoded.shape
        frames = encoded.reshape(B, S * T, C, H, W)
        timestamps = source_timestamps_ms.reshape(B, S * T).float()  # 转 float32 适配 AMP

        # ★ 关键修复: 根据 valid_start_ms / valid_end_ms 过滤帧,
        # 确保 attention 只能看到请求时间窗口内的帧
        valid_start_f = valid_start_ms.float().view(-1)
        valid_end_f = valid_end_ms.float().view(-1)
        window_mask = (
            (source_timestamps_ms.float() >= valid_start_f[:, None, None])
            & (source_timestamps_ms.float() <= valid_end_f[:, None, None])
        )
        effective_frame_mask = source_frame_mask & window_mask
        mask = (effective_frame_mask & source_input_mask[:, :, None]).reshape(B, S * T)

        # 安全回退: 如果某个 batch 项在窗口内没有任何帧,
        # 则回退到使用所有有效帧, 避免 attention softmax 产生 NaN
        mask_all_false = ~mask.any(dim=1)
        if mask_all_false.any():
            fallback_mask = (source_frame_mask & source_input_mask[:, :, None]).reshape(B, S * T)
            mask = torch.where(mask_all_false[:, None], fallback_mask, mask)

        # 时间编码
        time_codes = self.time_encoder(timestamps)
        # cast to float32 for AMP, ensure scalar or [B]
        valid_start = valid_start_f if valid_start_ms.dtype != torch.float32 else valid_start_ms.view(-1)
        valid_end = valid_end_f if valid_end_ms.dtype != torch.float32 else valid_end_ms.view(-1)
        window_code = self.window_encoder(valid_start, valid_end)

        # STP 主干
        x = frames
        for block in self.stp_blocks:
            if self.use_checkpoint and self.training:
                x = _checkpoint.checkpoint(block, x, time_codes, mask, use_reentrant=False)
            else:
                x = block(x, time_codes, mask)

        # 时间条件摘要
        pooled = x.mean(dim=(-2, -1))  # [B, T, C]
        query = self.summary_query(window_code)[:, None, :]
        attn_scores = torch.sum(query * pooled, dim=-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(attn_scores, dim=-1)
        summary = torch.einsum("bt,btchw->bchw", attn, x)

        # Dummy 使用 time_codes，确保 time_encoder 参数参与梯度流
        summary = summary + time_codes.sum() * 0.0

        return summary, window_code, attn

    def forward(
        self,
        source_frames: torch.Tensor,
        source_timestamps_ms: torch.Tensor,
        source_frame_mask: torch.Tensor,
        source_input_mask: torch.Tensor,
        source_type_ids: torch.Tensor,
        valid_start_ms: torch.Tensor,
        valid_end_ms: torch.Tensor,
        target_relative_time: torch.Tensor,
        target_metadata: torch.Tensor,
        target_loss_type: torch.Tensor | None = None,
        target_source_idx: torch.Tensor | None = None,
    ) -> AEFOutput:
        if source_type_ids is None:
            if source_frames.dim() == 6:
                source_type_ids = torch.zeros(source_frames.shape[:2], dtype=torch.long, device=source_frames.device)
            else:
                source_type_ids = torch.zeros(source_frames.shape[0], dtype=torch.long, device=source_frames.device)

        # 编码
        summary_map, window_code, _ = self.encode_frames(
            source_frames, source_timestamps_ms, source_frame_mask,
            source_input_mask, source_type_ids, valid_start_ms, valid_end_ms,
        )

        # 瓶颈
        embedding_map, embedding, pre_norm, pre_norm_map = self.bottleneck(summary_map)

        # Dummy 激活所有 sensor encoder 参数（避免 DDP 未使用参数报错）
        dummy_sensor = torch.tensor(0.0, device=source_frames.device)
        for encoder in self.sensor_encoder_bank.encoders.values():
            for p in encoder.parameters():
                dummy_sensor = dummy_sensor + p.sum() * 0.0
        summary_map = summary_map + dummy_sensor

        # 解码
        B = summary_map.shape[0]
        if target_relative_time.dim() == 1:
            target_relative_time = target_relative_time[:, None]
        if target_metadata.dim() == 2:
            target_metadata = target_metadata[:, None, :]

        T_tgt = target_relative_time.shape[1]
        # Flatten for encoder then reshape back
        flat_rel_time = target_relative_time.reshape(B * T_tgt, 1)
        relative_codes = self.relative_time_encoder(flat_rel_time)
        expanded_map = embedding_map[:, None, ...].repeat(1, T_tgt, 1, 1, 1)
        expanded_window = window_code[:, None, :].expand(-1, T_tgt, -1)

        flat_map = expanded_map.reshape(B * T_tgt, *embedding_map.shape[1:])
        flat_window = expanded_window.reshape(B * T_tgt, window_code.shape[-1])
        flat_relative = relative_codes  # already [B*T_tgt, code_dim]
        flat_metadata = target_metadata.reshape(B * T_tgt, target_metadata.shape[-1])

        # 7 类目标分别解码
        # V11: 静态目标 (dem=3, worldcover=4, jrc_water=6) 弱化时间编码
        STATIC_SOURCE_IDS = {3, 4, 6}
        flat_relative_modified = flat_relative.clone()
        if target_source_idx is not None:
            flat_src_idx = target_source_idx.reshape(B * T_tgt)
            static_mask = torch.zeros(B * T_tgt, dtype=torch.bool, device=flat_map.device)
            for sid in STATIC_SOURCE_IDS:
                static_mask |= (flat_src_idx == sid)
            if static_mask.any():
                flat_relative_modified[static_mask] = flat_relative_modified[static_mask] * 0.1

        max_ch = max(self._per_source_out_channels)
        flat_recon = torch.zeros(B * T_tgt, max_ch, *flat_map.shape[2:], device=flat_map.device, dtype=flat_map.dtype)

        dummy_dec = torch.tensor(0.0, device=flat_map.device)
        if target_source_idx is not None:
            flat_src_idx = target_source_idx.reshape(B * T_tgt)
            for src_id, dec in enumerate(self.per_source_decoders):
                mask = (flat_src_idx == src_id)
                if mask.any():
                    out = dec(flat_map[mask], flat_window[mask], flat_relative_modified[mask], flat_metadata[mask])
                    out_ch = self._per_source_out_channels[src_id]
                    flat_recon[mask, :out_ch] = out.to(flat_recon.dtype)
                # dummy loss to ensure decoder params always have gradients
                for p in dec.parameters():
                    dummy_dec = dummy_dec + p.sum() * 0.0
        else:
            # 默认: 全部用第一个 decoder
            out = self.per_source_decoders[0](flat_map, flat_window, flat_relative_modified, flat_metadata)
            flat_recon[:, :self._per_source_out_channels[0]] = out.to(flat_recon.dtype)
            for p in self.per_source_decoders[0].parameters():
                dummy_dec = dummy_dec + p.sum() * 0.0

        flat_recon = flat_recon + dummy_dec
        reconstructions = flat_recon.reshape(B, T_tgt, *flat_recon.shape[1:])

        # 分类头
        logits = self.classification_head(embedding)
        summary_pooled = summary_map.mean(dim=(-2, -1))
        aux_logits = self.aux_cls_head(summary_pooled)
        bottleneck_logits = self.bottleneck_cls_head(pre_norm)

        return AEFOutput(
            embedding_map=embedding_map,
            embedding=embedding,
            reconstructions=reconstructions,
            logits=logits,
            pre_norm_embedding=pre_norm,
            pre_norm_map=pre_norm_map,
            aux_logits=aux_logits,
            summary_pooled=summary_pooled,
            bottleneck_logits=bottleneck_logits,
        )

    def encode_dual_window(
        self,
        source_frames: torch.Tensor,
        source_timestamps_ms: torch.Tensor,
        source_frame_mask: torch.Tensor,
        source_input_mask: torch.Tensor,
        source_type_ids: torch.Tensor,
        valid_start_w1: torch.Tensor,
        valid_end_w1: torch.Tensor,
        valid_start_w2: torch.Tensor,
        valid_end_w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """对同一 batch 用两个不同 valid_period 分别编码.

        返回:
            emb_w1: [B, D, H, W] L2-normalized (window 1)
            emb_w2: [B, D, H, W] L2-normalized (window 2)
            pre_w1: [B, D, H, W] pre-norm (window 1) — 用于 temporal loss
            pre_w2: [B, D, H, W] pre-norm (window 2) — 用于 temporal loss
        """
        B = source_frames.shape[0] if source_frames.dim() > 5 else source_frames.shape[1]
        dev = source_frames.device
        num_tgt = self.cfg.data.num_target_sources
        meta_dim = getattr(self.cfg.data, "metadata_dim", 4)

        # Window 1
        out1 = self.forward(
            source_frames=source_frames,
            source_timestamps_ms=source_timestamps_ms,
            source_frame_mask=source_frame_mask,
            source_input_mask=source_input_mask,
            source_type_ids=source_type_ids,
            valid_start_ms=valid_start_w1,
            valid_end_ms=valid_end_w1,
            target_relative_time=torch.zeros(B, num_tgt, device=dev),
            target_metadata=torch.zeros(B, num_tgt, meta_dim, device=dev),
        )
        emb_w1 = F.normalize(out1.embedding_map, p=2, dim=1)
        pre_w1 = out1.pre_norm_map if out1.pre_norm_map is not None else out1.embedding_map

        # Window 2
        out2 = self.forward(
            source_frames=source_frames,
            source_timestamps_ms=source_timestamps_ms,
            source_frame_mask=source_frame_mask,
            source_input_mask=source_input_mask,
            source_type_ids=source_type_ids,
            valid_start_ms=valid_start_w2,
            valid_end_ms=valid_end_w2,
            target_relative_time=torch.zeros(B, num_tgt, device=dev),
            target_metadata=torch.zeros(B, num_tgt, meta_dim, device=dev),
        )
        emb_w2 = F.normalize(out2.embedding_map, p=2, dim=1)
        pre_w2 = out2.pre_norm_map if out2.pre_norm_map is not None else out2.embedding_map

        return emb_w1, emb_w2, pre_w1, pre_w2

    def encode_dual_window_v10(
        self,
        source_frames: torch.Tensor,
        source_timestamps_ms: torch.Tensor,
        source_frame_mask: torch.Tensor,
        source_input_mask: torch.Tensor,
        source_type_ids: torch.Tensor,
        valid_start_w1: torch.Tensor,
        valid_end_w1: torch.Tensor,
        valid_start_w2: torch.Tensor,
        valid_end_w2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """V10: 双窗口编码 — 显式 Difference Module.

        Returns:
            emb_w1: [B, D, H, W] L2-normalized (window 1)
            emb_w2: [B, D, H, W] L2-normalized (window 2)
            pre_w1: [B, D, H, W] pre-norm (window 1)
            pre_w2: [B, D, H, W] pre-norm (window 2)
            change_score: [B, 1, H, W] 变化概率图 (0~1)
            diff_feat: [B, D/2, H, W] 差分特征
        """
        # 编码两个窗口到 summary_map (共用 encoder，避免重复计算)
        summary_w1, _, _ = self.encode_frames(
            source_frames, source_timestamps_ms, source_frame_mask,
            source_input_mask, source_type_ids, valid_start_w1, valid_end_w1,
        )
        summary_w2, _, _ = self.encode_frames(
            source_frames, source_timestamps_ms, source_frame_mask,
            source_input_mask, source_type_ids, valid_start_w2, valid_end_w2,
        )

        # V10: bottleneck 显式差分
        emb_w1, emb_w2, pre_w1, pre_w2, change_score, diff_feat = self.bottleneck.forward_dual_window(
            summary_w1, summary_w2
        )

        # 推理时额外 L2 normalize
        if not (self.training and self.bottleneck.skip_l2_training):
            emb_w1 = F.normalize(emb_w1, p=2, dim=1)
            emb_w2 = F.normalize(emb_w2, p=2, dim=1)

        return emb_w1, emb_w2, pre_w1, pre_w2, change_score, diff_feat
