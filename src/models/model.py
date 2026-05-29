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
from src.models.blocks import STPEncoder
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
    dual_pre_w2: torch.Tensor | None = None  # [B, D, H, W] 第二窗口 pre_norm (用于 temporal loss)
    patch_id_logits: torch.Tensor | None = None  # [B*H*W, num_patches] 空间token级实例判别logits
    patch_id_spatial_hw: int = 1  # H*W 空间token数，供 trainer 扩展 labels


class AEFModel(nn.Module):
    """AEF 主模型 — 3类输入, 7类目标."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = cfg.data
        self.image_size = getattr(d, 'image_size', 128)

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
        self.time_to_summary = nn.Linear(m.time_code_dim, m.precision_dim)
        self.window_encoder = WindowCodeEncoder(m.window_code_dim)
        self.relative_time_encoder = RelativeTimeCodeEncoder(m.relative_time_code_dim)

        # STP Encoder — 玄女V2: 严格对齐论文的三路径独立 + 跨尺度交换
        self.stp_encoder = STPEncoder(
            precision_dim=m.precision_dim,
            time_dim=m.time_dim,
            space_dim=m.space_dim,
            num_blocks=m.num_blocks,
            num_heads=m.num_heads,
            use_checkpoint=m.gradient_checkpointing,
        )

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
            # 从 target_sources 配置中读取 out_channels，支持 per-source
            out_ch = m.reconstruction_channels
            if target_sources is not None and t_idx < len(target_sources):
                out_ch = target_sources[t_idx].get("out_channels", m.reconstruction_channels)
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
                    m.metadata_dim, out_channels=out_ch,
                )
                self._per_source_out_channels.append(out_ch)
            self.per_source_decoders.append(dec)

        # 分类头
        self.classification_head = CosineClassificationHead(m.embedding_dim, d.num_classes)
        self.aux_cls_head = CosineClassificationHead(m.precision_dim, d.num_classes)
        self.bottleneck_cls_head = CosineClassificationHead(m.embedding_dim, d.num_classes)

        # ★ 实例判别头: 预测 patch 身份 (0 ~ num_patches-1)，迫使全局 embedding 多样化
        t = cfg.training
        patch_id_num = getattr(t, 'patch_id_num_patches', 0)
        if patch_id_num > 0:
            self.patch_id_head: nn.Linear | None = nn.Linear(m.embedding_dim, patch_id_num)
        else:
            self.patch_id_head = None

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
        # 终极回退: 如果仍然全 False（所有 source 都被 drop），至少保留第一帧
        mask_all_false = ~mask.any(dim=1)
        if mask_all_false.any():
            mask[mask_all_false, 0] = True

        # 时间编码
        time_codes = self.time_encoder(timestamps)
        # cast to float32 for AMP, ensure scalar or [B]
        valid_start = valid_start_f if valid_start_ms.dtype != torch.float32 else valid_start_ms.view(-1)
        valid_end = valid_end_f if valid_end_ms.dtype != torch.float32 else valid_end_ms.view(-1)
        window_code = self.window_encoder(valid_start, valid_end)

        # STP 主干 — 玄女V2
        x = self.stp_encoder(frames, timestamps, mask)

        # 时间条件摘要
        pooled = x.mean(dim=(-2, -1))  # [B, T, C]
        query = self.summary_query(window_code)[:, None, :]
        attn_scores = torch.sum(query * pooled, dim=-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(attn_scores, dim=-1)
        summary = torch.einsum("bt,btchw->bchw", attn, x)

        # 将逐帧时间编码融入空间摘要（替代之前的 dummy 操作）
        time_summary = time_codes.mean(dim=1)  # [B, time_code_dim]
        summary = summary + self.time_to_summary(time_summary)[:, :, None, None]

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
        skip_decoder: bool = False,
        dual_window: bool = False,
        valid_start_w2: torch.Tensor | None = None,
        valid_end_w2: torch.Tensor | None = None,
        # Round 2: 目标窗口编码（跨时相重建）
        target_valid_start_ms: torch.Tensor | None = None,
        target_valid_end_ms: torch.Tensor | None = None,
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

        # Dual window 编码内联到同一 forward 中，避免 DDP 二次 forward 的
        # "mark ready only once" 和 inplace operation 错误。
        # 只编码 w2，不 backward 两次（所有梯度在同一 backward pass 中累积）。
        dual_pre_w2 = None
        if dual_window and valid_start_w2 is not None and valid_end_w2 is not None:
            summary_w2, _, _ = self.encode_frames(
                source_frames, source_timestamps_ms, source_frame_mask,
                source_input_mask, source_type_ids, valid_start_w2, valid_end_w2,
            )
            _, _, _, pre_norm_map_w2 = self.bottleneck(summary_w2)
            dual_pre_w2 = pre_norm_map_w2

        # Dummy 激活所有 sensor encoder 参数（避免 DDP 未使用参数报错）
        dummy_sensor = torch.tensor(0.0, device=source_frames.device)
        for encoder in self.sensor_encoder_bank.encoders.values():
            for p in encoder.parameters():
                dummy_sensor = dummy_sensor + p.sum() * 0.0
        embedding_map = embedding_map + dummy_sensor

        # 解码 (可选跳过，用于 dual-window forward 节省计算)
        B = summary_map.shape[0]
        if skip_decoder:
            # 返回 dummy 值，只保留 embedding 相关输出
            # 重建目标分辨率与原始输入同分辨率 (128x128)
            num_classes = self.cfg.data.num_classes
            num_tgt = self.cfg.data.num_target_sources
            reconstructions = torch.zeros(B, num_tgt, max(self._per_source_out_channels),
                                          self.image_size, self.image_size,
                                          device=embedding_map.device, dtype=embedding_map.dtype)
            logits = torch.zeros(B, num_classes, device=embedding_map.device, dtype=embedding_map.dtype)
            aux_logits = torch.zeros(B, num_classes, device=embedding_map.device, dtype=embedding_map.dtype)
            bottleneck_logits = torch.zeros(B, num_classes, device=embedding_map.device, dtype=embedding_map.dtype)
            summary_pooled = summary_map.mean(dim=(-2, -1))
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
                dual_pre_w2=dual_pre_w2,
            )

        if target_relative_time.dim() == 1:
            target_relative_time = target_relative_time[:, None]
        if target_metadata.dim() == 2:
            target_metadata = target_metadata[:, None, :]

        T_tgt = target_relative_time.shape[1]
        # AEF 对齐: 恢复条件解码 — 传递时间条件
        expanded_map = embedding_map[:, None, ...].repeat(1, T_tgt, 1, 1, 1)
        flat_map = expanded_map.reshape(B * T_tgt, *embedding_map.shape[1:])
        # ★ 关键修复: decoder 始终使用 L2 归一化后的 embedding，切断幅度信息泄漏通道。
        #   skip_l2_norm_training=True 时 embedding_map 是 pre-norm（有幅度），
        #   若 decoder 直接用幅度重建，recon 会快速降到 0 而不需要方向多样性 → 坍缩。
        flat_map = F.normalize(flat_map, p=2, dim=1)

        # Round 2: 使用目标窗口的 window_code（跨时相重建）
        if target_valid_start_ms is not None and target_valid_end_ms is not None:
            target_window_code = self.window_encoder(
                target_valid_start_ms.float().view(-1),
                target_valid_end_ms.float().view(-1),
            )
            cond_window = target_window_code[:, None, :].expand(B, T_tgt, -1).reshape(B * T_tgt, -1)
        else:
            cond_window = window_code[:, None, :].expand(B, T_tgt, -1).reshape(B * T_tgt, -1)

        # ★ Decoder Conditioning Dropout — 防止decoder依赖时间码走捷径
        #   训练时随机以 decoder_cond_dropout 概率将时间条件码清零，
        #   迫使 decoder 必须从 embedding 中提取信息，防止 embedding 坍缩。
        decoder_cond_dropout = getattr(self.cfg.training if hasattr(self.cfg, 'training') else self.cfg, 'decoder_cond_dropout', 0.0)
        if self.training and decoder_cond_dropout > 0:
            drop_mask = (torch.rand(B * T_tgt, 1, device=flat_map.device) < decoder_cond_dropout)
            cond_window = cond_window * (~drop_mask).float()
        
        # Round 2: 编码 relative_time
        # target_relative_time: [B, S_tgt] 或 [B, S_tgt, 1]
        if target_relative_time.dim() == 2 and target_relative_time.shape[-1] != 1:
            target_relative_time = target_relative_time.unsqueeze(-1)  # [B, S_tgt, 1]
        # 先 reshape 再编码
        flat_reltime = target_relative_time.reshape(B * T_tgt, -1)  # [B*T, 1]
        encoded_reltime = self.relative_time_encoder(flat_reltime)  # [B*T, reltime_dim]
        cond_reltime = encoded_reltime
        if target_metadata.dim() == 2:
            target_metadata = target_metadata[:, None, :]
        cond_meta = target_metadata.reshape(B * T_tgt, -1)

        max_ch = max(max(self._per_source_out_channels), self.cfg.data.num_classes)
        dec_h, dec_w = flat_map.shape[2], flat_map.shape[3]  # decoder 输出分辨率 (通常 64x64)
        flat_recon = torch.zeros(B * T_tgt, max_ch, dec_h, dec_w, device=flat_map.device, dtype=flat_map.dtype)

        if target_source_idx is not None:
            flat_src_idx = target_source_idx.reshape(B * T_tgt)
            for src_id, dec in enumerate(self.per_source_decoders):
                mask = (flat_src_idx == src_id)
                if mask.any():
                    out = dec(
                        flat_map[mask],
                        window_code=cond_window[mask],
                        relative_time=cond_reltime[mask],
                        metadata=cond_meta[mask],
                    )
                    out_ch = self._per_source_out_channels[src_id]
                    flat_recon[mask, :out_ch] = out.to(flat_recon.dtype)
        else:
            # 默认: 全部用第一个 decoder
            out = self.per_source_decoders[0](
                flat_map,
                window_code=cond_window,
                relative_time=cond_reltime,
                metadata=cond_meta,
            )
            flat_recon[:, :self._per_source_out_channels[0]] = out.to(flat_recon.dtype)

        # ★ 新增: bilinear upsample 回到原始分辨率 (128x128)
        if dec_h != self.image_size or dec_w != self.image_size:
            flat_recon = F.interpolate(
                flat_recon,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )

        reconstructions = flat_recon.reshape(B, T_tgt, max_ch, self.image_size, self.image_size)

        # 分类头 (不使用，但保留参数以兼容旧 checkpoint)
        logits = self.classification_head(embedding)
        summary_pooled = summary_map.mean(dim=(-2, -1))
        aux_logits = self.aux_cls_head(summary_pooled)
        bottleneck_logits = self.bottleneck_cls_head(pre_norm)

        # ★ 实例判别头: 使用 Global Max Pooling（GMP）代替 GAP 和空间 token
        # 根因修复：
        #   GAP PID: mean(pre_norm_map)=z* → 不同patch梯度互消 → 无效 (v1-v9)
        #   All-token PID: 驱动模型把判别信息压缩到1维 → erank从6→1.2 (v10)
        #   GMP PID: max(pre_norm_map) 不受零均值对称约束
        #            即使 mean(pre_norm_map)=z* for all patches，
        #            max 值仍然 patch-specific（不同土地覆盖→不同最大激活）
        if self.patch_id_head is not None and pre_norm_map is not None:
            # Global Max Pooling: [B, D, H, W] → [B, D]
            _gmp_h = pre_norm_map.max(dim=-1)[0]     # [B, D, H]
            _gmp = _gmp_h.max(dim=-1)[0]              # [B, D]
            patch_id_logits = self.patch_id_head(_gmp)  # [B, num_patches]
            patch_id_spatial_hw = 1  # single vector per sample
        else:
            patch_id_logits = None
            patch_id_spatial_hw = 1

        # Dummy 梯度流：确保 classification head 参数参与 backward
        # 加到 reconstructions 上（recon loss 一定会 backward）
        dummy_cls = (logits.sum() + aux_logits.sum() + bottleneck_logits.sum()) * 0.0
        if patch_id_logits is None and self.patch_id_head is not None:
            dummy_cls = dummy_cls  # head exists, logits computed
        reconstructions = reconstructions + dummy_cls.view(1, 1, 1, 1, 1)

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
            dual_pre_w2=dual_pre_w2,
            patch_id_logits=patch_id_logits,
            patch_id_spatial_hw=patch_id_spatial_hw,
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
        B = source_frames.shape[0]  # 无论 5D [B,T,C,H,W] 还是 6D [B,S,T,C,H,W]，第0维都是 batch
        dev = source_frames.device
        num_tgt = self.cfg.data.num_target_sources
        meta_dim = getattr(self.cfg.data, "metadata_dim", 4)

        # Window 1 — skip decoder 节省内存，temporal loss 只需要 embedding
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
            skip_decoder=True,
        )
        emb_w1 = F.normalize(out1.embedding_map, p=2, dim=1)
        pre_w1 = out1.pre_norm_map if out1.pre_norm_map is not None else out1.embedding_map

        # Window 2 — skip decoder 节省内存
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
            skip_decoder=True,
        )
        emb_w2 = F.normalize(out2.embedding_map, p=2, dim=1)
        pre_w2 = out2.pre_norm_map if out2.pre_norm_map is not None else out2.embedding_map

        return emb_w1, emb_w2, pre_w1, pre_w2

    def encode_dual_window_explicit_diff(
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
        """双窗口编码 — 显式 Difference Module（差分特征增强版）.

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

        # bottleneck 显式差分（差分增强路径）
        emb_w1, emb_w2, pre_w1, pre_w2, change_score, diff_feat = self.bottleneck.forward_dual_window(
            summary_w1, summary_w2
        )


        return emb_w1, emb_w2, pre_w1, pre_w2, change_score, diff_feat
