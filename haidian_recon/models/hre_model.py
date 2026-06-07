"""HaidianReconEncoder主模型 — 组装所有子模块."""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from haidian_recon.models.patch_embed import MultiScalePatchEmbed
from haidian_recon.models.encoder import HREncoder
from haidian_recon.models.decoder import HRDecoder
from haidian_recon.models.bottleneck import HRBottleneck
from haidian_recon.models.recon_head import HRReconstructionHead


class TokenEncoding(nn.Module):
    """时间/模态/位置编码."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_sources: int = 4,
        max_timesteps: int = 48,
        n_patches: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_embed = nn.Embedding(num_sources, embed_dim)
        self.time_embed = nn.Embedding(max_timesteps, embed_dim)
        # 2D位置编码: 可学习参数
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        source_id: int,
        time_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, T, N, D]
            source_id: int
            time_idx: int 或 [B, T]
        Returns:
            [B, T, N, D]
        """
        B, T, N, D = tokens.shape
        tokens = tokens + self.pos_embed[:, :N, :]
        tokens = tokens + self.modality_embed.weight[source_id].view(1, 1, 1, D)

        if isinstance(time_idx, int):
            tokens = tokens + self.time_embed.weight[time_idx].view(1, 1, 1, D)
        else:
            # time_idx: [B, T]
            te = self.time_embed(time_idx)  # [B, T, D]
            tokens = tokens + te[:, :, None, :]

        return tokens


class HREModel(nn.Module):
    """
    HaidianReconEncoder主模型。
    支持128x128输入 + 多尺度patch embed（分辨率分组）.
    """

    def __init__(
        self,
        source_channels: dict[str, int],
        image_size: int = 128,
        patch_size: int = 8,
        embed_dim: int = 512,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        output_dim: int = 64,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.source_channels = source_channels
        self.source_names = list(source_channels.keys())
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Patch Embedding — 多尺度，按分辨率分组
        self.patch_embed = MultiScalePatchEmbed(
            source_channels=source_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=image_size,
        )
        # 统一token数: 16x16 = 256
        self.n_patches = self.patch_embed.target_n_patches

        # Token Encoding
        self.token_encoding = TokenEncoding(
            embed_dim=embed_dim,
            num_sources=len(source_channels),
            max_timesteps=48,
            n_patches=self.n_patches,
        )

        # Encoder
        self.encoder = HREncoder(
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Bottleneck (全局embedding)
        self.bottleneck = HRBottleneck(
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Token-to-spatial: 从512维encoder token逐位置投影到64维
        self.token_to_spatial = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        # Spatial head: 对64维空间embedding做非线性调制
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Decoder
        self.decoder = HRDecoder(
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Mask token (decoder query)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 64维 → 512维投影，用于decoder重建条件
        self.cond_proj = nn.Sequential(
            nn.Linear(output_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Reconstruction heads
        self.recon_heads = nn.ModuleDict()
        for name, ch in source_channels.items():
            self.recon_heads[name] = HRReconstructionHead(
                embed_dim=embed_dim,
                out_channels=ch,
                patch_size=patch_size,
                image_size=image_size,
            )

    def forward(
        self,
        batch: dict[str, torch.Tensor | None],
        mask_info: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            batch: {source_name: [B, T, C, H, W] 或 None}
            mask_info: 若提供，表示batch已经过mask（训练时由外部masking模块生成）
                      若None，则不做mask（推理时）

        Returns:
            {
                "embedding": [B, output_dim],
                "embedding_map": [B, output_dim, H, W],
                "reconstructions": {source_name: [B, T, C, H, W]},
                "mask_info": mask_info,
            }
        """
        # 1. Patch Embedding + Token Encoding
        all_tokens = []
        source_info = []  # 记录每个源的空间布局信息

        for source_idx, source_name in enumerate(self.source_names):
            x = batch.get(source_name)
            if x is None:
                continue

            # Patch embed: [B, T, N, D] — N=256统一
            tokens = self.patch_embed(x, source_name)
            B, T, N, D = tokens.shape

            # Token encoding
            time_indices = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
            tokens = self.token_encoding(tokens, source_idx, time_indices)

            # 记录空间布局
            source_info.append({
                "name": source_name,
                "T": T,
                "N": N,
                "source_idx": source_idx,
            })

            # Flatten时间维度: [B, T*N, D]
            tokens = tokens.reshape(B, T * N, D)
            all_tokens.append(tokens)

        if len(all_tokens) == 0:
            raise ValueError("No valid input tokens")

        # 拼接所有token: [B, N_total, D]
        tokens = torch.cat(all_tokens, dim=1)

        # 2. Encoder
        encoder_output = self.encoder(tokens)

        # 3. Bottleneck → 全局64维Embedding
        embedding = self.bottleneck(encoder_output)

        # 4. 空间embedding: 从encoder_output逐位置投影 → [B, 64, 128, 128]
        grid_size = int(math.sqrt(self.n_patches))
        n_first = grid_size * grid_size  # 第一个源的256个空间token
        # 取encoder_output中前N个token，reshape为空间网格
        spatial_tokens = encoder_output[:, :n_first, :]  # [B, 256, 512]
        spatial_tokens = spatial_tokens.reshape(B, grid_size, grid_size, self.embed_dim)

        # 逐位置投影到64维（每个空间位置独立投影）
        spatial_emb = self.token_to_spatial(spatial_tokens)  # [B, g, g, 64]

        # 加入全局embedding做残差调制，增强全局-局部交互
        spatial_emb = spatial_emb + embedding.view(B, self.output_dim, 1, 1).permute(0, 2, 3, 1)

        # 通过spatial_head添加非线性变换
        spatial_emb = self.spatial_head(spatial_emb)  # [B, g, g, 64]
        spatial_emb_map = spatial_emb.permute(0, 3, 1, 2)  # [B, 64, g, g]

        # 上采样到128×128（每个像素一个64D向量）
        embedding_map = F.interpolate(
            spatial_emb_map,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # [B, 64, 128, 128]

        # L2 归一化：确保 embedding 分布在单位球面上（下游 kNN/cosine 评估需要）
        # 注意：保持 spatial_emb 不变，仅归一化返回的 embedding_map
        embedding_map_norm = embedding_map.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embedding_map = embedding_map / embedding_map_norm

        # 5. 推理时不需要重建
        if mask_info is None or not mask_info.get("decode_sources"):
            return {
                "embedding": embedding,
                "embedding_map": embedding_map,
                "reconstructions": {},
                "mask_info": mask_info,
            }

        # 6. Decoder: 为每个decode_source生成重建
        reconstructions = {}

        for source_name in mask_info["decode_sources"]:
            # 获取该源的时间步数（从原始batch中）
            original = mask_info.get("original_batch", {}).get(source_name)
            if original is None:
                continue
            if original.dim() != 5:
                continue
            T = original.shape[1]
            N = self.n_patches

            # 构建mask tokens: [B, T*N, D]
            mask_tokens = self.mask_token.expand(B, T * N, -1)

            # 添加位置/时间/模态编码
            source_idx = self.source_names.index(source_name)
            mask_tokens = mask_tokens.reshape(B, T, N, self.embed_dim)
            time_indices = torch.arange(T, device=mask_tokens.device).unsqueeze(0).expand(B, T)
            mask_tokens = self.token_encoding(mask_tokens, source_idx, time_indices)
            mask_tokens = mask_tokens.reshape(B, T * N, self.embed_dim)

            # Decoder: 使用空间64维embedding作为条件（每个位置独立，非广播）
            decoder_kv = spatial_emb.reshape(B, -1, self.output_dim)  # [B, 256, 64]
            decoder_kv = self.cond_proj(decoder_kv)  # [B, 256, 512]
            decoded = self.decoder(mask_tokens, decoder_kv)  # [B, T*N, D]

            # Reconstruction head
            decoded = decoded.reshape(B * T, N, self.embed_dim)
            recon = self.recon_heads[source_name](decoded)  # [B*T, C, H, W]
            C = recon.shape[1]
            recon = recon.reshape(B, T, C, self.image_size, self.image_size)
            reconstructions[source_name] = recon

        return {
            "embedding": embedding,
            "embedding_map": embedding_map,
            "reconstructions": reconstructions,
            "mask_info": mask_info,
        }
