"""
AlphaEarth Foundations Model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from einops import rearrange

from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.decoder import VonMisesFisherDecoder
from src.aef.architecture.encoder_utils import IndividualSourceEncoder, SummaryPeriodEncoder


class TimePooling(nn.Module):
    """
    Single-query multi-head attention over time at each (h,w).
    Inputs:
      feats: (B, T, H, W, C)
      q:     (B, C)           — from SummaryPeriodEncoder
      mask:  (B, T) optional  — 1 for valid frames, 0 for padded/missing
    Output:
      z:     (B, H, W, C)
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.kv = nn.Linear(dim, 2 * dim)     # to K,V
        self.q_proj = nn.Linear(dim, dim)     # to Q
        self.out = nn.Linear(dim, dim)

    def forward(self, feats: torch.Tensor, q: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, H, W, C = feats.shape
        BHW = B * H * W

        # keys/values from temporal features at each (h,w)
        x = feats.view(BHW, T, C)                                      # (BHW, T, C)
        kv = self.kv(x).view(BHW, T, 2, self.num_heads, self.head_dim)
        K, V = kv[:, :, 0], kv[:, :, 1]                                # (BHW, T, heads, d)
        K = K.permute(0, 2, 1, 3)                                      # (BHW, heads, T, d)
        V = V.permute(0, 2, 1, 3)                                      # (BHW, heads, T, d)

        # single query per sample, broadcast to all (h,w)
        qh = self.q_proj(q).view(B, self.num_heads, self.head_dim)     # (B, heads, d)
        qh = qh.unsqueeze(1).expand(B, H * W, self.num_heads, self.head_dim) \
               .reshape(BHW, self.num_heads, 1, self.head_dim)         # (BHW, heads, 1, d)

        # scaled dot-product attention over time
        logits = (qh * K).sum(-1) / (self.head_dim ** 0.5)             # (BHW, heads, 1, T)
        logits = logits.squeeze(2)                                      # (BHW, heads, T)

        if mask is not None:
            mask_flat = mask.unsqueeze(1).unsqueeze(1)                 # (B,1,1,T)
            mask_flat = mask_flat.expand(B, H * W, self.num_heads, T)  # (B,HW,heads,T)
            mask_flat = mask_flat.reshape(BHW, self.num_heads, T)      # (BHW,heads,T)
            logits = logits.masked_fill(mask_flat == 0, float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        # NaN protection: if all logits are -inf (fully masked), softmax outputs NaN
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        
        if attn.dim() == 2:
            attn = attn.unsqueeze(-1)
        
        z = torch.einsum('bht,bhtd->bhd', attn, V)
        z = z.reshape(BHW, self.num_heads * self.head_dim)
        z = self.out(z).view(B, H, W, C)                               # (B,H,W,C)
        return z


class TemporalSummarizer(nn.Module):
    """
    Time-conditional summarization using a single query per sample and
    multi-head attention over time at each spatial location.
    训练时输出 pre-norm 64D 向量（保留幅度信息，用于反坍缩损失）。
    """

    def __init__(self, feature_dim: int, embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.summarizer_q = SummaryPeriodEncoder(dim=feature_dim)
        # Spatial smoothing before time pooling to suppress residual grid artifacts
        self.spatial_smooth = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=1)
        self.time_pool = TimePooling(dim=feature_dim, num_heads=num_heads)
        self.proj_64 = nn.Linear(feature_dim, embed_dim, bias=False)
        # 增大初始化 std，让初始 embedding 更分散，打破坍缩死锁
        nn.init.xavier_normal_(self.proj_64.weight, gain=2.0)

    def forward(self, feats: torch.Tensor, timestamps: torch.Tensor,
                valid_periods: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            feats: (B, T, H, W, C)
            timestamps: (B, T) in consistent units (kept for API; mask can derive from it)
            valid_periods: (B, 2) [t_s, t_e)
            mask: optional (B, T) validity mask
        Returns:
            embeddings: (B, H, W, 64), pre-norm (training) / unit-norm (inference)
            **训练时不做 L2 norm**，反坍缩损失在 pre-norm 欧氏空间计算以绕过 Jacobian 梯度屏障。
        """
        # Spatial smoothing to suppress residual grid artifacts from encoder upsampling
        B, T, H, W, C = feats.shape
        feats_2d = feats.view(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()  # (BT, C, H, W)
        feats_2d = self.spatial_smooth(feats_2d)
        feats_smooth = feats_2d.permute(0, 2, 3, 1).contiguous().view(B, T, H, W, C)
        
        # Build single query per sample
        q = self.summarizer_q(valid_periods)                 # (B, C)
        # Pool over time at each (h,w)
        z = self.time_pool(feats_smooth, q, mask=mask)       # (B, H, W, C)
        # Project to 64D，**训练时不 L2 norm**，保留幅度信息
        mu = self.proj_64(z)                                 # (B, H, W, 64)
        return mu


class AlphaEarthFoundations(nn.Module):
    """
    Complete AlphaEarth Foundations model. 
    Architecture 
    - Preprocessing with normalization and sinusoidal timecodes
    - Individual source encoders
    - Teacher network: STP encoder generating embeddings
    - Student network: learns from teacher via consistency loss  
    - Text network: enables video-text contrastive learning
    - von Mises-Fisher decoder for source reconstruction
    """
    def __init__(self,
                 model_size: str = "small",
                 input_sources: Optional[Dict[str, int]] = None,
                 decode_sources: Optional[Dict[str, int]] = None,
                 per_source_latent: int = 32,
                 enable_text_align: bool = False):
        """
        Args:
            model_size: 
            input_sources: mapping of source name -> channel count (inputs only)
            decode_sources: mapping of source name -> channel count to decode
        """
        super().__init__()

        # Default to Sentinel-2 only for now (5 bands), matching the provided dataloader
        if input_sources is None:
            input_sources = {"sentinel2": 5}
        if decode_sources is None:
            decode_sources = {"sentinel2": input_sources.get("sentinel2", 5)}

        self.input_sources = input_sources
        self.decode_sources = decode_sources
        self.enable_text_align = enable_text_align

        # Choose dimensions as per paper (S2.4), scaled down 
        d_p, d_t, d_s, num_blocks = 64, 256, 512, 6

        # Per-source encoders (Preprocessing box in Fig. 2A)
        self.source_encoders = nn.ModuleDict()
        for name, c in input_sources.items():
            self.source_encoders[name] = IndividualSourceEncoder(c, per_source_latent)

        total_in_channels = per_source_latent * len(input_sources)
        self.encoder = STPEncoder(
            input_channels=total_in_channels,
            d_s=d_s, d_t=d_t, d_p=d_p,
            num_blocks=num_blocks,
        )

        # Summarizer produces 64D embeddings on S^63
        self.summarizer = TemporalSummarizer(feature_dim=d_p, embed_dim=64)

        # VMF implicit decoder for sources
        self.decoder = VonMisesFisherDecoder(
            embedding_dim=64,               
            source_dims=self.decode_sources,
            geometry_dim=16,
        )

    def _stack_inputs(self, source_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-source encode, then concatenate along channel axis -> (B, T, H, W, C_total)."""
        xs: List[torch.Tensor] = []
        # 获取参考张量以推断 shape
        ref = next(iter(source_data.values())) if source_data else None
        for name, C in self.input_sources.items():
            if name in source_data:
                x = source_data[name]  # (B, T, H, W, C)
            elif ref is not None:
                # 缺失的源用零张量填充（防御性处理）
                B, T, H, W, _ = ref.shape
                x = torch.zeros(B, T, H, W, C, dtype=ref.dtype, device=ref.device)
            else:
                raise ValueError(f"source_data is empty and cannot infer shape for {name}")
            B, T, H, W, _ = x.shape
            flat = rearrange(x, 'b t h w c -> (b t h w) c')
            enc = self.source_encoders[name](flat)
            enc = rearrange(enc, '(b t h w) c -> b t h w c', b=B, t=T, h=H, w=W)
            xs.append(enc)
        x_cat = torch.cat(xs, dim=-1)
        return x_cat

    def _perturb_inputs(self,
                        source_data: Dict[str, torch.Tensor],
                        timestamps: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply teacher-student input perturbations (S2.2.5): random source drops and time drops.
        Returns perturbed copies of source_data and timestamps for the student.
        """
        student_sources: Dict[str, torch.Tensor] = {}
        student_ts: Dict[str, torch.Tensor] = {}

        drop_prob = {name: (0.0 if name in ('sentinel2', 's2') else 0.3) for name in self.input_sources.keys()} # only keep S2 always for reconstruction

        for name, x in source_data.items():
            ts = timestamps[name]
            B, T, H, W, C = x.shape
            keep_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

            # Per-sample random: drop entire source for student with prob
            source_drop = torch.rand(B, device=x.device) < drop_prob.get(name, 0.2)
            keep_mask[source_drop, :] = False

            # Per-sample random time perturbation strategy
            # 0: random frame drops, 1: drop latter half-year, 2: drop former half-year
            strat = torch.randint(low=0, high=3, size=(B,), device=x.device)

            # Pre-compute half-year masks (vectorized)
            t0 = ts.min(dim=1, keepdim=True).values  # (B, 1)
            t1 = ts.max(dim=1, keepdim=True).values  # (B, 1)
            mid = (t0 + t1) / 2.0
            drop_latter = ts <= mid   # (B, T)
            drop_former = ts >= mid   # (B, T)

            # Per-sample time perturbation
            frac = 0.5 if name in ('sentinel2', 's2') else 0.3
            rand_drops = torch.rand(B, T, device=x.device) < frac

            for b in range(B):
                if source_drop[b]:
                    continue
                if strat[b] == 0:
                    keep_mask[b] = ~rand_drops[b]
                elif strat[b] == 1:
                    keep_mask[b] = drop_latter[b]
                else:
                    keep_mask[b] = drop_former[b]

            # Apply keep_mask: zero out dropped frames and repeat last timestamp to keep shapes
            keep_mask_4d = keep_mask.view(B, T, 1, 1, 1)
            x_pert = x * keep_mask_4d
            # timestamps: 将被丢弃帧的时间戳设置为该样本内最近保留帧的时间戳
            ts_pert = ts.clone()
            for b in range(B):
                kept_indices = torch.where(keep_mask[b])[0]
                if kept_indices.numel() > 0:
                    for t_idx in range(T):
                        if not keep_mask[b, t_idx]:
                            nearest = kept_indices[(kept_indices - t_idx).abs().argmin()]
                            ts_pert[b, t_idx] = ts[b, nearest]

            student_sources[name] = x_pert
            student_ts[name] = ts_pert

        return student_sources, student_ts

    def forward(self,
                source_data: Dict[str, torch.Tensor],
                timestamps: Dict[str, torch.Tensor],
                valid_periods: List[Tuple[float, float]],
                geometry_metadata: Optional[torch.Tensor] = None,
                num_decode_samples: int = 1) -> Dict[str, Any]:
        """
        End-to-end forward pass: encoder -> temporal summarization -> implicit decoding.

        Args:
            source_data: dict of tensors per input source (B, T, H, W, C)
            timestamps: dict of (B, T) ms; only the inputs' timestamps are required
            valid_periods: list of (start_ms, end_ms) length B
            geometry_metadata: optional (B, 16) sensor geometry; zeros if None
            num_decode_samples: number of VMF samples to draw during decoding

        Returns:
            Dict with keys:
              - embeddings: (B, H', W', 64) **pre-norm** vectors (训练时保留幅度)
              - reconstructions: dict of decoded sources -> (B, S, H', W', C_src)
        """
        # Prepare inputs
        x = self._stack_inputs(source_data)  # (B, T, H, W, C)
        # Use timestamps from the first input source (they are aligned by collate)
        first_src = next(iter(self.input_sources.keys()))
        ts = timestamps[first_src]  # (B, T)

        # valid_periods: convert to tensor on correct device/dtype
        first_tensor = source_data[first_src]
        if isinstance(valid_periods, list):
            vp = torch.tensor(valid_periods, dtype=first_tensor.dtype, device=first_tensor.device)
        else:
            vp = valid_periods.to(first_tensor.dtype).to(first_tensor.device)

        # Encode (teacher) with STP to full resolution (B, T, H, W, d_p)
        feats_teacher = self.encoder(x, ts)
        mu_t = self.summarizer(feats_teacher, ts, vp)  # (B, H, W, 64)

        # Build student inputs via perturbation, then encode student
        student_srcs, student_ts_dict = self._perturb_inputs(source_data, timestamps)
        x_student = self._stack_inputs(student_srcs)
        ts_student = student_ts_dict[first_src]
        feats_student = self.encoder(x_student, ts_student)
        mu_s = self.summarizer(feats_student, ts_student, vp)  # (B, H, W, 64)
        
        # Add spatial noise to break horizontal striping artifacts from Sentinel-2
        if self.training:
            noise_scale = 0.2
            mu_t = mu_t + torch.randn_like(mu_t) * noise_scale
            mu_s = mu_s + torch.randn_like(mu_s) * noise_scale

        B, H2, W2, _ = mu_t.shape
        if geometry_metadata is None:
            geometry_metadata = torch.zeros(B, 16, dtype=feats_teacher.dtype, device=feats_teacher.device)

        # For decoding timestamps, use the middle time of the support for each sample
        ts_center = ts.mean(dim=1)  # (B,)

        reconstructions: Dict[str, torch.Tensor] = {}
        for src, _ch in self.decode_sources.items():
            recon = self.decoder(
                embeddings=mu_t,
                geometry_metadata=geometry_metadata,
                timestamps=ts_center,
                valid_period=(vp[:, 0], vp[:, 1]),
                source=src,
                num_samples=num_decode_samples,
            )  # (B, S, H, W, C_src) — decoder 直接输出全分辨率
            reconstructions[src] = recon

        # Image-level pooled embeddings (for text alignment)
        img_embed_t = mu_t.mean(dim=(1, 2))  # (B, 64)
        img_embed_s = mu_s.mean(dim=(1, 2))  # (B, 64)

        out: Dict[str, Any] = {
            'embeddings': mu_t,
            'teacher_embeddings': mu_t,
            'student_embeddings': mu_s,
            'image_embeddings': img_embed_t,
            'reconstructions': reconstructions,
        }

        if self.enable_text_align:
            out['needs_text'] = True

        return out
