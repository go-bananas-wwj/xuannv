from typing import Any, Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.functional import F
import math


class AEFLoss:
    """
    AlphaEarth Foundations loss implementation.
    Aligned with official paper (arXiv:2507.22291):
      - Reconstruction (α=1.0)
      - Batch Uniformity (b=0.05)
      - Consistency (c=0.02)
      - Text Contrastive (d=0.001)
    Plus AEF distillation for staged training.
    """

    def __init__(self,
                 reconstruction_weight: float = 1.0,
                 uniformity_weight: float = 0.05,
                 consistency_weight: float = 0.02,
                 text_weight: float = 0.001,
                 distill_weight: float = 0.2,
                 # Disabled losses (kept for backward compatibility, weight=0)
                 raw_uniform_weight: float = 0.0,
                 variance_weight: float = 0.0,
                 covariance_weight: float = 0.0,
                 decorr_weight: float = 0.0,
                 erank_weight: float = 0.0,
                 coding_rate_weight: float = 0.0,
                 magnitude_weight: float = 0.0,
                 ):

        self.reconstruction_weight = reconstruction_weight
        self.uniformity_weight = uniformity_weight
        self.consistency_weight = consistency_weight
        self.text_weight = text_weight
        self.distill_weight = distill_weight
        self.raw_uniform_weight = raw_uniform_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.decorr_weight = decorr_weight
        self.erank_weight = erank_weight
        self.coding_rate_weight = coding_rate_weight
        self.magnitude_weight = magnitude_weight

        self.source_configs = {
            's2': {'weight': 1.0, 'loss_fn': F.l1_loss},
            's1': {'weight': 1.0, 'loss_fn': F.l1_loss},
            'tianyi_sar': {'weight': 1.0, 'loss_fn': F.l1_loss},
            'landsat': {'weight': 1.0, 'loss_fn': F.l1_loss},
            'planet': {'weight': 1.0, 'loss_fn': F.l1_loss},
            'dem': {'weight': 0.05, 'loss_fn': F.l1_loss},
            'jrc_water': {'weight': 0.3, 'loss_fn': F.l1_loss},
            'worldcover': {'weight': 0.5, 'loss_fn': F.cross_entropy},
            'dynamic_world': {'weight': 0.5, 'loss_fn': F.cross_entropy},
            'sentinel2': {'weight': 1.0, 'loss_fn': F.l1_loss},
        }

    def set_stage(self, stage: str):
        """Dynamic weight switching for staged training."""
        if stage == "distill_align":
            self.reconstruction_weight = 0.01
            self.distill_weight = 5.0
            self.uniformity_weight = 0.5
            self.consistency_weight = 0.02
            self.text_weight = 0.0
        elif stage == "normal":
            self.reconstruction_weight = 1.0
            self.distill_weight = 0.2
            self.uniformity_weight = 0.05
            self.consistency_weight = 0.02
            self.text_weight = 0.001
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def reconstruction_loss(self, predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = None

        for source in predictions:
            if source in targets:
                config = self.source_configs.get(source, {'weight': 1.0, 'loss_fn': F.l1_loss})
                mask = masks.get(source, torch.ones_like(targets[source]))

                pred_masked = predictions[source] * mask
                target_masked = targets[source] * mask

                if config['loss_fn'] == F.cross_entropy:
                    # NPU cross_entropy expects (N, C, ...) logits and (N, ...) targets
                    pred_ce = pred_masked.permute(0, 3, 1, 2)  # (B, C, H, W)
                    # 分类目标直接从 targets[source] 取，避免与 float mask 相乘导致类型错误
                    target_raw = targets[source]
                    if target_raw.dim() == 4 and target_raw.shape[-1] > 1:
                        target_ce = target_raw.argmax(dim=-1).long()
                    else:
                        target_ce = target_raw.squeeze(-1).long()
                    loss = F.cross_entropy(pred_ce, target_ce, ignore_index=255)
                else:
                    loss = config['loss_fn'](pred_masked, target_masked)

                if total_loss is None:
                    total_loss = config['weight'] * loss
                else:
                    total_loss += config['weight'] * loss

        if total_loss is None:
            return next(iter(predictions.values())).new_tensor(0.0)
        return total_loss

    def batch_uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2 space batch uniformity (lower is better).

        CRITICAL FIX: compute uniformity across distinct samples, never spatial neighbours.
        For small per-GPU batch sizes (B < 4), augments the sample pool with spatial
        pixels to ensure statistical validity. Uses random pair sampling instead of
        deterministic torch.roll to avoid forcing adjacency orthogonality.
        """
        x = embeddings
        if x.dim() == 5:
            # (B, T, H, W, D) -> pool over (H,W) -> (B, T, D) -> flatten time -> (B*T, D)
            B, T, H, W, D = x.shape
            x = x.permute(0, 1, 4, 2, 3).reshape(B * T, D, H, W)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(B * T, D)
        elif x.dim() == 4:
            B, H, W, D = x.shape
            if B >= 4:
                # Large batch: pool over space to get per-sample embeddings
                x = x.permute(0, 3, 1, 2)  # (B, D, H, W)
                x = F.adaptive_avg_pool2d(x, (1, 1)).view(B, D)
            else:
                # Small batch: use spatial pixels to augment sample pool
                x = x.reshape(B * H * W, D)

        x = F.normalize(x, p=2, dim=-1)
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        N = x.shape[0]
        if N < 2:
            return x.new_tensor(0.0)

        # Random pair sampling avoids deterministic adjacency artifacts
        K = min(N * 2, 512)
        idx_i = torch.randint(0, N, (K,), device=x.device)
        offset = torch.randint(1, N, (K,), device=x.device)
        idx_j = (idx_i + offset) % N

        dots = (x[idx_i] * x[idx_j]).sum(dim=-1).abs()
        return dots.mean()

    def raw_uniformity_loss(self, embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Pre-norm Euclidean uniformity (kept for compatibility, disabled by default)."""
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        N = x.size(0)
        if N < 2:
            return x.new_tensor(0.0)

        x = x + torch.randn_like(x) * 0.05
        K = min(N * 2, 512)
        idx_i = torch.randint(0, N, (K,), device=x.device)
        offset = torch.randint(1, N, (K,), device=x.device)
        idx_j = (idx_i + offset) % N
        sq_dists = ((x[idx_i] - x[idx_j]) ** 2).sum(dim=-1)
        D = x.size(-1)
        t = 2.0 / max(D, 1)
        loss = torch.exp(-t * sq_dists).mean()
        return torch.log(loss + eps)

    def vicreg_variance_loss(self, embeddings: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        N = x.size(0)
        if N < 2:
            return x.new_tensor(0.0)

        var = x.var(dim=0, unbiased=False)
        return torch.mean(torch.relu(gamma ** 2 - var))

    def vicreg_covariance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        N = x.size(0)
        if N < 2:
            return x.new_tensor(0.0)

        x = x - x.mean(dim=0)
        cov = (x.T @ x) / max(N - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / x.size(-1)

    def decorrelation_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        N = x.size(0)
        if N < 2:
            return x.new_tensor(0.0)

        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        c = (x.T @ x) / max(N, 1)
        identity = torch.eye(x.size(-1), device=x.device, dtype=x.dtype)
        diff = (c - identity) ** 2
        return diff.sum()

    def magnitude_loss(self, embeddings: torch.Tensor, min_norm: float = 0.5) -> torch.Tensor:
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        norms = x.norm(dim=-1)
        return torch.relu(min_norm - norms).mean()

    def erank_maximization_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        N, D = x.shape
        if N < 2:
            return x.new_tensor(0.0)

        x = x - x.mean(0, keepdim=True)
        col_var = x.pow(2).mean(dim=0).clamp(min=1e-8)
        probs = col_var / col_var.sum()
        entropy = -(probs * probs.log()).sum()
        max_entropy = math.log(float(D))
        return (max_entropy - entropy).clamp(min=0.0)

    def coding_rate_loss(self, embeddings: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        if x.shape[0] < 2:
            return x.new_tensor(0.0)

        N, D = x.shape
        Z = x - x.mean(dim=0, keepdim=True)
        col_var = Z.pow(2).mean(dim=0)
        alpha = D / (N * eps ** 2 + 1e-8)
        loss = -0.5 * (1.0 + alpha * col_var).clamp(min=1e-8).log().sum()
        if torch.isnan(loss) or torch.isinf(loss):
            return x.new_tensor(0.0)
        return loss

    def consistency_loss(self, teacher_embeddings: torch.Tensor,
                        student_embeddings: torch.Tensor) -> torch.Tensor:
        # Teacher stop-gradient: student learns to match teacher
        mu = torch.nn.functional.normalize(teacher_embeddings.detach(), p=2, dim=-1)
        mu_s = torch.nn.functional.normalize(student_embeddings, p=2, dim=-1)
        dots = (mu * mu_s).sum(dim=-1)
        return ((1.0 - dots) * 0.5).mean()

    def clip_loss(self, image_embeddings: torch.Tensor,
                  text_embeddings: torch.Tensor) -> torch.Tensor:
        img = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        txt = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        logits = img @ txt.t()
        targets = torch.arange(img.size(0), device=img.device)
        loss_i = torch.nn.functional.cross_entropy(logits, targets)
        loss_t = torch.nn.functional.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_i + loss_t)

    def aef_distill_loss(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred_embeddings is None or target_embeddings is None:
            device = pred_embeddings.device if pred_embeddings is not None else (
                target_embeddings.device if target_embeddings is not None else torch.device('cpu')
            )
            return torch.tensor(0.0, device=device)

        B, H, W, D_pred = pred_embeddings.shape
        B_t, D_t, H_t, W_t = target_embeddings.shape

        if H_t != H or W_t != W:
            target_2d = target_embeddings
            target_2d = F.interpolate(target_2d, size=(H, W), mode='bilinear', align_corners=False)
        else:
            target_2d = target_embeddings

        target = rearrange(target_2d, 'b c h w -> b h w c')

        pred_n = F.normalize(pred_embeddings, p=2, dim=-1)
        tgt_n = F.normalize(target, p=2, dim=-1)
        cosine_sim = (pred_n * tgt_n).sum(dim=-1)
        cosine_loss = (1.0 - cosine_sim)

        # Magnitude matching: prevent collapse to near-zero pre-norm vectors
        pred_mag = pred_embeddings.norm(dim=-1)
        tgt_mag = target.norm(dim=-1)
        mag_loss = ((pred_mag - tgt_mag) ** 2) / (tgt_mag ** 2 + 1e-8)

        loss = cosine_loss + 0.001 * mag_loss

        if valid_mask is not None:
            valid_mask_2d = valid_mask.view(B, 1, 1).expand(B, H, W)
            loss = (loss * valid_mask_2d.float()).sum() / (valid_mask_2d.float().sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def __call__(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        losses = {}
        device = next(iter(outputs.values())).device if outputs else torch.device('cpu')

        if 'predictions' in outputs and 'targets' in outputs:
            recon_loss = self.reconstruction_loss(
                outputs['predictions'],
                outputs['targets'],
                outputs.get('masks', {})
            )
            losses['reconstruction'] = recon_loss
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=device)

        if 'embeddings' in outputs:
            losses['uniformity'] = self.batch_uniformity_loss(outputs['embeddings'])
            losses['raw_uniform'] = self.raw_uniformity_loss(outputs['embeddings'])
            losses['variance'] = self.vicreg_variance_loss(outputs['embeddings'])
            losses['covariance'] = self.vicreg_covariance_loss(outputs['embeddings'])
            losses['decorr'] = self.decorrelation_loss(outputs['embeddings'])
            losses['erank'] = self.erank_maximization_loss(outputs['embeddings'])
            losses['coding_rate'] = self.coding_rate_loss(outputs['embeddings'])
            losses['magnitude'] = self.magnitude_loss(outputs['embeddings'])
        else:
            for k in ['uniformity', 'raw_uniform', 'variance', 'covariance',
                      'decorr', 'erank', 'coding_rate', 'magnitude']:
                losses[k] = torch.tensor(0.0, device=device)

        if 'teacher_embeddings' in outputs and 'student_embeddings' in outputs:
            losses['consistency'] = self.consistency_loss(
                outputs['teacher_embeddings'],
                outputs['student_embeddings']
            )
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)

        if 'image_embeddings' in outputs and 'text_embeddings' in outputs:
            losses['clip'] = self.clip_loss(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
        else:
            losses['clip'] = torch.tensor(0.0, device=device)

        if 'aef_embedding_pred' in outputs and 'aef_embedding_target' in outputs:
            losses['distill'] = self.aef_distill_loss(
                outputs['aef_embedding_pred'],
                outputs['aef_embedding_target'],
                outputs.get('aef_embedding_valid'),
            )
        else:
            losses['distill'] = torch.tensor(0.0, device=device)

        total_loss = (
            self.reconstruction_weight * losses['reconstruction'] +
            self.uniformity_weight * losses['uniformity'] +
            self.raw_uniform_weight * losses['raw_uniform'] +
            self.variance_weight * losses['variance'] +
            self.covariance_weight * losses['covariance'] +
            self.decorr_weight * losses['decorr'] +
            self.erank_weight * losses['erank'] +
            self.coding_rate_weight * losses['coding_rate'] +
            self.magnitude_weight * losses['magnitude'] +
            self.consistency_weight * losses['consistency'] +
            self.text_weight * losses['clip'] +
            self.distill_weight * losses['distill']
        )

        losses['total'] = total_loss

        return losses
