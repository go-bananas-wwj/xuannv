
from typing import Any, Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.functional import F
import math

"""
AEF loss = Reconstruction loss + Uniformity loss + Consistency loss + Text loss + Distill loss
新增: raw_uniformity (pre-norm), VICReg variance/covariance, decorrelation
"""


class AEFLoss:
    """
    AlphaEarth Foundations loss implementation.
    强化反坍缩: raw_uniformity + VICReg + 高权重 distill
    """
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,
                 uniformity_weight: float = 2.0,      # L2 uniformity
                 consistency_weight: float = 0.02,
                 text_weight: float = 0.001,
                 distill_weight: float = 0.5,         # 降低: 教师本身坍缩
                 raw_uniform_weight: float = 10.0,    # 提高: pre-norm 反坍缩
                 variance_weight: float = 50.0,       # 提高: VICReg 硬性约束
                 covariance_weight: float = 5.0,       # 提高: 维度去相关
                 decorr_weight: float = 0.0,           # 关闭: 无效且值巨大
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
        
        self.source_configs = {
            'sentinel2': {'weight': 1.0, 'loss_fn': F.l1_loss},
        }
    
    def reconstruction_loss(self, predictions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        
        for source in predictions:
            if source in targets:
                config = self.source_configs.get(source, {'weight': 1.0, 'loss_fn': F.l1_loss})
                mask = masks.get(source, torch.ones_like(targets[source]))
                
                pred_masked = predictions[source] * mask
                target_masked = targets[source] * mask
                
                if config['loss_fn'] == F.cross_entropy:
                    loss = config['loss_fn'](pred_masked, target_masked.long())
                else:
                    loss = config['loss_fn'](pred_masked, target_masked)

                total_loss += config['weight'] * loss
        
        return total_loss
    
    def batch_uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2 空间 batch uniformity (旧版，值越低越好)."""
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x_prime = torch.roll(x, shifts=1, dims=0)
        dots = (x * x_prime).sum(dim=-1).abs()
        return dots.mean()
    
    def raw_uniformity_loss(self, embeddings: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Pre-norm 欧氏空间 uniformity (NPU-native, 无 L2 Jacobian 梯度屏障).
        值越负越好（-4.0 ~ -1.0 为健康范围），越接近 0 越坍缩。
        """
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        
        # 计算平均配对距离 (NPU-safe，避免成对矩阵)
        x_shift = torch.roll(x, shifts=x.size(0)//2, dims=0)
        sq_dists = ((x - x_shift) ** 2).sum(dim=-1)
        
        # 自适应温度: t = 2 / D
        D = x.size(-1)
        t = 2.0 / max(D, 1)
        
        loss = torch.exp(-t * sq_dists).mean()
        # 返回 log：坍缩时→0，展开时→-∞，最小化即推开 embedding
        return torch.log(loss + eps)
    
    def vicreg_variance_loss(self, embeddings: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """
        VICReg variance loss: 约束 batch 中每个维度的标准差 >= gamma.
        这是防止坍缩的硬性约束。
        """
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        
        # x: (N, D)
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        # hinge: max(0, gamma - std)
        return torch.mean(torch.relu(gamma - std))
    
    def vicreg_covariance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        VICReg covariance loss: 让维度之间去相关.
        """
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        
        # x: (N, D)
        N = x.size(0)
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (N - 1)
        # 非对角线元素的平方和
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D
    
    def decorrelation_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Barlow Twins 风格去相关损失.
        """
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        
        # 标准化
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        N = x.size(0)
        c = (x.T @ x) / N
        
        # 目标是单位矩阵
        identity = torch.eye(D, device=x.device, dtype=x.dtype)
        diff = (c - identity) ** 2
        # 对角线权重 1，非对角线权重 1
        return diff.sum()
    
    def consistency_loss(self, teacher_embeddings: torch.Tensor, 
                        student_embeddings: torch.Tensor) -> torch.Tensor:
        mu = torch.nn.functional.normalize(teacher_embeddings, p=2, dim=-1)
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
            return torch.tensor(0.0, device=pred_embeddings.device if pred_embeddings is not None else 'cpu')
        
        B, H, W, D_pred = pred_embeddings.shape
        B_t, D_t, H_t, W_t = target_embeddings.shape
        
        if H_t != H or W_t != W:
            target_2d = target_embeddings
            target_2d = F.interpolate(target_2d, size=(H, W), mode='bilinear', align_corners=False)
        else:
            target_2d = target_embeddings
        
        target = rearrange(target_2d, 'b c h w -> b h w c')
        
        pred_n = torch.nn.functional.normalize(pred_embeddings, p=2, dim=-1)
        tgt_n = torch.nn.functional.normalize(target, p=2, dim=-1)
        cosine_sim = (pred_n * tgt_n).sum(dim=-1)
        
        loss = (1.0 - cosine_sim)
        
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
        
        # L2 uniformity (旧版)
        if 'embeddings' in outputs:
            uniformity_loss = self.batch_uniformity_loss(outputs['embeddings'])
            losses['uniformity'] = uniformity_loss
        else:
            losses['uniformity'] = torch.tensor(0.0, device=device)
        
        # Raw uniformity (pre-norm, 新增)
        if 'embeddings' in outputs:
            raw_uniform_loss = self.raw_uniformity_loss(outputs['embeddings'])
            losses['raw_uniform'] = raw_uniform_loss
        else:
            losses['raw_uniform'] = torch.tensor(0.0, device=device)
        
        # VICReg variance (新增)
        if 'embeddings' in outputs:
            var_loss = self.vicreg_variance_loss(outputs['embeddings'])
            losses['variance'] = var_loss
        else:
            losses['variance'] = torch.tensor(0.0, device=device)
        
        # VICReg covariance (新增)
        if 'embeddings' in outputs:
            cov_loss = self.vicreg_covariance_loss(outputs['embeddings'])
            losses['covariance'] = cov_loss
        else:
            losses['covariance'] = torch.tensor(0.0, device=device)
        
        # Decorrelation (新增)
        if 'embeddings' in outputs:
            decorr_loss = self.decorrelation_loss(outputs['embeddings'])
            losses['decorr'] = decorr_loss
        else:
            losses['decorr'] = torch.tensor(0.0, device=device)

        if 'teacher_embeddings' in outputs and 'student_embeddings' in outputs:
            consistency_loss = self.consistency_loss(
                outputs['teacher_embeddings'],
                outputs['student_embeddings']
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)
        
        if 'image_embeddings' in outputs and 'text_embeddings' in outputs:
            clip_loss = self.clip_loss(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
            losses['clip'] = clip_loss
        else:
            losses['clip'] = torch.tensor(0.0, device=device)
        
        # AEF 官方 embedding 蒸馏
        if 'aef_embedding_pred' in outputs and 'aef_embedding_target' in outputs:
            distill_loss = self.aef_distill_loss(
                outputs['aef_embedding_pred'],
                outputs['aef_embedding_target'],
                outputs.get('aef_embedding_valid'),
            )
            losses['distill'] = distill_loss
        else:
            losses['distill'] = torch.tensor(0.0, device=device)
        
        total_loss = (
            self.reconstruction_weight * losses['reconstruction'] +
            self.uniformity_weight * losses['uniformity'] +
            self.raw_uniform_weight * losses['raw_uniform'] +
            self.variance_weight * losses['variance'] +
            self.covariance_weight * losses['covariance'] +
            self.decorr_weight * losses['decorr'] +
            self.consistency_weight * losses['consistency'] +
            self.text_weight * losses['clip'] +
            self.distill_weight * losses['distill']
        )
        
        losses['total'] = total_loss
        
        return losses
