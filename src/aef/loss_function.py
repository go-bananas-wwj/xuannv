
from typing import Any, Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.functional import F
import math

"""
AEF loss = Reconstruction loss + Uniformity loss + Consistency loss + Text loss
"""

class AEFLoss:
    """
    AlphaEarth Foundations loss implementation following Equation 3 in the paper.
    新增 AEF 官方 embedding 蒸馏损失，用于防止 uniformity 坍缩。
    """
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,  # a = 1.0
                 uniformity_weight: float = 0.05,    # b = 0.05
                 consistency_weight: float = 0.02,   # c = 0.02
                 text_weight: float = 0.001,         # d = 0.001
                 distill_weight: float = 0.5):       # e = 0.5 (AEF 官方 embedding 蒸馏)
        
        self.reconstruction_weight = reconstruction_weight
        self.uniformity_weight = uniformity_weight
        self.consistency_weight = consistency_weight
        self.text_weight = text_weight
        self.distill_weight = distill_weight
        
        # Source-specific loss configurations from Table S2
        self.source_configs = {
            'sentinel2': {'weight': 1.0, 'loss_fn': F.l1_loss},
        }
    
    def reconstruction_loss(self, predictions: Dict[str, torch.Tensor], 
                          targets: Dict[str, torch.Tensor],
                          masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reconstruction loss for all sources

        Compares predicted observation y_i' with ground truth y_i for each source i --> this leads the model to force the embeddings to carry enough information to be able to reconstruct the raw EO inputs. 
        For continuous sources: use L1 Loss; for categorical sources: use Cross-entropy.

        """
        
        total_loss = 0.0
        
        for source in predictions:
            if source in targets:
                config = self.source_configs.get(source, {'weight': 1.0, 'loss_fn': F.l1_loss})
                mask = masks.get(source, torch.ones_like(targets[source]))
                
                # Apply the mask and compute reconstruction loss
                pred_masked = predictions[source] * mask
                target_masked = targets[source] * mask
                
                if config['loss_fn'] == F.cross_entropy:
                    # For categorical sources 
                    loss = config['loss_fn'](pred_masked, target_masked.long())
                else:
                    # For continuous sources
                    loss = config['loss_fn'](pred_masked, target_masked)

                # weight the loss by source
                total_loss += config['weight'] * loss
        
        return total_loss
    
    def batch_uniformity_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute batch uniformity objective (Equation 4) --> objective: to have the embeddings be uniformly distributed.
        Takes the embeddings, rotates & shuffles them across the batch and then minimizes the absolute dot product between matched pairs
        """
        # embeddings: (B, H, W, D) or (B, T, H, W, D); flatten to N vectors in D
        x = embeddings
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = rearrange(x, 'b t h w d -> (b t h w) d')
        elif x.dim() == 4:
            B, H, W, D = x.shape
            x = rearrange(x, 'b h w d -> (b h w) d')
        else:
            # (N, D)
            pass

        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        # Rotate (roll) sample pairs to approximate u' in the paper
        x_prime = torch.roll(x, shifts=1, dims=0)
        dots = (x * x_prime).sum(dim=-1).abs()  # |u · u'|
        return dots.mean()

    
    def consistency_loss(self, teacher_embeddings: torch.Tensor, 
                        student_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute teacher-student consistency loss (Equation 5)."""
        # 1 - mu · mu_s over 2, averaged over all pixels
        mu = torch.nn.functional.normalize(teacher_embeddings, p=2, dim=-1)
        mu_s = torch.nn.functional.normalize(student_embeddings, p=2, dim=-1)
        dots = (mu * mu_s).sum(dim=-1)
        return ((1.0 - dots) * 0.5).mean()
    
    def clip_loss(self, image_embeddings: torch.Tensor, 
                  text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute CLIP-style contrastive loss."""
        # Expect (B, D) vs (B, D)
        img = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        txt = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        logits = img @ txt.t()  # (B, B)
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
        """
        AEF 官方 embedding 蒸馏损失。
        
        pred_embeddings: (B, H, W, D) 模型输出的空间 embedding
        target_embeddings: (B, D, 128, 128) 预计算的官方 AEF embedding
        valid_mask: (B,) bool 表示哪些样本有官方 embedding
        
        目标：将 pred_embeddings 对齐到官方 embedding 的方向和结构，
        防止 uniformity 坍缩（官方 embedding 已确认是均匀分布的）。
        """
        if pred_embeddings is None or target_embeddings is None:
            return torch.tensor(0.0, device=pred_embeddings.device if pred_embeddings is not None else 'cpu')
        
        B, H, W, D_pred = pred_embeddings.shape
        B_t, D_t, H_t, W_t = target_embeddings.shape
        
        # 官方 embedding 空间分辨率可能不同，下采样到模型输出分辨率
        if H_t != H or W_t != W:
            target_2d = target_embeddings  # (B, D, H_t, W_t)
            target_2d = F.interpolate(target_2d, size=(H, W), mode='bilinear', align_corners=False)
        else:
            target_2d = target_embeddings
        
        # 转为 (B, H, W, D)
        target = rearrange(target_2d, 'b c h w -> b h w c')
        
        # L2 归一化后计算余弦距离 (1 - cosine_similarity)
        pred_n = torch.nn.functional.normalize(pred_embeddings, p=2, dim=-1)
        tgt_n = torch.nn.functional.normalize(target, p=2, dim=-1)
        cosine_sim = (pred_n * tgt_n).sum(dim=-1)  # (B, H, W)
        
        loss = (1.0 - cosine_sim)
        
        if valid_mask is not None:
            # 只对有效样本求平均
            valid_mask_2d = valid_mask.view(B, 1, 1).expand(B, H, W)
            loss = (loss * valid_mask_2d.float()).sum() / (valid_mask_2d.float().sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def __call__(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute total loss following Equation 3."""
        
        losses = {}
        
        if 'predictions' in outputs and 'targets' in outputs:
            recon_loss = self.reconstruction_loss(
                outputs['predictions'],
                outputs['targets'],
                outputs.get('masks', {})
            )
            losses['reconstruction'] = recon_loss
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=next(iter(outputs.values())).device if outputs else 'cpu')
        
        if 'embeddings' in outputs:
            uniformity_loss = self.batch_uniformity_loss(outputs['embeddings'])
            losses['uniformity'] = uniformity_loss
        else:
            losses['uniformity'] = torch.tensor(0.0, device=next(iter(outputs.values())).device if outputs else 'cpu')

        if 'teacher_embeddings' in outputs and 'student_embeddings' in outputs:
            consistency_loss = self.consistency_loss(
                outputs['teacher_embeddings'],
                outputs['student_embeddings']
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        if 'image_embeddings' in outputs and 'text_embeddings' in outputs:
            clip_loss = self.clip_loss(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
            losses['clip'] = clip_loss
        else:
            losses['clip'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        # AEF 官方 embedding 蒸馏损失
        if 'aef_embedding_pred' in outputs and 'aef_embedding_target' in outputs:
            distill_loss = self.aef_distill_loss(
                outputs['aef_embedding_pred'],
                outputs['aef_embedding_target'],
                outputs.get('aef_embedding_valid'),
            )
            losses['distill'] = distill_loss
        else:
            losses['distill'] = torch.tensor(0.0, device=losses['reconstruction'].device)
        
        total_loss = (
            self.reconstruction_weight * losses['reconstruction'] +
            self.uniformity_weight * losses['uniformity'] +
            self.consistency_weight * losses['consistency'] +
            self.text_weight * losses['clip'] +
            self.distill_weight * losses['distill']
        )
        
        losses['total'] = total_loss
        
        return losses
