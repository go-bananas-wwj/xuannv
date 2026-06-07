"""最小化训练step测试 — 快速定位问题."""
from __future__ import annotations

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch_npu

from haidian_recon.config import Config
from haidian_recon.data.dataset import HaidianReconDataset, collate_fn
from haidian_recon.data.masking import FourLayerMask
from haidian_recon.models.hre_model import HREModel
from haidian_recon.losses.reconstruction import reconstruction_loss
from haidian_recon.losses.uniformity import uniformity_loss
from torch.utils.data import DataLoader

print("Loading dataset...")
dataset = HaidianReconDataset(split="train")
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)

print("Building model...")
source_channels = {s["name"]: s["channels"] for s in Config().data.sources}
model = HREModel(
    source_channels=source_channels,
    embed_dim=64,
    num_encoder_layers=2,
    num_decoder_layers=1,
    use_gradient_checkpointing=False,
)

device = torch.device("npu:0")
model = model.to(device)

masking = FourLayerMask(
    source_names=list(source_channels.keys()),
    image_size=128,
    patch_size=8,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Running one training step...")
for batch in loader:
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
    masked_batch, mask_info = masking(batch)
    output = model(masked_batch, mask_info)
    loss_recon = reconstruction_loss(output["reconstructions"], batch, mask_info)
    loss_uniform = uniformity_loss(output["embedding"])
    loss = loss_recon + 0.01 * loss_uniform
    print(f"Loss: {loss.item():.4f} recon={loss_recon.item():.4f} uniform={loss_uniform.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Step OK!")
    break

print("Done!")
