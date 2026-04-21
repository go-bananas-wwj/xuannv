#!/usr/bin/env python3
"""绘制 V5 训练曲线."""
import sys
import re
from pathlib import Path

sys.path.insert(0, "/workspace/xuannv")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/train.log"
OUT_PATH = "/workspace/outputs/aef_qwen_v5_mixed_scale/eval/v5_training_curve.png"

epochs = []
totals = []
recons = []
uniforms = []
temporals = []
cls_vals = []
lrs = []

with open(LOG_PATH) as f:
    for line in f:
        if "Epoch " not in line or "total=" not in line:
            continue
        # Epoch 123/300 | total=-2.456 recon=0.132 ...
        m = re.search(r'Epoch\s+(\d+)', line)
        if not m:
            continue
        ep = int(m.group(1))
        def _extract(key):
            pat = rf'{key}=([\d\.\-]+)'
            m2 = re.search(pat, line)
            return float(m2.group(1)) if m2 else None
        epochs.append(ep)
        totals.append(_extract("total"))
        recons.append(_extract("recon"))
        uniforms.append(_extract("uniform"))
        temporals.append(_extract("temporal"))
        cls_vals.append(_extract("cls"))
        lrs.append(_extract("lr"))

print(f"Parsed {len(epochs)} epochs from {LOG_PATH}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

def _plot(ax, yvals, title, color='blue', ylim=None):
    ax.plot(epochs, yvals, color=color, linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

_plot(axes[0, 0], totals, 'Total Loss', 'purple')
_plot(axes[0, 1], recons, 'Reconstruction Loss', 'blue')
_plot(axes[0, 2], uniforms, 'Uniformity', 'green')

# temporal loss 单独处理（有很多0值）
ax = axes[1, 0]
temporal_nonzero = [(e, t) for e, t in zip(epochs, temporals) if t is not None and t > 1e-6]
if temporal_nonzero:
    es, ts = zip(*temporal_nonzero)
    ax.scatter(es, ts, color='red', s=3, alpha=0.6)
ax.set_xlabel('Epoch')
ax.set_ylabel('Temporal Loss')
ax.set_title('Temporal Loss (non-zero only)')
ax.grid(True, alpha=0.3)

_plot(axes[1, 1], cls_vals, 'Classification Loss', 'orange')
_plot(axes[1, 2], lrs, 'Learning Rate', 'brown')

plt.suptitle('V5 Mixed Scale Training Curves', fontsize=14)
plt.tight_layout()
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved training curve to {OUT_PATH}")
