"""短训练测试 — 单卡运行2个epoch验证."""
from __future__ import annotations

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch_npu

from haidian_recon.config import Config
from haidian_recon.training.trainer import HRETrainer

cfg = Config()
cfg.training.epochs = 2
cfg.training.save_every = 1
cfg.training.eval_every = 1
cfg.training.log_every = 10
cfg.data.batch_size = 4

trainer = HRETrainer(cfg, rank=0, world_size=1, local_rank=0)
trainer.train()
print("Short training completed!")
