"""Task Head 推理引擎 — 加载并应用训练好的 ChangeDetectionHead + PrototypeFewShotHead."""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.heads import ChangeDetectionHead, PrototypeFewShotHead


class TaskHeadEngine:
    """全局单例管理训练好的 task heads."""

    _instance: Optional["TaskHeadEngine"] = None
    _cd_head: Optional[ChangeDetectionHead] = None
    _proto_head: Optional[PrototypeFewShotHead] = None
    _device: torch.device = torch.device("cpu")
    _loaded: bool = False

    @classmethod
    def get_instance(cls, device: torch.device | str = "npu:0") -> "TaskHeadEngine":
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def __init__(self, device: torch.device | str = "npu:0"):
        self.device = torch.device(device if torch.npu.is_available() else "cpu")
        self._load()

    def _load(self) -> None:
        """尝试从输出目录加载训练好的 heads."""
        if TaskHeadEngine._loaded:
            return
        head_path = Path("/workspace/outputs/aef_qwen_v2_taskheads/task_heads.pt")
        if not head_path.exists():
            print(f"[TaskHeadEngine] No trained heads found at {head_path}")
            TaskHeadEngine._loaded = True
            return

        try:
            ckpt = torch.load(head_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {"embedding_dim": 128, "hidden_dim": 64, "num_classes": 2, "temperature": 10.0})

            if "cd_head" in ckpt:
                cd = ChangeDetectionHead(cfg["embedding_dim"], cfg["hidden_dim"]).to(self.device)
                cd.load_state_dict(ckpt["cd_head"])
                cd.eval()
                TaskHeadEngine._cd_head = cd
                print(f"[TaskHeadEngine] Loaded CD head from {head_path}")

            if "proto_head" in ckpt:
                proto = PrototypeFewShotHead(
                    cfg["embedding_dim"],
                    num_classes=cfg.get("num_classes", 2),
                    hidden_dim=cfg["hidden_dim"],
                    temperature=cfg.get("temperature", 10.0),
                ).to(self.device)
                proto.load_state_dict(ckpt["proto_head"])
                proto.eval()
                TaskHeadEngine._proto_head = proto
                print(f"[TaskHeadEngine] Loaded Proto head from {head_path}")

            TaskHeadEngine._device = self.device
            TaskHeadEngine._loaded = True
        except Exception as e:
            print(f"[TaskHeadEngine] Failed to load heads: {e}")
            TaskHeadEngine._loaded = True

    @property
    def has_cd_head(self) -> bool:
        return TaskHeadEngine._cd_head is not None

    @property
    def has_proto_head(self) -> bool:
        return TaskHeadEngine._proto_head is not None

    def predict_change(
        self,
        emb_before: np.ndarray,
        emb_after: np.ndarray,
    ) -> Optional[np.ndarray]:
        """输入 before/after embedding [D, H, W]，输出变化概率 [H, W]."""
        head = TaskHeadEngine._cd_head
        if head is None:
            return None

        with torch.no_grad():
            eb = torch.from_numpy(emb_before).unsqueeze(0).float().to(TaskHeadEngine._device)
            ea = torch.from_numpy(emb_after).unsqueeze(0).float().to(TaskHeadEngine._device)
            logits = head(eb, ea)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return prob

    def predict_change_proto(
        self,
        emb_before: np.ndarray,
        emb_after: np.ndarray,
    ) -> Optional[np.ndarray]:
        """使用 PrototypeFewShotHead 输出变化概率 [H, W]."""
        head = TaskHeadEngine._proto_head
        if head is None:
            return None

        with torch.no_grad():
            eb = torch.from_numpy(emb_before).unsqueeze(0).float().to(TaskHeadEngine._device)
            ea = torch.from_numpy(emb_after).unsqueeze(0).float().to(TaskHeadEngine._device)
            logits = head(eb, ea)  # [1, 2, H, W]
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        return prob
