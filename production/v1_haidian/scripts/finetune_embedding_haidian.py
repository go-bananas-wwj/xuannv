#!/usr/bin/env python3
"""在 Haidian 施工工地监测标注上对生产 backbone 做有监督微调.

思路：
1. 加载生产 AEFModel（多分辨率 v1）checkpoint；
2. 只使用有标注的 63 个 patch，构造 (2025-12, 2026-04) 前后时相样本对；
3. 在 embedding_map 上接一个轻量变化检测头；
4. 用 Focal + Dice 损失端到端微调 backbone + CD head；
5. 保存 checkpoint，并重新提取 embedding 供下游头评估。

用法（单卡 NPU）：
    cd production/v1_haidian
    python scripts/finetune_embedding_haidian.py \
        --model-dir model \
        --label-dir /workspace/xuannv/haidian_label/labeljson \
        --output-dir outputs/finetune_haidian_cd \
        --device npu:0 \
        --epochs 30 \
        --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

PROD_DIR = Path(__file__).resolve().parents[1]
SRC_ROOT = PROD_DIR.parents[1]
sys.path.insert(0, str(PROD_DIR))
sys.path.insert(0, str(SRC_ROOT))

from src.config import load_config
from src.data.dataset import HarbinPatchDataset
from src.models.heads import ChangeDetectionHeadV2
from src.models.model import AEFModel
from src.utils.checkpoint import load_checkpoint

from xuannv_v1.haidian_tasks import LABEL_NORMALIZE, MERGED_TASKS, load_label_json, resize_mask


FORWARD_KEYS = {
    "source_frames",
    "source_timestamps_ms",
    "source_frame_mask",
    "source_input_mask",
    "source_type_ids",
    "valid_start_ms",
    "valid_end_ms",
    "target_relative_time",
    "target_metadata",
    "target_loss_type",
    "target_source_idx",
    "target_valid_start_ms",
    "target_valid_end_ms",
    "valid_start_w2",
    "valid_end_w2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Haidian 嵌入模型有监督微调")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--label-dir", default="/workspace/xuannv/haidian_label/labeljson")
    parser.add_argument("--output-dir", default="outputs/finetune_haidian_cd")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--task", default="shigongjiandu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-head-ratio", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4, help="patch 对 batch size")
    parser.add_argument("--accum-steps", type=int, default=2, help="梯度累积 batch 数")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--val-patches", type=int, default=10, help="验证集 patch 数")
    parser.add_argument("--patience", type=int, default=10, help="验证 F1 不提升早停 epoch 数")
    parser.add_argument("--freeze-sensor-encoder", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pos-weight", type=float, default=10.0, help="正例像素 BCE 权重（<=0 则自动计算）")
    parser.add_argument("--resume-model", type=str, default=None, help="从指定 model checkpoint（含 model_state_dict）继续微调，覆盖 model-dir/epoch_80.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _discover_labeled_patches(label_dir: Path) -> list[str]:
    pids: set[str] = set()
    for f in label_dir.glob("*.json"):
        m = re.search(r"(patch_\d+)", f.name)
        if m:
            pids.add(m.group(1))
    return sorted(pids)


def _find_monthly_idx(dataset: HarbinPatchDataset, patch_id: str, year: int, month: int) -> int | None:
    for idx, (pid, y, m) in enumerate(dataset.monthly_samples):
        if pid == patch_id and y == year and m == month:
            return idx
    return None


def _batchify(batch: dict, device: torch.device) -> dict:
    """把 dataset __getitem__ 返回的 dict 转成 model.forward 接受的 batch=1 dict."""
    out: dict = {}
    for k, v in batch.items():
        if k not in FORWARD_KEYS:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.unsqueeze(0).to(device) for t in v]
    return out


def _collate_pair_batches(samples: list[dict], device: torch.device) -> tuple[dict, dict, torch.Tensor]:
    """把 B 个 (before_dict, after_dict, label) 样本拼成 batch（每个单样本已有 batch=1 维度）."""
    before_list = [s["before"] for s in samples]
    after_list = [s["after"] for s in samples]
    labels = np.stack([s["label"] for s in samples], axis=0)  # [B, H, W]
    label_t = torch.from_numpy(labels).float().to(device).unsqueeze(1)  # [B, 1, H, W]

    def _cat(key: str, src_list: list[dict]):
        first = src_list[0][key]
        if isinstance(first, torch.Tensor):
            return torch.cat([s[key] for s in src_list], dim=0).to(device)
        if isinstance(first, list):
            n_sources = len(first)
            return [
                torch.cat([s[key][i] for s in src_list], dim=0).to(device)
                for i in range(n_sources)
            ]
        raise TypeError(f"unsupported type for key {key}: {type(first)}")

    before_batch = {k: _cat(k, before_list) for k in FORWARD_KEYS}
    after_batch = {k: _cat(k, after_list) for k in FORWARD_KEYS}
    return before_batch, after_batch, label_t


def _merged_mask_for_patch(json_path: Path, task: str, size: int) -> np.ndarray | None:
    """获取某个 patch 在指定任务下的二值 mask（已 resize 到模型输出尺寸）."""
    try:
        masks = load_label_json(json_path, image_size=(427, 427))
    except Exception:
        return None
    sources = MERGED_TASKS.get(task, [task])
    first = next((masks[s] for s in sources if s in masks), None)
    if first is None:
        return None
    merged = np.zeros_like(first)
    for src in sources:
        if src in masks:
            merged |= masks[src]
    return resize_mask(merged, size)


def focal_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """Focal + Dice（二分类）."""
    weight = torch.where(targets > 0.5, pos_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * weight).mean()

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal = ((1 - p_t) ** 2 * bce).mean()

    pred = torch.sigmoid(logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = targets.view(targets.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = 1 - (2.0 * intersection + 1.0) / (union + 1.0)
    return focal + dice.mean()


def _compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """计算二分类像素级指标."""
    preds = (probs > 0.5).astype(np.uint8)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(labels.flatten(), probs.flatten()))
    except Exception:
        auc = 0.0
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall, "auc": auc}


class _PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, int, int]], dataset: HarbinPatchDataset, labels: dict[str, np.ndarray], device: torch.device):
        self.pairs = pairs
        self.dataset = dataset
        self.labels = labels
        self.device = device

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pid, dec_idx, apr_idx = self.pairs[idx]
        return {
            "before": _batchify(self.dataset[dec_idx], self.device),
            "after": _batchify(self.dataset[apr_idx], self.device),
            "label": self.labels[pid].astype(np.float32),
        }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(args.label_dir)
    if not dist.is_initialized() and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="hccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device.startswith("npu"):
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # 1. 加载生产模型
    model_dir = Path(args.model_dir)
    cfg_path = model_dir / "config_multires_v1.yaml"
    ckpt_path = Path(args.resume_model) if args.resume_model else model_dir / "epoch_80.pt"
    cfg = load_config(cfg_path)

    labeled_pids = _discover_labeled_patches(label_dir)
    print(f"[finetune] 有标注 patch 数: {len(labeled_pids)}")
    cfg.data.patch_list = labeled_pids

    dataset = HarbinPatchDataset(cfg)
    dataset.training = False
    print(f"[finetune] 月度样本数: {len(dataset.monthly_samples)}")

    model = AEFModel(cfg).to(device)
    state = load_checkpoint(ckpt_path, device=device, keys=("model_state_dict",))
    model.load_state_dict(state, strict=False)
    model.train()

    if args.freeze_sensor_encoder:
        for p in model.sensor_encoder_bank.parameters():
            p.requires_grad = False
        print("[finetune] sensor_encoder_bank 已冻结")

    # 2. 获取模型输出空间尺寸
    sample_idx = _find_monthly_idx(dataset, labeled_pids[0], 2026, 4)
    if sample_idx is None:
        sample_idx = 0
    with torch.no_grad():
        sample_batch = _batchify(dataset[sample_idx], device)
        sample_out = model(**sample_batch, skip_decoder=True)
        _, _, H, W = sample_out.embedding_map.shape
    print(f"[finetune] embedding map 空间尺寸: {H}x{W}")

    # 3. 构造时相对与 label
    all_pairs: list[tuple[str, int, int]] = []
    labels: dict[str, np.ndarray] = {}
    for pid in labeled_pids:
        json_path = label_dir / f"{pid}_20260430_rgb_uint8.json"
        if not json_path.exists():
            continue
        mask = _merged_mask_for_patch(json_path, args.task, H)
        if mask is None:
            mask = np.zeros((H, W), dtype=np.uint8)
        labels[pid] = mask

        dec_idx = _find_monthly_idx(dataset, pid, 2025, 12)
        apr_idx = _find_monthly_idx(dataset, pid, 2026, 4)
        if dec_idx is None or apr_idx is None:
            continue
        all_pairs.append((pid, dec_idx, apr_idx))

    print(f"[finetune] 可用前后时相对数: {len(all_pairs)}")
    if len(all_pairs) == 0:
        print("[error] 未找到可用的前后时相对")
        return 1

    # 训练 / 验证划分
    rng = np.random.RandomState(args.seed)
    pos_pairs = [p for p in all_pairs if labels[p[0]].sum() > 0]
    neg_pairs = [p for p in all_pairs if labels[p[0]].sum() == 0]
    rng.shuffle(pos_pairs)
    rng.shuffle(neg_pairs)
    n_val_pos = max(1, int(len(pos_pairs) * args.val_patches / len(all_pairs)))
    n_val_neg = max(0, args.val_patches - n_val_pos)
    n_val_pos = min(n_val_pos, max(1, len(pos_pairs) - 1))
    n_val_neg = min(n_val_neg, max(0, len(neg_pairs) - 1))
    val_pairs = pos_pairs[:n_val_pos] + neg_pairs[:n_val_neg]
    train_pairs = pos_pairs[n_val_pos:] + neg_pairs[n_val_neg:]
    rng.shuffle(train_pairs)
    print(f"[finetune] 训练对: {len(train_pairs)}，验证对: {len(val_pairs)}")

    # 自动正例权重
    train_pos = sum(labels[p[0]].sum() for p in train_pairs)
    train_neg = sum((labels[p[0]].size - labels[p[0]].sum()) for p in train_pairs)
    pos_ratio = train_pos / max(1, train_pos + train_neg)
    if args.pos_weight <= 0:
        pos_weight = max(1.0, (train_neg / max(1, train_pos)) * 0.5)
    else:
        pos_weight = args.pos_weight
    print(f"[finetune] 训练集正例比例: {pos_ratio:.4f}，使用 pos_weight={pos_weight:.2f}")

    # 4. CD head
    cd_head = ChangeDetectionHeadV2(
        embedding_dim=cfg.model.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    cd_head.train()
    # 用训练集正例比例初始化最终 conv bias，使初始输出接近数据分布
    with torch.no_grad():
        final_conv = cd_head.out[-1]
        if isinstance(final_conv, torch.nn.Conv2d):
            init_logit = math.log(pos_ratio / max(1e-6, 1 - pos_ratio)) if 0 < pos_ratio < 1 else -2.0
            final_conv.bias.fill_(init_logit)

    # 5. 优化器
    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": cd_head.parameters(), "lr": args.lr * args.lr_head_ratio},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    # 6. DataLoader（自定义 collate 一次性把 B 个样本拼好）
    train_dataset = _PairDataset(train_pairs, dataset, labels, device)
    val_dataset = _PairDataset(val_pairs, dataset, labels, device)

    def _collate_fn(samples: list[dict]) -> tuple[dict, dict, torch.Tensor]:
        return _collate_pair_batches(samples, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    # 7. 训练循环
    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    total_batches = len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        cd_head.train()
        epoch_losses: list[float] = []
        optimizer.zero_grad()

        for batch_idx, (before_batch, after_batch, label_t) in enumerate(train_loader):
            before_out = model(**before_batch, skip_decoder=True)
            after_out = model(**after_batch, skip_decoder=True)

            logits = cd_head(before_out.embedding_map, after_out.embedding_map)
            loss = focal_dice_loss(logits, label_t, pos_weight=pos_weight)
            loss = loss / args.accum_steps
            loss.backward()

            epoch_losses.append(loss.item() * args.accum_steps)

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == total_batches:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cd_head.parameters()), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # ---- validation ----
        val_loss = 0.0
        val_probs: list[np.ndarray] = []
        val_labels: list[np.ndarray] = []
        model.eval()
        cd_head.eval()
        with torch.no_grad():
            for before_batch, after_batch, label_t in val_loader:
                before_out = model(**before_batch, skip_decoder=True)
                after_out = model(**after_batch, skip_decoder=True)
                logits = cd_head(before_out.embedding_map, after_out.embedding_map)
                vloss = focal_dice_loss(logits, label_t, pos_weight=pos_weight)
                val_loss += vloss.item() * label_t.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()  # [B, 1, H, W]
                for i in range(probs.shape[0]):
                    val_probs.append(probs[i, 0])
                    val_labels.append(label_t[i, 0].cpu().numpy())

        val_loss /= max(1, len(val_probs))
        val_metrics = _compute_metrics(
            np.concatenate([p.flatten() for p in val_probs]),
            np.concatenate([l.flatten() for l in val_labels]),
        )

        print(
            f"[finetune] epoch {epoch+1}/{args.epochs} "
            f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_p/r={val_metrics['precision']:.3f}/{val_metrics['recall']:.3f}",
            flush=True,
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "cd_head_state_dict": cd_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[finetune] 新最佳 val_f1={best_f1:.4f}，保存 best.pt")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[finetune] 早停：{args.patience} 个 epoch 验证 F1 未提升")
            break

    # 8. 保存微调后模型权重
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    final_ckpt = {
        "epoch": best_ckpt["epoch"],
        "model_state_dict": best_ckpt["model_state_dict"],
        "val_metrics": best_ckpt.get("val_metrics", {}),
    }
    torch.save(final_ckpt, out_dir / "epoch_finetuned.pt")
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2)
    )
    print(f"[finetune] 完成，最佳 epoch={best_epoch} val_f1={best_f1:.4f}，输出: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
