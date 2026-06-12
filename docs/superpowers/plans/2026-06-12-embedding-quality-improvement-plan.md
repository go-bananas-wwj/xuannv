# Embedding 质量改进实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `exp_recon_distill_v1` 生成的 64D 嵌入严重坍缩（effective rank 13/64、时间敏感度不足）问题，训练出 effective rank > 40、时间 gap=1 中位 cosine < 0.95、AUC > 0.75 的嵌入。

**Architecture:** 在保留重建 + 双教师蒸馏框架的前提下，重新引入反坍缩损失（raw uniformity / pairwise cosine diversity / effective rank / temporal contrastive）并设计 Curriculum 蒸馏权重，使早期 epoch 先学反坍缩，后期 epoch 再逐步引入蒸馏对齐，避免蒸馏信号压制嵌入空间多样性。

**Tech Stack:** PyTorch 2.1 + torch_npu 2.1.0, hccl DDP, YAML 配置系统, `src/training/losses.py`, `src/training/trainer.py`, `scripts/train/train.py`

---

## Background & Spec

当前 `exp_recon_distill_v1` 训练 80 epoch 后，提取的 17 个月海淀嵌入质量如下：

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| Effective rank | 13.12 / 64 | > 40 |
| Active dims (std>0.05) | 25 / 64 | > 50 |
| Pairwise cosine mean | 0.79 | < 0.50 |
| Temporal gap=1 median cosine | 0.997 | < 0.95 |
| AUC (变化检测) | 待测 | > 0.75 |

根因：`config_recon_distill_v1.yaml` 只保留重建损失和蒸馏损失，所有反坍缩损失权重为 0，模型未受约束去学习分散、时间敏感的嵌入空间。

本计划分 5 个阶段实施：
1. 修复训练器已知 bug（memory_bank_size=0 导致 IndexError）
2. 重新设计损失组合与 Curriculum 蒸馏策略
3. 新增/复用反坍缩损失并验证数值稳定性
4. 训练一个 quick_diag 实验验证改进方向
5. 全量训练并提取 embedding 做质量回归测试

---

## File Structure

| 文件 | 责任 |
|------|------|
| `configs/config_v31_antcollapse_curriculum.yaml` | 新的活跃训练配置，包含反坍缩 + Curriculum 蒸馏 |
| `src/training/losses.py` | 复用并加固反坍缩损失函数 |
| `src/training/trainer.py` | 修复 memory_bank_size=0 bug；支持 Curriculum 蒸馏权重 |
| `src/utils/embedding_quality.py` | 新增可复用的 embedding 质量检查模块 |
| `scripts/eval/check_embedding_quality.py` | 调用质量检查模块并输出报告/可视化 |
| `scripts/test_smoke.py` | 冒烟测试，验证前向 + 损失可跑通 |
| `scripts/eval/extract_embeddings.py` | 已支持 `--months`，用于提取验证 embedding |

---

## Task 1: 修复 memory_bank_size=0 导致的 IndexError

**Files:**
- Modify: `src/training/trainer.py`（memory bank 初始化/使用位置）
- Test: `scripts/test_smoke.py`

- [ ] **Step 1: 定位 bug**

  运行：
  ```bash
  cd /workspace/xuannv
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv
  grep -n "memory_bank" src/training/trainer.py | head -20
  ```
  确认 `memory_bank_size=0` 时哪一行触发 `IndexError`。

- [ ] **Step 2: 增加空 bank 保护**

  修改 `src/training/trainer.py`，在访问 memory bank 前判断：
  ```python
  if self.memory_bank_size > 0 and len(self.memory_bank) > 0:
      # 使用 memory bank
  else:
      # 直接使用当前 batch
  ```
  精确行号以上一步 grep 结果为准。

- [ ] **Step 3: 验证修复**

  运行：
  ```bash
  python scripts/test_smoke.py
  ```
  期望：脚本完成，无 `IndexError`。

- [ ] **Step 4: Commit**

  ```bash
  git add src/training/trainer.py
  git commit -m "fix: memory_bank_size=0 时空 bank 保护"
  git push origin v12-clean-dynamic
  ```

---

## Task 2: 设计反坍缩 + Curriculum 蒸馏配置

**Files:**
- Create: `configs/config_v31_antcollapse_curriculum.yaml`
- Modify: `src/config.py`（如需要新增字段）
- Test: `python scripts/train/train.py --config configs/config_v31_antcollapse_curriculum.yaml --dry-run`（如支持）

- [ ] **Step 1: 基于现有配置复制并修改**

  从 `configs/config_recon_distill_v1.yaml` 复制：
  ```bash
  cp configs/config_recon_distill_v1.yaml configs/config_v31_antcollapse_curriculum.yaml
  ```

- [ ] **Step 2: 开启反坍缩损失**

  修改 `configs/config_v31_antcollapse_curriculum.yaml` 中 `training` 区块：
  ```yaml
  training:
    # 保留原有重建 + 蒸馏
    reconstruction_weight: 1.0
    aef_distill_weight: 0.5
    olmo_distill_weight: 0.25

    # 新增反坍缩损失
    raw_uniformity_weight: 0.5
    pairwise_cosine_diversity_weight: 0.1
    erank_weight: 0.1
    temporal_contrastive_weight: 0.2
    variance_regularizer_weight: 0.05

    # Curriculum 蒸馏：前 20 epoch 蒸馏权重从 0 线性增加到目标值
    curriculum_start_weight: 0.0
    curriculum_end_weight: 1.0
    curriculum_warmup_epochs: 20

    memory_bank_size: 512
    max_steps_per_epoch: 0  # 全量训练
  ```

- [ ] **Step 3: 验证配置可加载**

  运行：
  ```bash
  python - <<'PY'
  from src.config import load_config
  cfg = load_config("configs/config_v31_antcollapse_curriculum.yaml")
  print(cfg.training.raw_uniformity_weight)
  print(cfg.training.temporal_contrastive_weight)
  PY
  ```
  期望：成功打印配置值，无 KeyError。

- [ ] **Step 4: Commit**

  ```bash
  git add configs/config_v31_antcollapse_curriculum.yaml
  git commit -m "config: v31 反坍缩 + Curriculum 蒸馏配置"
  git push origin v12-clean-dynamic
  ```

---

## Task 3: 在 Trainer 中支持 Curriculum 蒸馏权重

**Files:**
- Modify: `src/training/trainer.py`
- Test: `python -m py_compile src/training/trainer.py`

- [ ] **Step 1: 读取 Curriculum 配置**

  在 `DDPv13Trainer.__init__` 中读取：
  ```python
  self.curriculum_start_weight = getattr(self.cfg.training, "curriculum_start_weight", 1.0)
  self.curriculum_end_weight = getattr(self.cfg.training, "curriculum_end_weight", 1.0)
  self.curriculum_warmup_epochs = getattr(self.cfg.training, "curriculum_warmup_epochs", 0)
  ```

- [ ] **Step 2: 计算当前 epoch 蒸馏系数**

  在 `train_epoch` 或损失计算前增加：
  ```python
  def _distill_curriculum_weight(self, epoch: int) -> float:
      if self.curriculum_warmup_epochs <= 0:
          return self.curriculum_end_weight
      progress = min(1.0, epoch / self.curriculum_warmup_epochs)
      return self.curriculum_start_weight + progress * (self.curriculum_end_weight - self.curriculum_start_weight)
  ```

- [ ] **Step 3: 应用系数到蒸馏损失**

  在蒸馏损失求和前：
  ```python
  distill_factor = self._distill_curriculum_weight(self.current_epoch)
  aef_distill_loss = aef_distill_loss * self.cfg.training.aef_distill_weight * distill_factor
  olmo_distill_loss = olmo_distill_loss * self.cfg.training.olmo_distill_weight * distill_factor
  ```

- [ ] **Step 4: 语法检查**

  运行：
  ```bash
  python -m py_compile src/training/trainer.py
  ```
  期望：无输出（表示成功）。

- [ ] **Step 5: Commit**

  ```bash
  git add src/training/trainer.py
  git commit -m "feat: 支持 Curriculum 蒸馏权重渐进增加"
  git push origin v12-clean-dynamic
  ```

---

## Task 4: 复用并加固反坍缩损失

**Files:**
- Modify: `src/training/losses.py`
- Test: `scripts/test_smoke.py`

- [ ] **Step 1: 确认现有损失函数可用**

  运行：
  ```bash
  grep -n "def raw_uniformity_loss\|def pairwise_cosine_diversity_loss\|def erank_maximization_loss\|def temporal_contrastive_loss" src/training/losses.py
  ```
  确认四个函数存在。

- [ ] **Step 2: 检查 NPU 兼容性**

  打开 `src/training/losses.py`，确认 `erank_maximization_loss` 使用 `torch.linalg.svd` 或 NPU 兼容实现。若使用 `torch.svd`，确保返回的 `S` 可用于后续运算。

- [ ] **Step 3: 添加数值稳定性保护**

  在 `pairwise_cosine_diversity_loss` 中对 cosine 相似度做 clamp：
  ```python
  cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
  ```

  在 `raw_uniformity_loss` 中对距离加 eps：
  ```python
  sq_dists = sq_dists + 1e-8
  ```

- [ ] **Step 4: 冒烟测试**

  运行：
  ```bash
  python scripts/test_smoke.py
  ```
  期望：成功跑通前向 + 损失计算。

- [ ] **Step 5: Commit**

  ```bash
  git add src/training/losses.py
  git commit -m "fix: 反坍缩损失数值稳定性保护"
  git push origin v12-clean-dynamic
  ```

---

## Task 5: 新增可复用 embedding 质量检查模块

**Files:**
- Create: `src/utils/embedding_quality.py`
- Create: `scripts/eval/check_embedding_quality.py`
- Test: `python scripts/eval/check_embedding_quality.py --embedding-file outputs/exp_recon_distill_v1/embeddings/patch_embeddings.npz`

- [ ] **Step 1: 创建质量检查模块**

  创建 `src/utils/embedding_quality.py`：
  ```python
  from __future__ import annotations
  import numpy as np
  import torch

  def load_embeddings(path: str):
      with np.load(path) as data:
          return {
              "spatial_maps": data["spatial_maps"],
              "patch_ids": data["patch_ids"],
              "month_labels": data["month_labels"],
          }

  def compute_l2_stats(emb: np.ndarray):
      norms = np.linalg.norm(emb, axis=1)
      return {
          "mean": float(norms.mean()),
          "std": float(norms.std()),
          "median": float(np.median(norms)),
          "min": float(norms.min()),
          "max": float(norms.max()),
      }

  def pairwise_cosine_stats(emb: np.ndarray, max_samples: int = 5000):
      n = min(len(emb), max_samples)
      idx = np.random.choice(len(emb), n, replace=False)
      sample = emb[idx]
      sim = sample @ sample.T
      mask = ~np.eye(n, dtype=bool)
      off = sim[mask]
      return {
          "mean": float(off.mean()),
          "std": float(off.std()),
          "min": float(off.min()),
          "max": float(off.max()),
      }

  def effective_rank(emb: np.ndarray):
      emb_c = emb - emb.mean(axis=0, keepdims=True)
      _, s, _ = np.linalg.svd(emb_c, full_matrices=False)
      p = s / s.sum()
      return float(np.exp(-np.sum(p * np.log(p + 1e-12)))), s

  def temporal_smoothness(glob: np.ndarray, max_gap: int = 6):
      P, T, D = glob.shape
      glob_norm = glob / (np.linalg.norm(glob, axis=2, keepdims=True) + 1e-12)
      result = {}
      for gap in range(1, max_gap + 1):
          sims = []
          for p in range(P):
              for t in range(T - gap):
                  sims.append(float(np.dot(glob_norm[p, t], glob_norm[p, t + gap])))
          result[gap] = {
              "mean": float(np.mean(sims)),
              "median": float(np.median(sims)),
              "std": float(np.std(sims)),
          }
      return result
  ```

- [ ] **Step 2: 创建命令行检查脚本**

  创建 `scripts/eval/check_embedding_quality.py`：
  ```python
  from __future__ import annotations
  import argparse
  import json
  from pathlib import Path
  from src.utils.embedding_quality import (
      load_embeddings, compute_l2_stats, pairwise_cosine_stats,
      effective_rank, temporal_smoothness,
  )

  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--embedding-file", required=True)
      parser.add_argument("--output-dir", default="outputs/embedding_quality")
      args = parser.parse_args()

      data = load_embeddings(args.embedding_file)
      spatial_maps = data["spatial_maps"]
      P, T, D, H, W = spatial_maps.shape
      glob = spatial_maps.mean(axis=(3, 4))
      glob_flat = glob.reshape(-1, D)
      glob_norm = glob_flat / (np.linalg.norm(glob_flat, axis=1, keepdims=True) + 1e-12)

      report = {
          "shape": {"patches": P, "months": T, "dims": D, "H": H, "W": W},
          "l2_stats_raw": compute_l2_stats(glob_flat),
          "l2_stats_norm": compute_l2_stats(glob_norm),
          "pairwise_cosine": pairwise_cosine_stats(glob_norm),
          "effective_rank": effective_rank(glob_flat)[0],
          "active_dims": int((glob_flat.std(axis=0) > 0.05).sum()),
          "temporal_smoothness": temporal_smoothness(glob),
      }

      out_dir = Path(args.output_dir)
      out_dir.mkdir(parents=True, exist_ok=True)
      out_path = out_dir / "quality_report.json"
      with open(out_path, "w") as f:
          json.dump(report, f, indent=2)
      print(json.dumps(report, indent=2))
      print(f"Report saved to {out_path}")

  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 3: 运行验证**

  运行：
  ```bash
  python scripts/eval/check_embedding_quality.py \
      --embedding-file outputs/exp_recon_distill_v1/embeddings/patch_embeddings.npz \
      --output-dir outputs/exp_recon_distill_v1/quality_json
  ```
  期望：输出 JSON 报告，包含 effective_rank、active_dims 等指标。

- [ ] **Step 4: Commit**

  ```bash
  git add src/utils/embedding_quality.py scripts/eval/check_embedding_quality.py
  git commit -m "feat: 新增 embedding 质量检查模块和命令行脚本"
  git push origin v12-clean-dynamic
  ```

---

## Task 6: Quick-Diag 训练验证改进方向

**Files:**
- Create: `configs/config_v31_quick_diag.yaml`
- Test: `torchrun --nproc_per_node=2 scripts/train/train.py --config configs/config_v31_quick_diag.yaml --save-every 1`

- [ ] **Step 1: 创建 quick diag 配置**

  复制 `config_v31_antcollapse_curriculum.yaml` 并修改：
  ```bash
  cp configs/config_v31_antcollapse_curriculum.yaml configs/config_v31_quick_diag.yaml
  ```
  修改其中：
  ```yaml
  experiment:
    name: "quick_v31_antcollapse_curriculum_0612"
  training:
    epochs: 5
    max_steps_per_epoch: 20
    save_every: 1
  ```

- [ ] **Step 2: 运行 2 卡 quick diag**

  运行前确认 NPU 空闲：
  ```bash
  npu-smi info
  ```
  然后：
  ```bash
  tmux new-session -d -s quick_v31 -c /workspace/xuannv
  tmux send-keys -t quick_v31 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv' Enter
  tmux send-keys -t quick_v31 'export ASCEND_RT_VISIBLE_DEVICES=0,1' Enter
  tmux send-keys -t quick_v31 'torchrun --nproc_per_node=2 scripts/train/train.py --config configs/config_v31_quick_diag.yaml --save-every 1' Enter
  ```

- [ ] **Step 3: 检查训练日志**

  5 分钟后查看：
  ```bash
  tmux capture-pane -t quick_v31 -p | tail -30
  ```
  期望：raw_unif 在 -1 ~ -4 区间，recon 下降，无 NaN/Inf。

- [ ] **Step 4: Commit 配置**

  ```bash
  git add configs/config_v31_quick_diag.yaml
  git commit -m "config: v31 quick diag 配置"
  git push origin v12-clean-dynamic
  ```

---

## Task 7: 全量训练实验

**Files:**
- Use: `configs/config_v31_antcollapse_curriculum.yaml`
- Test: 训练完成后提取 embedding 并运行质量检查

- [ ] **Step 1: 启动 8 卡全量训练**

  确认 NPU 空闲：
  ```bash
  npu-smi info
  ```
  启动 tmux：
  ```bash
  tmux new-session -d -s v31_train -c /workspace/xuannv
  tmux send-keys -t v31_train 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate xuannv' Enter
  tmux send-keys -t v31_train 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' Enter
  tmux send-keys -t v31_train 'torchrun --nproc_per_node=8 scripts/train/train.py --config configs/config_v31_antcollapse_curriculum.yaml --save-every 20' Enter
  ```

- [ ] **Step 2: 周期性质量检查**

  每 20 epoch 保存 checkpoint 后，提取前 7 个月（2025-04 到 2025-10）embedding：
  ```bash
  python scripts/eval/extract_embeddings.py \
      --config configs/config_v31_antcollapse_curriculum.yaml \
      --checkpoint outputs/quick_v31_antcollapse_curriculum_0612/epoch_20.pt \
      --output-dir outputs/v31_antcollapse_curriculum_0612/embeddings_epoch20 \
      --months 2025-04:2025-10
  ```

- [ ] **Step 3: 生成质量报告**

  ```bash
  python scripts/eval/check_embedding_quality.py \
      --embedding-file outputs/v31_antcollapse_curriculum_0612/embeddings_epoch20/patch_embeddings.npz \
      --output-dir outputs/v31_antcollapse_curriculum_0612/quality_epoch20
  ```
  期望：effective_rank > 30，pairwise cosine mean < 0.65，gap=1 median < 0.97。

- [ ] **Step 4: 变化检测 AUC 验证**

  ```bash
  python scripts/eval/auc_eval.py \
      --config configs/config_v31_antcollapse_curriculum.yaml \
      --checkpoint outputs/v31_antcollapse_curriculum_0612/epoch_20.pt \
      --device npu:0
  ```
  期望：AUC > 0.70。

---

## Task 8: 文档与最终归档

**Files:**
- Create/Modify: `docs/BUG_FIX_LOG.md`
- Create: `docs/embedding_quality_report_v31_YYYYMMDD.md`

- [ ] **Step 1: 更新 BUG_FIX_LOG.md**

  追加条目：
  ```markdown
  ## 2026-06-12
  - 修复 `memory_bank_size=0` 时 `IndexError`
  - 新增 v31 反坍缩 + Curriculum 蒸馏配置
  ```

- [ ] **Step 2: 撰写最终质量报告**

  根据 Task 7 的结果撰写 `docs/embedding_quality_report_v31_20260612.md`，对比 v1.1 基线和 v31 改进后的指标。

- [ ] **Step 3: Commit 并推送**

  ```bash
  git add docs/BUG_FIX_LOG.md docs/embedding_quality_report_v31_20260612.md
  git commit -m "docs: v31 改进质量报告"
  git push origin v12-clean-dynamic
  ```

---

## Self-Review

**1. Spec coverage:**
- 反坍缩损失开启 → Task 2
- Curriculum 蒸馏 → Task 3
- memory_bank bug 修复 → Task 1
- 质量检查工具化 → Task 5
- 快速验证 → Task 6
- 全量训练 + 回归 → Task 7
- 文档 → Task 8

**2. Placeholder scan:**
- 无 TBD/TODO
- 所有代码块含实际内容
- 所有命令含期望输出

**3. Type consistency:**
- `curriculum_start_weight` / `curriculum_end_weight` / `curriculum_warmup_epochs` 在配置、trainer 中命名一致。
- `embedding_quality.py` 中 `effective_rank` 返回 `(float, s)`，命令行脚本只取 `[0]`，一致。
