#!/usr/bin/env bash
# launch_eval.sh — 标准化评估启动脚本
#
# 用法:
#   bash scripts/eval/launch_eval.sh --checkpoint /path/to/epoch_XX.pt [选项]
#
# 模式 (--mode):
#   extract   仅提取 embedding（7 卡并行）
#   knn       KNN 下游评估
#   auc       变化检测 AUC 评估
#   all       extract → knn → auc（默认）
#
# 示例:
#   # 完整评估
#   bash scripts/eval/launch_eval.sh \
#     --checkpoint /workspace/xuannv/outputs/exp_v13_0601/epoch_40.pt
#
#   # 仅 AUC（已有 embedding）
#   bash scripts/eval/launch_eval.sh \
#     --checkpoint /workspace/xuannv/outputs/exp_v13_0601/epoch_40.pt \
#     --mode auc --skip-extract

set -euo pipefail

# ── 默认值 ───────────────────────────────────────────────────────────────────
CONFIG="configs/config.yaml"
CHECKPOINT=""
DEVICE="npu:0"
MODE="all"
TOTAL_GPUS=7
OUTPUT_DIR=""
SKIP_EXTRACT=false
KNN_BACKEND="pytorch"
EMB_TYPE="normalized"

# ── 参数解析 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG="$2";       shift 2 ;;
    --checkpoint)   CHECKPOINT="$2";   shift 2 ;;
    --device)       DEVICE="$2";       shift 2 ;;
    --mode)         MODE="$2";         shift 2 ;;
    --total-gpus)   TOTAL_GPUS="$2";   shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
    --skip-extract) SKIP_EXTRACT=true; shift 1 ;;
    --knn-backend)  KNN_BACKEND="$2";  shift 2 ;;
    --emb-type)     EMB_TYPE="$2";     shift 2 ;;
    *) echo "[错误] 未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "[错误] 必须指定 --checkpoint"
  exit 1
fi

# 默认输出目录：checkpoint 所在目录的 eval/ 子目录
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(dirname "$CHECKPOINT")/eval"
fi
mkdir -p "$OUTPUT_DIR"

EMBED_NPZ="$OUTPUT_DIR/patch_embeddings.npz"

echo "========================================================"
echo "  玄女 Embedding 评估"
echo "  Config:      $CONFIG"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Device:      $DEVICE"
echo "  Mode:        $MODE"
echo "  Output:      $OUTPUT_DIR"
echo "========================================================"

# ── 阶段 1: 提取 Embedding ────────────────────────────────────────────────────
run_extract() {
  if [[ "$SKIP_EXTRACT" == "true" ]] && [[ -f "$EMBED_NPZ" ]]; then
    echo "[跳过] 已存在 embedding 文件: $EMBED_NPZ"
    return
  fi

  echo ""
  echo "[提取] 启动 $TOTAL_GPUS 卡并行 embedding 提取..."
  EMBED_NP_DIR="$OUTPUT_DIR/npy_shards"
  mkdir -p "$EMBED_NP_DIR"

  pids=()
  for i in $(seq 0 $((TOTAL_GPUS - 1))); do
    ASCEND_RT_VISIBLE_DEVICES=$i python scripts/eval/extract_embeddings.py \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$EMBED_NP_DIR" \
      --device npu:0 \
      --format npy \
      --gpu-idx "$i" \
      --total-gpus "$TOTAL_GPUS" \
      >> "$OUTPUT_DIR/extract_gpu${i}.log" 2>&1 &
    pids+=($!)
    echo "    GPU $i: PID ${pids[-1]}"
  done

  # 等待所有分片完成
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "[错误] 提取进程 $pid 失败"; exit 1; }
  done

  echo "[提取] 合并分片 → $EMBED_NPZ"
  python scripts/eval/extract_embeddings.py \
    --merge-only \
    --merge-npy-dir "$EMBED_NP_DIR" \
    --output-dir "$OUTPUT_DIR"
  echo "[提取] 完成"
}

# ── 阶段 2: KNN 评估 ───────────────────────────────────────────────────────────
run_knn() {
  if [[ ! -f "$EMBED_NPZ" ]]; then
    echo "[错误] 找不到 embedding 文件: $EMBED_NPZ"
    exit 1
  fi
  echo ""
  echo "[KNN] 启动下游 KNN 评估..."
  python scripts/eval/knn_eval.py \
    --embedding-file "$EMBED_NPZ" \
    --output-dir "$OUTPUT_DIR/knn" \
    --device "$DEVICE" \
    --backend "$KNN_BACKEND"
  echo "[KNN] 完成"
}

# ── 阶段 3: AUC 评估 ──────────────────────────────────────────────────────────
run_auc() {
  echo ""
  echo "[AUC] 启动变化检测 AUC 评估..."
  python scripts/eval/auc_eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --emb-type "$EMB_TYPE" \
    --output "$OUTPUT_DIR/auc_result.json"
  echo "[AUC] 完成"
}

# ── 执行 ─────────────────────────────────────────────────────────────────────
case "$MODE" in
  extract) run_extract ;;
  knn)     run_knn ;;
  auc)     run_auc ;;
  all)
    run_extract
    run_knn
    run_auc
    echo ""
    echo "========================================================"
    echo "  全部评估完成！结果目录: $OUTPUT_DIR"
    echo "========================================================"
    ;;
  *) echo "[错误] 未知 mode: $MODE"; exit 1 ;;
esac
