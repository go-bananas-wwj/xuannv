#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$PROD_DIR")")"

SRC_CKPT="/workspace/xuannv/outputs/exp_multires_v1_0612_backup/epoch_80.pt"
SRC_CFG="/workspace/xuannv/configs/config_multires_v1.yaml"
DEST_DIR="$PROD_DIR/model"

mkdir -p "$DEST_DIR"

if [ ! -f "$SRC_CKPT" ]; then
    echo "ERROR: 源 checkpoint 不存在: $SRC_CKPT" >&2
    exit 1
fi

cp -v "$SRC_CFG" "$DEST_DIR/config_multires_v1.yaml"
cp -v "$SRC_CKPT" "$DEST_DIR/epoch_80.pt"

{
    echo "# Checkpoint 来源记录"
    echo "source_checkpoint=$SRC_CKPT"
    echo "source_config=$SRC_CFG"
    echo "copied_at=$(date -Iseconds)"
} > "$DEST_DIR/CHECKPOINT_SOURCE"

echo "生产模型已复制到 $DEST_DIR"
