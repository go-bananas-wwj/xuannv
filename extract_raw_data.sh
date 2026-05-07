#!/bin/bash
# 解压 raw_backup.tar.gz 到 /workspace/raw/harbin_scenes

set -e

TAR_FILE="/workspace/datasets/xuannv_raw_backup.tar.gz"
TARGET_DIR="/workspace/raw/harbin_scenes"

if [ ! -f "$TAR_FILE" ]; then
    echo "Error: $TAR_FILE not found"
    exit 1
fi

echo "Extracting $TAR_FILE ..."
mkdir -p "$TARGET_DIR"
tar -xzf "$TAR_FILE" -C "$TARGET_DIR" --strip-components=1 2>/dev/null || tar -xzf "$TAR_FILE" -C "$TARGET_DIR"

echo "Extraction complete. Data located at: $TARGET_DIR"
ls -la "$TARGET_DIR"
