#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="/workspace/xuannv/data_raw/beijing/planetscene"
DST_ROOT="/workspace/xuannv/data_raw/haidian/scenes"

if [ ! -d "$SRC_ROOT" ]; then
    echo "ERROR: Planet source not found: $SRC_ROOT"
    exit 1
fi

linked=0
skipped=0
for src_dir in "$SRC_ROOT"/patch_*; do
    [ -d "$src_dir" ] || continue
    pid=$(basename "$src_dir")
    dst_dir="$DST_ROOT/$pid/planet"
    if [ -L "$dst_dir" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    if [ -e "$dst_dir" ]; then
        echo "WARNING: $dst_dir already exists and is not a symlink; skipping"
        continue
    fi
    ln -s "$src_dir" "$dst_dir"
    linked=$((linked + 1))
done

echo "Linked $linked patches, skipped $skipped existing symlinks."
