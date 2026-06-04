#!/bin/bash
# 双教师蒸馏训练启动脚本 (AEF + OlmoEarth)
set -e
cd /workspace/xuannv
export PYTHONPATH=/workspace/xuannv:$PYTHONPATH
export no_proxy=*
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG="configs/config_dual_teacher_v1.yaml"

echo "========================================"
echo "双教师蒸馏训练启动"
echo "Config: $CONFIG"
echo "========================================"

AEF_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('data',{}).get('aef_embed_dir',''))")
if [ -d "$AEF_DIR" ]; then
    echo "[OK] AEF 嵌入目录: $AEF_DIR"
    echo "     patches 数量: $(ls $AEF_DIR/*.npy 2>/dev/null | wc -l)"
else
    echo "[警告] AEF 嵌入目录不存在: $AEF_DIR"
fi

OLMO_DIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('data',{}).get('teacher_embed_dir',''))")
if [ -d "$OLMO_DIR" ]; then
    echo "[OK] OlmoEarth 嵌入目录: $OLMO_DIR"
else
    echo "[警告] OlmoEarth 嵌入目录不存在: $OLMO_DIR"
fi

echo ""
echo "启动训练..."
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train/train.py \
    --config $CONFIG \
    "$@"
