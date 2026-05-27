#!/bin/bash
# Round7 8卡并行实验启动脚本
# 每个实验运行在1张NPU上

set -e

EXPERIMENTS=(
    "exp1_pre_norm_strong"
    "exp2_pre_norm_low_recon"
    "exp3_spatial_raw_unif"
    "exp4_vicreg_heavy"
    "exp5_l2_space_ctrl"
    "exp6_high_recon_ctrl"
    "exp7_strong_orth"
    "exp8_full_combo"
)

PORTS=(29601 29602 29603 29604 29605 29606 29607 29608)

cd /workspace/xuannv

for i in {0..7}; do
    exp=${EXPERIMENTS[$i]}
    port=${PORTS[$i]}
    session="r7_${exp}"
    
    # 如果session已存在，先kill
    tmux kill-session -t "$session" 2>/dev/null || true
    
    # 创建新session
    tmux new-session -d -s "$session" -c /workspace/xuannv
    
    # 先激活conda环境，再启动训练
    tmux send-keys -t "$session" "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\"" Enter
    tmux send-keys -t "$session" "conda activate xuannv" Enter
    tmux send-keys -t "$session" "export ASCEND_RT_VISIBLE_DEVICES=$i" Enter
    tmux send-keys -t "$session" "torchrun --nproc_per_node=1 --master_port=$port scripts/train/train_ddp_v12.py --config configs/round7_8gpu/${exp}.yaml --save-every 5" Enter
    
    echo "[$session] 已启动在 NPU $i, port $port"
    sleep 2
done

echo ""
echo "全部8个实验已启动！"
echo "查看日志: tmux attach -t r7_exp1_pre_norm_strong"
echo "列出所有session: tmux list-sessions | grep r7_"
