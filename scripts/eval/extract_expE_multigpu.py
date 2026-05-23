#!/usr/bin/env python3
"""多卡并行提取 ExpE embedding — 7 NPU 同时工作."""
import sys, os, subprocess, time
sys.path.insert(0, "/workspace/xuannv")

# 先获取 patch 列表
from src.config import load_config
from src.data.dataset import HarbinPatchDataset

cfg = load_config("configs/xuannv_v2_expE_pure_recon.yaml")
cfg.data.preload = False
dataset = HarbinPatchDataset(cfg)
all_patches = dataset.patches  # 424 patches

# 分成 7 份
NUM_GPU = 7
chunk_size = (len(all_patches) + NUM_GPU - 1) // NUM_GPU
chunks = [all_patches[i*chunk_size:(i+1)*chunk_size] for i in range(NUM_GPU)]

output_dir = "/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/eval/embeddings_all_months"
os.makedirs(output_dir, exist_ok=True)

print(f"总 patches: {len(all_patches)}")
print(f"GPU 数: {NUM_GPU}")
for i, chunk in enumerate(chunks):
    print(f"  NPU {i}: patches {chunk[0]} ~ {chunk[-1]} ({len(chunk)} 个)")

# 启动 7 个并行进程
processes = []
for gpu_idx, patch_chunk in enumerate(chunks):
    patch_list_str = ",".join(patch_chunk)
    log_file = f"/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/eval/extract_gpu{gpu_idx}.log"
    
    cmd = [
        "python", "scripts/eval/extract_expE_worker.py",
        "--config", "configs/xuannv_v2_expE_pure_recon.yaml",
        "--checkpoint", "/workspace/outputs/exp_v2_E_pure_recon_7card_100ep_0523/epoch_best_epoch52.pt",
        "--device", "npu:0",
        "--batch-size", "4",
        "--output-dir", output_dir,
        "--patches", patch_list_str,
    ]
    
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_idx)
    
    log_fp = open(log_file, "w")
    p = subprocess.Popen(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT, cwd="/workspace/xuannv")
    processes.append((gpu_idx, p, log_fp))
    print(f"[GPU {gpu_idx}] PID={p.pid}, log={log_file}")

print("\n=== 所有 GPU 已启动，等待完成 ===")
start_time = time.time()

while True:
    time.sleep(30)
    elapsed = time.time() - start_time
    
    # 检查每个进程状态
    all_done = True
    for gpu_idx, p, log_fp in processes:
        if p.poll() is None:
            all_done = False
            status = "运行中"
        else:
            status = f"已退出 (code={p.returncode})"
        
        # 统计该 GPU 完成的 patch 数
        done = len([f for f in os.listdir(output_dir) if f.startswith(f"patch_{gpu_idx:03d}") or f.startswith("patch_")])
    
    # 总体统计
    total_done = len([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    print(f"[{elapsed/60:.1f}min] 完成 {total_done}/{len(all_patches)} patches")
    
    if all_done:
        break

print(f"\n=== 全部完成，耗时 {elapsed/60:.1f} 分钟 ===")

# 关闭日志文件
for gpu_idx, p, log_fp in processes:
    log_fp.close()

# 验证
final_count = len([f for f in os.listdir(output_dir) if f.endswith(".npz")])
print(f"最终文件数: {final_count}/{len(all_patches)}")
if final_count == len(all_patches):
    print("✅ 所有 patch 提取完成！")
else:
    print(f"⚠️ 缺少 {len(all_patches) - final_count} 个 patch")
