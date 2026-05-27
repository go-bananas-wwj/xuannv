"""诊断: spatial embedding_map vs global embedding_vector 的坍缩差异."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config
from src.models.model import AEFModel
from src.training.losses import batch_uniformity_loss_l2, variance_regularizer

def analyze():
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    model = AEFModel(cfg)
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    B, S, T, C, H, W = 8, 3, 8, 6, 128, 128
    frames = torch.randn(B, S, T, C, H, W, device=device)
    ts = torch.randint(0, 1000000, (B, S, T), device=device)
    frame_mask = torch.ones(B, S, T, dtype=torch.bool, device=device)
    input_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    type_ids = torch.zeros(B, S, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            source_frames=frames,
            source_timestamps_ms=ts,
            source_frame_mask=frame_mask,
            source_input_mask=input_mask,
            source_type_ids=type_ids,
            valid_start_ms=torch.zeros(B, device=device),
            valid_end_ms=torch.ones(B, device=device) * 1000000,
            target_relative_time=torch.zeros(B, 4, device=device),
            target_metadata=torch.zeros(B, 4, device=device),
        )

    emb_map = out.embedding_map       # [B, D, H, W] 空间
    emb_vec = out.embedding           # [B, D] 全局mean pool后
    pre_map = out.pre_norm_map        # [B, D, H, W] 未归一化空间
    pre_vec = out.pre_norm_embedding  # [B, D] 未归一化全局

    D = emb_vec.shape[1]
    print(f"Batch={B}, Dim={D}, Spatial={emb_map.shape[2]}x{emb_map.shape[3]}")
    print("=" * 60)

    # 1. 范数
    print("\n[1. 范数分析]")
    print(f"  emb_vec norms:  mean={emb_vec.norm(dim=1).mean():.4f} (应为1.0)")
    print(f"  pre_vec norms:  mean={pre_vec.norm(dim=1).mean():.4f} std={pre_vec.norm(dim=1).std():.4f}")
    print(f"  emb_map norms:  mean={emb_map.norm(dim=1).mean():.4f} (spatial mean)")
    print(f"  pre_map norms:  mean={pre_map.norm(dim=1).mean():.4f} std={pre_map.norm(dim=1).std():.4f}")

    # 2. Uniformity (batch_uniformity_loss_l2)
    print("\n[2. Uniformity Loss]")
    u_vec = batch_uniformity_loss_l2(emb_vec)
    u_map = batch_uniformity_loss_l2(emb_map)
    print(f"  on emb_vec [B,D]:      {u_vec.item():.4f}")
    print(f"  on emb_map [B,D,H,W]:  {u_map.item():.4f}")
    print(f"  difference:            {u_vec.item() - u_map.item():.4f}")
    if u_vec.item() > u_map.item() + 0.1:
        print("  >>> Global mean pool 显著增加了坍缩度!")

    # 3. 每维度 std
    print("\n[3. 每维度 std & active_dims]")
    std_vec = torch.sqrt(emb_vec.var(dim=0, unbiased=False) + 1e-6)
    std_map = torch.sqrt(emb_map.reshape(B, D, -1).var(dim=(0, 2), unbiased=False) + 1e-6)
    print(f"  emb_vec: std_mean={std_vec.mean():.4f} active={(std_vec>0.05).sum().item()}/{D}")
    print(f"  emb_map: std_mean={std_map.mean():.4f} active={(std_map>0.05).sum().item()}/{D}")

    # 4. Global Mean Pool 到底抹掉了多少空间信息？
    print("\n[4. Global Mean Pool 信息损失]")
    # 把 emb_map 的每个空间位置当作独立样本
    emb_pixels = emb_map.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
    std_pixels = torch.sqrt(emb_pixels.var(dim=0, unbiased=False) + 1e-6)
    print(f"  像素级 emb_map: std_mean={std_pixels.mean():.4f} active={(std_pixels>0.05).sum().item()}/{D}")
    print(f"  对比 global vec:  std_mean={std_vec.mean():.4f} active={(std_vec>0.05).sum().item()}/{D}")
    
    # 空间位置的 pairwise similarity
    sim_within_sample = []
    for b in range(min(B, 4)):
        pixels = emb_map[b].permute(1, 2, 0).reshape(-1, D)  # [H*W, D]
        sim = pixels @ pixels.T  # [H*W, H*W]
        offdiag = sim[~torch.eye(pixels.shape[0], dtype=torch.bool)]
        sim_within_sample.append(offdiag.mean().item())
    print(f"  同一样本内像素平均相似度: {sum(sim_within_sample)/len(sim_within_sample):.4f}")
    print(f"  (0=完全无关, 1=完全相同; 高值说明空间上很一致)")

    # 5. 对比: 去掉 global mean pool 会怎样
    print("\n[5. 模拟 '去掉 global mean pool' 的 uniformity]")
    # 直接用 emb_map 的 spatial mean（不重新归一化）
    raw_mean = emb_map.mean(dim=(-2, -1))  # [B, D]
    u_raw = batch_uniformity_loss_l2(raw_mean)
    print(f"  emb_map spatial mean (不额外L2): {u_raw.item():.4f}")

    # 或者用 pre_map 的 spatial mean
    pre_mean = pre_map.mean(dim=(-2, -1))
    u_pre = batch_uniformity_loss_l2(pre_mean)
    print(f"  pre_map spatial mean (不额外L2): {u_pre.item():.4f}")

    # 6. VICReg variance 在 emb_map vs emb_vec 上的差异
    print("\n[6. VICReg Variance Loss]")
    v_vec = variance_regularizer(emb_vec, min_std=1.0)
    v_map = variance_regularizer(emb_pixels, min_std=1.0)
    print(f"  on emb_vec:  {v_vec.item():.4f}")
    print(f"  on emb_map pixels: {v_map.item():.4f}")

    # 7. 如果改成 min_std=0.1 呢？
    print("\n[7. VICReg with min_std=0.1]")
    v_vec_01 = variance_regularizer(emb_vec, min_std=0.1)
    v_map_01 = variance_regularizer(emb_pixels, min_std=0.1)
    print(f"  on emb_vec:       {v_vec_01.item():.4f}")
    print(f"  on emb_map pixels:{v_map_01.item():.4f}")
    print(f"  (0=达标, >0=未达标; 0.1 对 L2 向量更合理)")

if __name__ == '__main__':
    analyze()
