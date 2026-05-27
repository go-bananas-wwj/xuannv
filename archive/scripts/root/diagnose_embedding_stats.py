"""诊断 embedding 统计特性 — 区分 pre_norm 与 L2-normalized 的差异."""
import sys, torch, torch.nn.functional as F, math
sys.path.insert(0, '/workspace/xuannv')
from src.config import load_config
from src.models.model import AEFModel

def analyze_embedding_stats():
    cfg = load_config('configs/xuannv_v12_clean.yaml')
    model = AEFModel(cfg)
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    B, S, T, C, H, W = 4, 3, 8, 6, 128, 128
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

    pre = out.pre_norm_embedding      # [B, D]  未归一化
    l2 = out.embedding                 # [B, D]  L2 归一化后

    print("=" * 60)
    print(f"Batch size: {B}, Embedding dim: {pre.shape[1]}")
    print("=" * 60)

    # 范数分析
    pre_norms = pre.norm(dim=1)
    l2_norms = l2.norm(dim=1)
    print(f"\n[范数]")
    print(f"  pre_norm:  mean={pre_norms.mean():.4f} std={pre_norms.std():.4f} min={pre_norms.min():.4f} max={pre_norms.max():.4f}")
    print(f"  L2_norm:   mean={l2_norms.mean():.4f} std={l2_norms.std():.4f} (应为 1.0)")

    # 维度 std 分析
    pre_std = torch.sqrt(pre.var(dim=0, unbiased=False) + 1e-6)
    l2_std = torch.sqrt(l2.var(dim=0, unbiased=False) + 1e-6)
    print(f"\n[每维度 std]")
    print(f"  pre_norm:  mean={pre_std.mean():.4f} min={pre_std.min():.4f} max={pre_std.max():.4f}")
    print(f"  L2_norm:   mean={l2_std.mean():.4f} min={l2_std.min():.4f} max={l2_std.max():.4f}")
    print(f"  pre_norm active (std>0.05): {(pre_std > 0.05).sum().item()}/{pre.shape[1]}")
    print(f"  L2_norm  active (std>0.05): {(l2_std > 0.05).sum().item()}/{l2.shape[1]}")

    # 模拟不同 batch size 下的 L2 active_dims
    print(f"\n[模拟不同 batch size 下 L2 embedding 的 active_dims]")
    for n in [8, 16, 32, 48, 64, 128, 256, 512]:
        if n <= 10000:
            samples = torch.randn(n, pre.shape[1], device=device)
            samples = F.normalize(samples, p=2, dim=1)
            std = torch.sqrt(samples.var(dim=0, unbiased=False) + 1e-6)
            active = (std > 0.05).sum().item()
            print(f"  random_batch={n:4d}: active={active:3d}/{pre.shape[1]}  std_mean={std.mean():.4f}")

    # 模拟 memory bank（512个随机L2向量）
    bank = torch.randn(512, pre.shape[1], device=device)
    bank = F.normalize(bank, p=2, dim=1)
    bank_std = torch.sqrt(bank.var(dim=0, unbiased=False) + 1e-6)
    bank_active = (bank_std > 0.05).sum().item()
    print(f"\n[随机 memory bank 512 个 L2 向量]")
    print(f"  active={bank_active}/{pre.shape[1]}  std_mean={bank_std.mean():.4f}")

    # 计算 batch_uniformity_loss_l2 在完全随机 vs 完全坍缩时的值
    print(f"\n[batch_uniformity_loss_l2 参考值]")
    from src.training.losses import batch_uniformity_loss_l2
    rand_emb = F.normalize(torch.randn(512, pre.shape[1], device=device), p=2, dim=1)
    collapse_emb = torch.ones(512, pre.shape[1], device=device)
    collapse_emb = F.normalize(collapse_emb, p=2, dim=1)
    rand_loss = batch_uniformity_loss_l2(rand_emb)
    col_loss = batch_uniformity_loss_l2(collapse_emb)
    print(f"  完全随机: {rand_loss.item():.4f}")
    print(f"  完全坍缩: {col_loss.item():.4f}")

    # VICReg variance 的 min_std=1.0 实际意义
    print(f"\n[VICReg variance loss 参考值]")
    from src.training.losses import variance_regularizer
    rand_var = variance_regularizer(rand_emb, min_std=1.0)
    col_var = variance_regularizer(collapse_emb, min_std=1.0)
    print(f"  随机 L2 embedding:  variance_loss={rand_var.item():.4f}")
    print(f"  坍缩 L2 embedding:  variance_loss={col_var.item():.4f}")

    # 用实际 pre_norm 数据
    print(f"\n[实际模型输出]")
    actual_l2_loss = batch_uniformity_loss_l2(l2)
    actual_var_loss = variance_regularizer(l2, min_std=1.0)
    print(f"  batch_uniformity: {actual_l2_loss.item():.4f}")
    print(f"  variance (min_std=1.0): {actual_var_loss.item():.4f}")

    # 关键验证：step 级别 vs epoch 级别的 all_pre 差异
    print(f"\n[step 级别 '混合' 数据模拟]")
    mixed = torch.cat([pre, bank], dim=0)  # 未归一化 + L2归一化
    mixed_std = torch.sqrt(mixed.var(dim=0, unbiased=False) + 1e-6)
    mixed_active = (mixed_std > 0.05).sum().item()
    print(f"  mixed (pre_norm + L2_bank): active={mixed_active}/{mixed.shape[1]}  std_mean={mixed_std.mean():.4f}")

    bank_only_std = torch.sqrt(bank.var(dim=0, unbiased=False) + 1e-6)
    bank_only_active = (bank_only_std > 0.05).sum().item()
    print(f"  bank_only (L2 only):        active={bank_only_active}/{bank.shape[1]}  std_mean={bank_only_std.mean():.4f}")

if __name__ == '__main__':
    analyze_embedding_stats()
