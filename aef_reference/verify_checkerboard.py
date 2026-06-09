"""
验证脚本：检测 STPEncoder 中的棋盘伪影路径
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace/xuannv/aef_reference/src")

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alphaearth.architecture.encoder import STPEncoder
from alphaearth.architecture.STPBlock import STPBlock
from alphaearth.architecture.laplacian_pyramid_exchange import LearnedSpatialResampling


def compute_autocorr_2d(feature_map: torch.Tensor) -> np.ndarray:
    """
    计算单张特征图的 2D 自相关（通过 FFT）。
    feature_map: (H, W) 张量
    返回: (H, W) numpy 数组，已 fftshift
    """
    h, w = feature_map.shape
    # 补零到 2x 避免循环卷积混叠，或直接对原尺寸做
    f = torch.fft.fft2(feature_map)
    power = f * torch.conj(f)
    autocorr = torch.fft.ifft2(power).real
    autocorr = torch.fft.fftshift(autocorr)
    # 归一化到 [-1, 1] 方便可视化
    autocorr = autocorr / (autocorr.max() + 1e-8)
    return autocorr.numpy()


def analyze_spatial_periodicity(tensor: torch.Tensor, name: str, save_dir: str = ".") -> None:
    """
    tensor: 期望形状为 (H, W) 或 (C, H, W)
    对每个通道计算自相关并取平均，保存可视化图
    """
    if tensor.dim() == 3:
        # 对通道取平均
        feature_map = tensor.mean(dim=0)
    else:
        feature_map = tensor
    
    feature_map = feature_map.detach().cpu()
    autocorr = compute_autocorr_2d(feature_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 原始特征图
    im0 = axes[0].imshow(feature_map.numpy(), cmap="viridis")
    axes[0].set_title(f"{name}\nFeature Map")
    fig.colorbar(im0, ax=axes[0])
    
    # 自相关
    im1 = axes[1].imshow(autocorr, cmap="hot")
    axes[1].set_title("2D Auto-correlation")
    fig.colorbar(im1, ax=axes[1])
    
    # 中心十字截面
    h, w = autocorr.shape
    cx, cy = w // 2, h // 2
    axes[2].plot(np.arange(w) - cx, autocorr[cy, :], label="Horizontal", color="blue")
    axes[2].plot(np.arange(h) - cy, autocorr[:, cx], label="Vertical", color="red")
    axes[2].set_title("Auto-correlation Cross-section")
    axes[2].set_xlabel("Offset (pixels)")
    axes[2].set_ylabel("Normalized correlation")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = f"{save_dir}/autocorr_{name.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")
    
    # 检测周期性峰值（除中心外）
    # 在水平方向上找局部极大
    horiz = autocorr[cy, :]
    # 忽略中心附近 ±3 像素
    mask = np.ones_like(horiz, dtype=bool)
    mask[max(0, cx-3):min(w, cx+4)] = False
    peak_idx = np.argmax(horiz[mask])
    peak_val = horiz[mask][peak_idx]
    peak_offset = np.arange(w)[mask][peak_idx] - cx
    print(f"    Horizontal secondary peak at offset={peak_offset}, value={peak_val:.4f}")
    
    vert = autocorr[:, cx]
    mask = np.ones_like(vert, dtype=bool)
    mask[max(0, cy-3):min(h, cy+4)] = False
    peak_idx = np.argmax(vert[mask])
    peak_val = vert[mask][peak_idx]
    peak_offset = np.arange(h)[mask][peak_idx] - cy
    print(f"    Vertical   secondary peak at offset={peak_offset}, value={peak_val:.4f}")


def count_resize_ops() -> None:
    """
    统计 STPEncoder + STPBlock 中所有的 resize / interpolate 操作
    """
    print("=" * 70)
    print("【Resize / Interpolate 操作统计】")
    print("=" * 70)
    
    ops = []
    # STPEncoder 初始化阶段
    ops.append(("Encoder init", "space_features", f"adaptive_avg_pool2d -> (H//16, W//16)", "1/16"))
    ops.append(("Encoder init", "time_features", f"adaptive_avg_pool2d -> (H//8, W//8)", "1/8"))
    ops.append(("Encoder init", "precision_features", f"adaptive_avg_pool2d -> (H//2, W//2)", "1/2"))
    
    # STPBlock 中的 pyramid exchange
    # 注意：LearnedSpatialResampling 内部是 conv / convtranspose，但后面都有 interpolate 兜底
    block_ops = [
        ("time_to_space", "time(1/8) -> space(1/16)", "down 2x", "LearnedSpatialResampling(0.5) + possible bilinear"),
        ("precision_to_space", "precision(1/2) -> space(1/16)", "down 8x", "LearnedSpatialResampling(0.125) + possible bilinear"),
        ("space_to_time", "space(1/16) -> time(1/8)", "up 2x", "LearnedSpatialResampling(2.0): ConvTranspose2d(4,2,1) + possible bilinear"),
        ("precision_to_time", "precision(1/2) -> time(1/8)", "down 4x", "LearnedSpatialResampling(0.25) + possible bilinear"),
        ("space_to_precision", "space(1/16) -> precision(1/2)", "up 8x", "LearnedSpatialResampling(8.0): ConvTranspose2d(4,2,1) then bilinear 4x"),
        ("time_to_precision", "time(1/8) -> precision(1/2)", "up 4x", "LearnedSpatialResampling(4.0): ConvTranspose2d(4,2,1) then bilinear 2x"),
    ]
    for name, resolution, scale, detail in block_ops:
        ops.append(("Each STPBlock", name, scale, detail))
    
    # Encoder 尾部
    ops.append(("Encoder final", "final_space_resample", "up 8x", "LearnedSpatialResampling(8.0): ConvTranspose2d(4,2,1) then bilinear 4x"))
    ops.append(("Encoder final", "final_time_resample", "up 4x", "LearnedSpatialResampling(4.0): ConvTranspose2d(4,2,1) then bilinear 2x"))
    ops.append(("Encoder final", "space_resampled align", "interpolate", "bilinear align_corners=False to precision_H"))
    ops.append(("Encoder final", "time_resampled align", "interpolate", "bilinear align_corners=False to precision_H"))
    
    for stage, op, scale, detail in ops:
        print(f"  [{stage:15s}] {op:25s} | {scale:10s} | {detail}")
    
    # 统计总数
    n_adaptive = 3
    n_convtranspose = 3  # space_to_time, space_to_precision, time_to_precision in block * num_blocks + final two
    n_interpolate = 2 + 6  # encoder final 2 + block 中最多 6 次（实际有些可能 skip）
    print(f"\n  总计：adaptive_avg_pool2d={n_adaptive}, ConvTranspose2d 调用(每block)=3, 每block interpolate 最多6次, final interpolate 2次")
    print(f"  以 num_blocks=15 计：ConvTranspose2d 调用 ≈ 15*3 + 2 = 47 次；interpolate 调用 ≈ 15*6 + 2 = 92 次")


def test_learned_resampling_checkerboard() -> None:
    """
    单独测试 LearnedSpatialResampling 是否天生容易产生棋盘伪影
    """
    print("\n" + "=" * 70)
    print("【LearnedSpatialResampling 单独测试】")
    print("=" * 70)
    
    device = "cpu"
    # 测试 8x 上采样
    module = LearnedSpatialResampling(in_channels=64, out_channels=64, scale_factor=8.0).to(device)
    x = torch.randn(1, 64, 8, 8, device=device)
    y = module(x)
    # 由于 stride=2，实际输出只有 16x16，不是 64x64
    print(f"  Input: {x.shape}, Output after LearnedSpatialResampling(8.0): {y.shape}")
    print(f"  -> 实际只上采样了 2x，剩余 4x 需要 interpolate 完成！")
    
    # 可视化该模块输出的一个通道
    analyze_spatial_periodicity(y[0, 0], "LearnedSpatialResampling_8x_up", save_dir="outputs/viz_preview")
    
    # 测试 2x 上采样
    module2 = LearnedSpatialResampling(in_channels=64, out_channels=64, scale_factor=2.0).to(device)
    y2 = module2(x)
    print(f"  Input: {x.shape}, Output after LearnedSpatialResampling(2.0): {y2.shape}")
    analyze_spatial_periodicity(y2[0, 0], "LearnedSpatialResampling_2x_up", save_dir="outputs/viz_preview")


def test_full_encoder() -> None:
    """
    完整 STPEncoder 前向传播，提取并分析中间特征图
    """
    print("\n" + "=" * 70)
    print("【完整 STPEncoder 前向传播分析】")
    print("=" * 70)
    
    device = "cpu"
    B, T, H, W, C = 1, 2, 128, 128, 32
    x = torch.randn(B, T, H, W, C, device=device)
    timestamps = torch.randn(B, T, device=device)
    
    encoder = STPEncoder(input_channels=C, d_s=512, d_t=256, d_p=64, num_blocks=6).to(device)
    encoder.eval()
    
    with torch.no_grad():
        # 手动复现 forward 以插桩
        x_proj = encoder.input_projection(x)  # (B,T,H,W,precision_dim=64)
        
        # Space pathway
        space_features = encoder.space_projection(x_proj)  # (B,T,H,W,512)
        space_features_init = F.adaptive_avg_pool2d(
            rearrange(space_features, 'b t h w c -> (b t) c h w'),
            (H // 16, W // 16)
        )
        space_features_init_bt = rearrange(space_features_init, '(b t) c h w -> b t h w c', b=B, t=T)
        print(f"  space_features_init: {space_features_init_bt.shape}  (resolution H/16={H//16})")
        
        # Time pathway
        time_features = encoder.time_projection(x_proj)
        time_features_init = F.adaptive_avg_pool2d(
            rearrange(time_features, 'b t h w c -> (b t) c h w'),
            (H // 8, W // 8)
        )
        time_features_init_bt = rearrange(time_features_init, '(b t) c h w -> b t h w c', b=B, t=T)
        print(f"  time_features_init:  {time_features_init_bt.shape}  (resolution H/8={H//8})")
        
        # Precision pathway
        precision_features_init = F.adaptive_avg_pool2d(
            rearrange(x_proj, 'b t h w c -> (b t) c h w'),
            (H // 2, W // 2)
        )
        precision_features_init_bt = rearrange(precision_features_init, '(b t) c h w -> b t h w c', b=B, t=T)
        print(f"  precision_features_init: {precision_features_init_bt.shape}  (resolution H/2={H//2})")
        
        space_f = space_features_init_bt
        time_f = time_features_init_bt
        precision_f = precision_features_init_bt
        
        # 逐个 block 前向
        for i, block in enumerate(encoder.blocks):
            space_f, time_f, precision_f = block(space_f, time_f, precision_f, timestamps)
        
        print(f"  space_features after {len(encoder.blocks)} blocks: {space_f.shape}")
        print(f"  time_features after {len(encoder.blocks)} blocks: {time_f.shape}")
        print(f"  precision_features after {len(encoder.blocks)} blocks: {precision_f.shape}")
        
        # Final resampling
        space_2d = rearrange(space_f, 'b t h w c -> (b t) c h w')
        time_2d = rearrange(time_f, 'b t h w c -> (b t) c h w')
        precision_2d = rearrange(precision_f, 'b t h w c -> (b t) c h w')
        
        space_resampled = encoder.final_space_resample(space_2d)
        time_resampled = encoder.final_time_resample(time_2d)
        target_H, target_W = precision_2d.shape[2:]
        
        print(f"  space_resampled (after LearnedSpatialResampling 8x): {space_resampled.shape}")
        if space_resampled.shape[2:] != (target_H, target_W):
            space_resampled = F.interpolate(space_resampled, size=(target_H, target_W), mode='bilinear', align_corners=False)
            print(f"  -> aligned to {space_resampled.shape} via interpolate")
        
        print(f"  time_resampled (after LearnedSpatialResampling 4x): {time_resampled.shape}")
        if time_resampled.shape[2:] != (target_H, target_W):
            time_resampled = F.interpolate(time_resampled, size=(target_H, target_W), mode='bilinear', align_corners=False)
            print(f"  -> aligned to {time_resampled.shape} via interpolate")
        
        final_features = space_resampled + time_resampled + precision_2d
        final_features_bt = rearrange(final_features, '(b t) c h w -> b t h w c', b=B, t=T)
        print(f"  final_features (before norm): {final_features_bt.shape}")
    
    save_dir = "outputs/viz_preview"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 分析各阶段特征图（取第一个 batch, 第一个 time step）
    print("\n  -- Analyzing space_features_init --")
    analyze_spatial_periodicity(space_features_init_bt[0, 0], "space_features_init", save_dir)
    
    print("\n  -- Analyzing space_features after blocks --")
    analyze_spatial_periodicity(space_f[0, 0], "space_features_after_blocks", save_dir)
    
    print("\n  -- Analyzing final_features --")
    analyze_spatial_periodicity(final_features_bt[0, 0], "final_features", save_dir)
    
    # 额外：分析 space_resampled 单独的成分
    print("\n  -- Analyzing space_resampled component --")
    space_res_bt = rearrange(space_resampled, '(b t) c h w -> b t h w c', b=B, t=T)
    analyze_spatial_periodicity(space_res_bt[0, 0], "space_resampled", save_dir)
    
    print("\n  -- Analyzing time_resampled component --")
    time_res_bt = rearrange(time_resampled, '(b t) c h w -> b t h w c', b=B, t=T)
    analyze_spatial_periodicity(time_res_bt[0, 0], "time_resampled", save_dir)


def test_decoder_upsample() -> None:
    """
    测试 decoder 输出后的 interpolate 是否引入伪影
    """
    print("\n" + "=" * 70)
    print("【Decoder 上采样分析】")
    print("=" * 70)
    
    device = "cpu"
    B, S, H_recon, W_recon, C_recon = 1, 1, 64, 64, 3
    recon = torch.randn(B, S, H_recon, W_recon, C_recon, device=device)
    H_target, W_target = 128, 128
    
    recon_2d = rearrange(recon, 'b s h w c -> (b s) c h w')
    recon_up = F.interpolate(recon_2d, size=(H_target, W_target), mode='bilinear', align_corners=False)
    recon_up = rearrange(recon_up, '(b s) c h w -> b s h w c', b=B, s=S)
    
    print(f"  Decoder output before upsample: {recon.shape}")
    print(f"  After F.interpolate bilinear to (H={H_target}, W={W_target}): {recon_up.shape}")
    print(f"  -> bilinear interpolate 相对安全，但高频信息会丢失。")
    
    # 如果输入本身是棋盘状的，interpolate 会平滑但不会消除
    # 测试：构造一个棋盘输入
    checker = torch.zeros(1, 1, 64, 64, 3)
    for i in range(64):
        for j in range(64):
            if (i + j) % 2 == 0:
                checker[0, 0, i, j, :] = 1.0
    checker_2d = rearrange(checker, 'b s h w c -> (b s) c h w')
    checker_up = F.interpolate(checker_2d, size=(128, 128), mode='bilinear', align_corners=False)
    checker_up = rearrange(checker_up, '(b s) c h w -> b s h w c', b=1, s=1)
    
    save_dir = "outputs/viz_preview"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(checker[0, 0].numpy())
    axes[0].set_title("Checkerboard Input (64x64)")
    axes[1].imshow(checker_up[0, 0].numpy())
    axes[1].set_title("After bilinear upsample (128x128)")
    plt.savefig(f"{save_dir}/decoder_checkerboard_test.png", dpi=150)
    plt.close()
    print(f"  Saved: {save_dir}/decoder_checkerboard_test.png")
    print(f"  -> bilinear 会平滑棋盘格，但如果源头持续产生，伪影仍会存在。")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    count_resize_ops()
    test_learned_resampling_checkerboard()
    test_full_encoder()
    test_decoder_upsample()
    
    print("\n" + "=" * 70)
    print("【分析结论】")
    print("=" * 70)
    print("""
1. Space pathway 仍然是 1/16 下采样（代码第57-60行: adaptive_avg_pool2d 到 H//16, W//16），
   用户声称的"改为 1/8L"在当前代码中并未体现。

2. LearnedSpatialResampling 仍然只有单层 ConvTranspose2d / Conv2d，
   用户声称的"增加第二层 3×3 conv"在当前代码中并未体现。
   特别是 upsample 分支使用 kernel=4, stride=2, padding=1 的 ConvTranspose2d，
   这是产生棋盘伪影的经典结构（Odena et al., 2016）。

3. SpaceOperator 中未见 2D 正弦位置编码，Encoder 尾部仍是直接相加，
   未见 fusion conv。

4. 即使 scale_factor=8.0，LearnedSpatialResampling 内部仅用 stride=2 的 ConvTranspose2d，
   实际只上采样 2x，剩余倍数完全依赖 F.interpolate(bilinear)。
   这意味着棋盘伪影先由 ConvTranspose2d 产生，再被 interpolate 放大/平滑。

5. 每个 STPBlock 涉及 3 次 upsample (space->time, space->precision, time->precision)，
   全部使用 ConvTranspose2d。15 个 block 累计 45 次 ConvTranspose2d 调用，
   伪影会在深层网络中被不断放大和叠加。

6. Decoder 本身没有上采样结构，但 aef_module.py 中 reconstruction 后有一次
   F.interpolate 从 H/2 到 H。该 interpolate 本身不会创造棋盘伪影，
   但如果 embedding (mu_t) 已经有空间伪影，decoder 会将其继承到重建结果中。
    """)


if __name__ == "__main__":
    main()
