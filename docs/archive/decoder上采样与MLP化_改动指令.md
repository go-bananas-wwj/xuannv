# Decoder 上采样 + MLP 化 — 改动指令

## 背景

经调研 AEF 论文原文 (arXiv:2507.22291, Supplemental S2.4) 和 GitHub 最完整复现 (Brayden-Zhang/alphaearth-foundations)，发现当前代码的 Decoder 设计与论文存在两处关键偏离：

1. **目标分辨率 64×64**：论文和复现的重建目标都是**原始分辨率**（经上采样或 re-gridding）， ours 主动降采样到 64×64，导致 75% 像素信息丢失，重建任务过于简单。
2. **Conv2d Decoder**：论文明确说 "Implicit decoders were two-hidden-layer MLPs with a width of 512"，且 "a small decoder is applied at each pixel embedding"。当前使用 2 层 3×3 Conv2d，引入了感受野，与论文的逐像素独立解码不符。

## 改动目标

- **目标 1**：Decoder 输出后通过 bilinear upsample 回到 128×128，重建目标保持原始分辨率。
- **目标 2**：将 Conv2d Decoder 替换为逐像素 MLP（2 层 hidden=512，GELU + LayerNorm）。

## 影响文件

1. `src/models/decoders.py` — Decoder 架构重写
2. `src/models/model.py` — forward 中 decoder 输出后加上 upsample
3. `src/data/dataset.py` — 目标不再 resize 到 64×64

---

## 改动一：dataset.py — 目标保持 128×128

### 改前（第 1011-1013 行附近）

```python
# 3. 构建目标 — 与输入同一帧（自编码器）
target_res = H // 2  # ← 64×64
tgt_ch = max(self.reconstruction_channels, self.num_classes)
target_images = np.zeros((S_tgt, tgt_ch, target_res, target_res), dtype=np.float32)
```

### 改后

```python
# 3. 构建目标 — 与输入同一帧（自编码器），保持原始分辨率
target_res = H  # ← 128×128，与输入同分辨率
tgt_ch = max(self.reconstruction_channels, self.num_classes)
target_images = np.zeros((S_tgt, tgt_ch, target_res, target_res), dtype=np.float32)
```

### 连锁改动

搜索所有 `target_res` 的使用，确认以下地方同步改：

- `_generate_recon_mask()` 中 `target_res = self.image_size // 2` → `target_res = self.image_size`
- 所有调用 `_resize_to_target(data, target_res, ...)` 的地方保持正确（因为 target_res 变量值已变）

**验证点**：`recon_mask` 现在是 128×128，不再是 64×64。

---

## 改动二：decoders.py — Conv2d → 逐像素 MLP

### 改前（完整文件）

当前 `ContinuousDecoder` 和 `CategoricalDecoder` 都是 2 层 Conv2d：

```python
class ContinuousDecoder(nn.Module):
    def __init__(self, embedding_dim, window_code_dim, relative_time_code_dim,
                 metadata_dim, out_channels, hidden_mult=1):
        self.injector = ConditionInjector(...)
        hidden = embedding_dim * hidden_mult
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, embedding_map, window_code=None, relative_time=None, metadata=None):
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        return self.head(x)  # [B, out_ch, H, W]
```

### 改后

```python
class ContinuousDecoder(nn.Module):
    """连续值解码器 — 逐像素 MLP (对齐 AEF 论文 S2.4).
    
    论文原文: "Implicit decoders were two-hidden-layer MLPs with a width of 512."
    对每个空间位置 (h, w) 的 embedding 向量独立做 MLP 解码。
    
    输入: embedding_map [B, D, H, W]
    输出: reconstruction [B, out_ch, H, W] (与 embedding_map 同分辨率)
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_width: int = 512,  # 论文指定 width=512
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim, window_code_dim, relative_time_code_dim, metadata_dim,
        )
        # 条件维度拼接后输入 MLP
        cond_dim = window_code_dim + relative_time_code_dim + metadata_dim
        input_dim = embedding_dim + (cond_dim if cond_dim > 0 else 0)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, out_channels),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor | None = None,
        relative_time: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. 条件注入 (与原来相同)
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        # x: [B, D, H, W]
        
        # 2. 逐像素 MLP: 把 [B, D, H, W] 变成 [B*H*W, D]，过 MLP，再 reshape 回来
        B, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, D)  # [BHW, D]
        out_flat = self.mlp(x_flat)  # [BHW, out_ch]
        out = out_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, out_ch, H, W]
        return out


class CategoricalDecoder(nn.Module):
    """类别型解码器 — 同样改为逐像素 MLP."""

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_width: int = 512,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim, window_code_dim, relative_time_code_dim, metadata_dim,
        )
        cond_dim = window_code_dim + relative_time_code_dim + metadata_dim
        input_dim = embedding_dim + (cond_dim if cond_dim > 0 else 0)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, out_channels),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        B, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, D)
        out_flat = self.mlp(x_flat)
        out = out_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out
```

### 关键注意点

1. **ConditionInjector 不变**：它已经在 `embedding_map` 上做了条件加法（broadcast 到空间维度），所以 MLP 接收的 `x` 已经包含了条件信息。
2. **Dropout2d 去掉**：MLP 不需要空间 Dropout。如果确实需要正则化，可以换成 `nn.Dropout(0.1)`。
3. **hidden_width=512**：论文明确指定。如果显存不够，可以暂时用 256，但需记录。

---

## 改动三：model.py — Decoder 输出后加上上采样

### 改前（第 328-343 行附近）

```python
flat_recon = torch.zeros(B * T_tgt, max_ch, *flat_map.shape[2:], device=flat_map.device, dtype=flat_map.dtype)

if target_source_idx is not None:
    flat_src_idx = target_source_idx.reshape(B * T_tgt)
    for src_id, dec in enumerate(self.per_source_decoders):
        mask = (flat_src_idx == src_id)
        if mask.any():
            out = dec(
                flat_map[mask],
                window_code=cond_window[mask],
                relative_time=cond_reltime[mask],
                metadata=cond_meta[mask],
            )
            out_ch = self._per_source_out_channels[src_id]
            flat_recon[mask, :out_ch] = out.to(flat_recon.dtype)

# ... reshape back
reconstructions = flat_recon.reshape(B, T_tgt, max_ch, *flat_map.shape[2:])
```

### 改后

```python
# Decoder 输出分辨率 (与 embedding_map 同分辨率，即 H/2 x W/2)
dec_h, dec_w = flat_map.shape[2], flat_map.shape[3]  # 通常是 64×64

flat_recon = torch.zeros(B * T_tgt, max_ch, dec_h, dec_w, device=flat_map.device, dtype=flat_map.dtype)

if target_source_idx is not None:
    flat_src_idx = target_source_idx.reshape(B * T_tgt)
    for src_id, dec in enumerate(self.per_source_decoders):
        mask = (flat_src_idx == src_id)
        if mask.any():
            out = dec(
                flat_map[mask],
                window_code=cond_window[mask],
                relative_time=cond_reltime[mask],
                metadata=cond_meta[mask],
            )
            out_ch = self._per_source_out_channels[src_id]
            flat_recon[mask, :out_ch] = out.to(flat_recon.dtype)

# ★ 新增: bilinear upsample 回到原始分辨率 (128×128)
if dec_h != self.image_size or dec_w != self.image_size:
    flat_recon_upsampled = F.interpolate(
        flat_recon, 
        size=(self.image_size, self.image_size),
        mode='bilinear',
        align_corners=False,
    )
else:
    flat_recon_upsampled = flat_recon

reconstructions = flat_recon_upsampled.reshape(B, T_tgt, max_ch, self.image_size, self.image_size)
```

### 关键注意点

1. **upsample 只在 decoder 输出后做一次**：不要对每个源分别 upsample，等所有源都 decode 完再统一 upsample，节省计算。
2. **`self.image_size`** 需要在 model 中保存（当前 cfg.data.image_size 应该已经可访问）。
3. **skip_decoder 路径也要同步**：如果 `skip_decoder=True`（dual window 编码时），返回的 dummy reconstructions 尺寸也要变成 `(B, num_tgt, max_ch, image_size, image_size)`。

---

## 改动四：config 兼容性检查

搜索所有引用 `reconstruction_channels`、`hidden_mult` 的 config：

```bash
grep -r "hidden_mult" configs/
```

如果有 config 设置了 `hidden_mult`，需要改为 `hidden_width=512` 或删除（用默认值）。

---

## 验证方法

改完后运行 `scripts/test_v6_launch.py` 或写一个快速测试脚本：

```python
import torch
from src.models.decoders import ContinuousDecoder

dec = ContinuousDecoder(embedding_dim=64, window_code_dim=32, 
                        relative_time_code_dim=16, metadata_dim=8,
                        out_channels=6, hidden_width=512)
emb = torch.randn(2, 64, 64, 64)
wc = torch.randn(2, 32)
rt = torch.randn(2, 16)
meta = torch.randn(2, 8)
out = dec(emb, wc, rt, meta)
assert out.shape == (2, 6, 64, 64), f"Expected (2,6,64,64), got {out.shape}"
print("Decoder output shape:", out.shape)  # 应为 [2, 6, 64, 64]

# 测试 upsample
up = torch.nn.functional.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)
assert up.shape == (2, 6, 128, 128)
print("Upsampled shape:", up.shape)
```

然后运行完整 trainer 的 forward + loss 链路测试：

```bash
cd /workspace/xuannv
python scripts/test_v6_trainer.py --config configs/v14/v14_multi_baseline.yaml
```

确认：
1. 无 crash
2. `reconstructions` shape 为 `[B, T_tgt, C, 128, 128]`
3. `target_images` shape 为 `[B, T_tgt, C, 128, 128]`
4. reconstruction loss 能正常计算

---

## 预期影响

| 指标 | 预期变化 |
|------|---------|
| recon loss | 可能上升（任务变难了），但更能反映 embedding 质量 |
| 显存 | 目标从 64×64→128×128，显存占用增加约 4×（但 batch=4 应该仍可承受） |
| 参数量 | Decoder 从 Conv2d 变为 MLP，参数量可能增加（512×512 的权重矩阵） |
| 训练速度 | MLP 比 Conv2d 慢（因为有 reshape 操作），但差别不大 |

---

## 参考资料

- AEF 论文原文: arXiv:2507.22291, Supplemental S2.4
- Brayden-Zhang 复现: https://github.com/Brayden-Zhang/alphaearth-foundations
  - `src/alphaearth/architecture/decoder.py`
  - `src/alphaearth/architecture/aef_module.py` (forward 中 upsample 部分)
