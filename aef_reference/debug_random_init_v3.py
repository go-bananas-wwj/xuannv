"""定位条带来源：检查 encoder 输出、summarizer 输出"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from src.aef.architecture.aef_module import AlphaEarthFoundations

torch.manual_seed(42)

model = AlphaEarthFoundations(
    input_sources={"s1": 2, "s2": 6, "tianyi_sar": 1, "landsat": 6, "planet": 4},
    decode_sources={"dem": 1, "worldcover": 11, "dynamic_world": 9, "jrc_water": 1},
    enable_text_align=False,
).npu()
model.eval()

B, H, W = 2, 128, 128
x = {
    "s1": torch.randn(B, 4, H, W, 2).npu(),
    "s2": torch.randn(B, 4, H, W, 6).npu(),
    "tianyi_sar": torch.randn(B, 4, H, W, 1).npu(),
    "landsat": torch.randn(B, 4, H, W, 6).npu(),
    "planet": torch.randn(B, 4, H, W, 4).npu(),
}
ts = {k: torch.randn(B, 4).npu() for k in x.keys()}
vp = torch.tensor([[0.0, 1.0] for _ in range(B)], dtype=torch.float32).npu()

first_src = next(iter(model.input_sources.keys()))
x_stacked = model._stack_inputs(x)
ts_first = ts[first_src]

with torch.no_grad():
    # 1. Encoder output
    feats = model.encoder(x_stacked, ts_first)  # (B, T, H, W, d_p)
    
    # 2. Summarizer output
    mu = model.summarizer(feats, ts_first, vp)  # (B, H, W, 64)

def analyze(name, tensor):
    arr = tensor.cpu().numpy()
    B, *rest, D = arr.shape
    if len(rest) == 2:  # (B, H, W, D)
        H, W = rest
        for bi in range(min(B, 2)):
            y_grad = np.abs(arr[bi, 1:, :, :] - arr[bi, :-1, :, :]).mean()
            y_corrs = []
            for c in range(D):
                for yi in range(H - 1):
                    c1 = np.corrcoef(arr[bi, yi, :, c], arr[bi, yi+1, :, c])[0, 1]
                    if not np.isnan(c1):
                        y_corrs.append(c1)
            flat = arr[bi].reshape(H * W, D)
            cc = np.corrcoef(flat.T)
            off_diag = cc[np.triu_indices(D, k=1)]
            print(f"{name} [sample {bi}]: y_grad={y_grad:.4f}, y_corr={np.mean(y_corrs):.4f}, ch_corr={np.mean(off_diag):.4f}")
    elif len(rest) == 3:  # (B, T, H, W, D)
        T, H, W = rest
        # Average over T
        arr_t = arr.mean(axis=1)
        for bi in range(min(B, 2)):
            y_grad = np.abs(arr_t[bi, 1:, :, :] - arr_t[bi, :-1, :, :]).mean()
            y_corrs = []
            for c in range(D):
                for yi in range(H - 1):
                    c1 = np.corrcoef(arr_t[bi, yi, :, c], arr_t[bi, yi+1, :, c])[0, 1]
                    if not np.isnan(c1):
                        y_corrs.append(c1)
            flat = arr_t[bi].reshape(H * W, D)
            cc = np.corrcoef(flat.T)
            off_diag = cc[np.triu_indices(D, k=1)]
            print(f"{name} [sample {bi}]: y_grad={y_grad:.4f}, y_corr={np.mean(y_corrs):.4f}, ch_corr={np.mean(off_diag):.4f}")

analyze("encoder_output", feats)
analyze("summarizer_output", mu)
