"""检查 STPBlock 全局上下文交换是否引入条带"""
import os
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.insert(0, "/workspace/xuannv")

import torch
import torch_npu
import numpy as np
from src.aef.architecture.STPBlock import STPBlock

torch.manual_seed(42)

def y_corr(arr):
    B, T, H, W, D = arr.shape
    corrs = []
    for bi in range(B):
        for t in range(T):
            for c in range(D):
                for yi in range(H - 1):
                    c1 = np.corrcoef(arr[bi, t, yi, :, c], arr[bi, t, yi+1, :, c])[0, 1]
                    if not np.isnan(c1):
                        corrs.append(c1)
    return np.mean(corrs)

B, T = 1, 1
space = torch.randn(B, T, 16, 16, 64).npu()
time = torch.randn(B, T, 16, 16, 32).npu()
prec = torch.randn(B, T, 16, 16, 16).npu()
ts = torch.randn(B, T).npu()

block = STPBlock(space_dim=64, time_dim=32, precision_dim=16).npu()
block.eval()

print(f"Before block:")
print(f"  space y_corr: {y_corr(space.cpu().numpy()):.4f}")
print(f"  time y_corr:  {y_corr(time.cpu().numpy()):.4f}")
print(f"  prec y_corr:  {y_corr(prec.cpu().numpy()):.4f}")

with torch.no_grad():
    space_out, time_out, prec_out = block(space, time, prec, ts)

print(f"\nAfter block:")
print(f"  space y_corr: {y_corr(space_out.cpu().numpy()):.4f}")
print(f"  time y_corr:  {y_corr(time_out.cpu().numpy()):.4f}")
print(f"  prec y_corr:  {y_corr(prec_out.cpu().numpy()):.4f}")

# Now test WITHOUT global exchange
class STPBlockNoExchange(STPBlock):
    def forward(self, space_x, time_x, precision_x, timestamps):
        space_out = self.space_op(space_x)
        time_out = self.time_op(time_x, timestamps)
        precision_out = self.precision_op(precision_x)
        return space_out, time_out, precision_out

block2 = STPBlockNoExchange(space_dim=64, time_dim=32, precision_dim=16).npu()
block2.load_state_dict(block.state_dict())
block2.eval()

with torch.no_grad():
    space_out2, time_out2, prec_out2 = block2(space, time, prec, ts)

print(f"\nAfter block WITHOUT global exchange:")
print(f"  space y_corr: {y_corr(space_out2.cpu().numpy()):.4f}")
print(f"  time y_corr:  {y_corr(time_out2.cpu().numpy()):.4f}")
print(f"  prec y_corr:  {y_corr(prec_out2.cpu().numpy()):.4f}")
