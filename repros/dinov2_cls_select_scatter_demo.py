"""
Minimal repro: DINOv2 CLS token backward creates 99.93% zeros via select_scatter.

The forward does x[:, 0] to extract CLS token.
The backward creates full(0, [B, seq, hidden]) + select_scatter(grad, dim=1, index=0).
Then LayerNorm backward operates on this mostly-zero tensor, wasting compute.

Run:
  python repros/dinov2_cls_select_scatter_demo.py
"""
import torch
import torch.nn as nn
import time

B = 128
SEQ = 1370  # 14*14 patches + 1 CLS
HIDDEN = 768

class MinimalViTHead(nn.Module):
    """Minimal ViT: LayerNorm -> select CLS -> Linear head"""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, 1000)

    def forward(self, x):
        x = self.norm(x)       # [B, 1370, 768] — full sequence
        x = x[:, 0]           # [B, 768] — CLS token only (THIS creates the problem in backward)
        x = self.head(x)      # [B, 1000]
        return x

model = MinimalViTHead().cuda()
x = torch.randn(B, SEQ, HIDDEN, device='cuda', requires_grad=True)

# Forward
out = model(x)
loss = out.sum()

# Backward — this triggers the select_scatter pattern
loss.backward()

# Now show what torch.compile does with this:
print("=== Compiled backward analysis ===")
model2 = MinimalViTHead().cuda()
x2 = torch.randn(B, SEQ, HIDDEN, device='cuda', requires_grad=True)

compiled = torch.compile(model2)
out2 = compiled(x2)
loss2 = out2.sum()
loss2.backward()

# Benchmark the backward
from triton.testing import do_bench

def bench_backward():
    x3 = torch.randn(B, SEQ, HIDDEN, device='cuda', requires_grad=True)
    out3 = compiled(x3)
    loss3 = out3.sum()
    loss3.backward()

# Can't easily benchmark just backward with do_bench, so show the issue differently:
print(f"\nThe issue: select_scatter creates [{B}, {SEQ}, {HIDDEN}] = {B*SEQ*HIDDEN*4/1e6:.1f} MB")
print(f"But only [{B}, 1, {HIDDEN}] = {B*1*HIDDEN*4/1e6:.3f} MB has non-zero values")
print(f"Sparsity: {(SEQ-1)/SEQ*100:.2f}% zeros")
print(f"All downstream ops (mul, sub, sum) waste {(SEQ-1)/SEQ*100:.1f}% of compute")
print(f"\nOur select_scatter_sparsity_pass fixes this by only computing on the CLS row.")
