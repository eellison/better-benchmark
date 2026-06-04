"""
Minimal repro: DINOv2 CLS token backward creates 99.93% zeros via select_scatter.

Forward:  x[:, 0] extracts CLS token from [B, 1370, 768] -> [B, 768]
Backward: full(0, [B, 1370, 768]) + select_scatter(grad, dim=1, index=0)
          Then LayerNorm backward operates on the 99.93%-zero tensor wastefully.

WHY THIS IS BAD:
  - Autograd decomposes `x[:, 0]` backward as zeros + select_scatter
  - LayerNorm backward then does mul/sub/sum on ALL 1370 rows
  - Only row 0 has non-zero grad — the other 1369 rows are pure waste
  - torch.compile does NOT fix this (Inductor cannot reason about structural sparsity)

Run:
  python repros/dinov2_cls_select_scatter_demo.py

Inspect generated kernels:
  TORCH_LOGS="output_code" python repros/dinov2_cls_select_scatter_demo.py
"""
import torch
import torch.nn as nn
import os

os.environ.setdefault("TORCH_LOGS", "")

B = 128
SEQ = 1370   # DINOv2 at 518px: (518/14)^2 + 1 CLS = 1370
HIDDEN = 768


class MinimalViTHead(nn.Module):
    """LayerNorm -> select CLS -> Linear head (tail of any ViT)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, 1000)

    def forward(self, x):
        x = self.norm(x)   # [B, 1370, 768]
        x = x[:, 0]        # [B, 768] — backward creates the select_scatter problem
        x = self.head(x)   # [B, 1000]
        return x


# --- Eager backward: triggers select_scatter ---
print("=" * 60)
print("Eager backward")
print("=" * 60)

model = MinimalViTHead().cuda()
x = torch.randn(B, SEQ, HIDDEN, device='cuda', requires_grad=True)

out = model(x)
out.sum().backward()

# Show the sparsity in the resulting gradient
nonzero_per_row = (x.grad.abs().sum(dim=-1) > 0).float().mean(dim=0)
print(f"  x.grad shape: {list(x.grad.shape)}")
print(f"  Row 0 (CLS) has gradient:  {nonzero_per_row[0].item():.0f}")
print(f"  Row 1 has gradient:        {nonzero_per_row[1].item():.0f}")
print(f"  -> LayerNorm backward processed all {SEQ} rows for 1 useful row.")

# --- torch.compile: still wastes compute ---
print(f"\n{'=' * 60}")
print("torch.compile backward (still wasteful)")
print("=" * 60)

model2 = MinimalViTHead().cuda()
x2 = torch.randn(B, SEQ, HIDDEN, device='cuda', requires_grad=True)

compiled = torch.compile(model2)
compiled(x2).sum().backward()
print("  Compiled successfully — but generated kernels loop over full [B,1370,768].")

# --- Quantify ---
print(f"\n{'=' * 60}")
print("Waste summary")
print("=" * 60)

total = B * SEQ * HIDDEN
useful = B * 1 * HIDDEN
print(f"  Backward tensor: [{B}, {SEQ}, {HIDDEN}] = {total*4/1e6:.0f} MB")
print(f"  Useful data:     [{B}, 1, {HIDDEN}]    = {useful*4/1e6:.2f} MB")
print(f"  Sparsity: {(SEQ-1)/SEQ*100:.2f}% zeros")
print(f"  Every op in LayerNorm backward does {SEQ}x more work than needed.")

# --- Benchmark: full vs sparse ---
print(f"\n{'=' * 60}")
print("Benchmark: wasteful vs sparse LayerNorm backward")
print("=" * 60)

from triton.testing import do_bench

weight = torch.randn(HIDDEN, device='cuda')
full_grad = torch.zeros(B, SEQ, HIDDEN, device='cuda')
full_grad[:, 0] = torch.randn(B, HIDDEN, device='cuda')
normed_input = torch.randn(B, SEQ, HIDDEN, device='cuda')
mean = normed_input.mean(dim=-1, keepdim=True)

def wasteful():
    """Current backward: LayerNorm grad on full [B, 1370, 768]."""
    dx = full_grad * weight
    dx_sum = dx.sum(dim=-1, keepdim=True)
    dx_dot = (dx * (normed_input - mean)).sum(dim=-1, keepdim=True)
    return dx - dx_sum / HIDDEN - (normed_input - mean) * dx_dot / HIDDEN

def sparse():
    """Fixed backward: LayerNorm grad on only [B, 1, 768]."""
    g = full_grad[:, 0:1]
    inp = normed_input[:, 0:1]
    m = inp.mean(dim=-1, keepdim=True)
    dx = g * weight
    dx_sum = dx.sum(dim=-1, keepdim=True)
    dx_dot = (dx * (inp - m)).sum(dim=-1, keepdim=True)
    return dx - dx_sum / HIDDEN - (inp - m) * dx_dot / HIDDEN

t_full = do_bench(wasteful)
t_sparse = do_bench(sparse)

print(f"  Full [{B},{SEQ},{HIDDEN}]: {t_full:.3f} ms")
print(f"  CLS  [{B},1,{HIDDEN}]:    {t_sparse:.3f} ms")
print(f"  Speedup: {t_full/t_sparse:.1f}x\n")
print("select_scatter_sparsity_pass fixes this by rewriting the graph to only")
print("compute LayerNorm backward on the CLS row, then scatter the result.")
