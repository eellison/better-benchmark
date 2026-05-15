"""Quick regression check for DEFAULT persistent threshold change."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")

import torch
import torch._inductor.config as cfg
from triton.testing import do_bench

cfg.force_disable_caches = True

results = []

def bench(label, fn, *args):
    compiled = torch.compile(fn)
    with torch.no_grad():
        compiled(*args)
        compiled(*args)
    ms = do_bench(lambda: compiled(*args))
    us = ms * 1000
    results.append((label, us))
    print(f"  {label:50s}  {us:7.1f}us")

print("Running benchmarks...")

# 1. The target: 513-dim max (should improve)
bmm = torch.randn(256, 512, 512, dtype=torch.float32, device='cuda')
sinks = torch.randn(64, dtype=torch.float32, device='cuda')
def fn_513max(bmm, sinks):
    x = bmm.reshape(4, 64, 512, 512) * 0.125
    seq = torch.arange(512, device='cuda')
    mask = seq.unsqueeze(1) >= seq.unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(4, 1, 512, 512)
    x = x + torch.where(mask, 0.0, -3.4e38)
    s = sinks.reshape(1, 64, 1, 1).expand(4, 64, 512, 1)
    combined = torch.cat([x, s], dim=-1)
    return combined.max(dim=-1, keepdim=True).values
bench("513-dim max (gpt-oss sink)", fn_513max, bmm, sinks)

# 2. GELU + var_mean (should stay same — already INNER hint)
x1 = torch.randn(16384, 768, dtype=torch.float32, device='cuda')
def fn_varmean(x):
    h = torch.nn.functional.gelu(x, approximate='tanh')
    return torch.var_mean(h, dim=-1, correction=0, keepdim=True)
bench("GELU+var_mean (16384x768)", fn_varmean, x1)

# 3. Softmax (common reduction, should be fine)
x2 = torch.randn(256, 512, 512, dtype=torch.float32, device='cuda')
def fn_softmax(x):
    return torch.softmax(x, dim=-1)
bench("softmax (256x512x512)", fn_softmax, x2)

# 4. LayerNorm (should stay same)
x3 = torch.randn(32, 512, 768, dtype=torch.float32, device='cuda')
ln = torch.nn.LayerNorm(768).cuda()
def fn_ln(x):
    return ln(x)
bench("LayerNorm (32x512x768)", fn_ln, x3)

# 5. Large reduction (should stay same — already over threshold)
x4 = torch.randn(256, 4096, dtype=torch.float32, device='cuda')
def fn_large_red(x):
    return x.sum(dim=-1)
bench("sum (256x4096)", fn_large_red, x4)

# 6. Small pointwise (no reduction, unaffected)
x5 = torch.randn(32, 512, 768, dtype=torch.float32, device='cuda')
def fn_pw(x):
    return torch.nn.functional.silu(x) * x
bench("SiLU*x pointwise (32x512x768)", fn_pw, x5)

print("\nDone. Check for regressions against baseline.")
