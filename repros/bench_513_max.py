"""Benchmark the 513-dim max reduction (gpt-oss sink attention)."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")

import torch
import torch._inductor.config as cfg
from triton.testing import do_bench

cfg.force_disable_caches = True

bmm = torch.randn(256, 512, 512, dtype=torch.float32, device='cuda')
sinks = torch.randn(64, dtype=torch.float32, device='cuda')

def fn(bmm, sinks):
    x = bmm.reshape(4, 64, 512, 512) * 0.125
    seq = torch.arange(512, device='cuda')
    mask = seq.unsqueeze(1) >= seq.unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(4, 1, 512, 512)
    x = x + torch.where(mask, 0.0, -3.4e38)
    s = sinks.reshape(1, 64, 1, 1).expand(4, 64, 512, 1)
    combined = torch.cat([x, s], dim=-1)
    return combined.max(dim=-1, keepdim=True).values

compiled = torch.compile(fn)
with torch.no_grad():
    compiled(bmm, sinks)
    compiled(bmm, sinks)

ms = do_bench(lambda: compiled(bmm, sinks))
print(f"Time: {ms*1000:.1f}us")

# Check what kernel type was generated
from torch._inductor.choices import InductorChoices
import inspect
src = inspect.getsource(InductorChoices.should_use_persistent_reduction)
if ", 64)" in src:
    print("DEFAULT threshold: 64 (baseline)")
elif ", 1024)" in src:
    print("DEFAULT threshold: 1024 (modified)")
