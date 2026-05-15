"""Benchmark var_mean kernels — compare persistent vs non-persistent."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")

import torch
import torch._inductor.config as cfg
from triton.testing import do_bench

cfg.force_disable_caches = True

def bench_one(rows, rdim, label):
    x = torch.randn(rows, rdim, dtype=torch.float32, device='cuda')

    def fn(x):
        h = torch.nn.functional.gelu(x, approximate='tanh')
        return torch.var_mean(h, dim=-1, correction=0, keepdim=True)

    compiled = torch.compile(fn)
    with torch.no_grad():
        compiled(x)
        compiled(x)

    ms = do_bench(lambda: compiled(x))
    us = ms * 1000

    total_bytes = rows * rdim * 4 + 2 * (rows * 4)
    sol_us = total_bytes / (8e12) * 1e6  # B200 ~8 TB/s peak

    print(f"{label:40s}  time={us:6.1f}us  sol={sol_us:5.1f}us  gap={us/sol_us:.2f}x  bytes={total_bytes/1e6:.1f}MB")
    return us

cases = [
    (16384, 768, "GoogleFnet/Bert 16384x768"),
    (4096, 1536, "DebertaV2 4096x1536"),
    (8192, 768, "Longformer 8192x768"),
    (32768, 768, "DistilBert 32768x768"),
    (32768, 256, "Reformer 32768x256"),
    (32768, 128, "Electra 32768x128"),
    (2048, 2560, "Blenderbot 2048x2560"),
]

from torch._inductor.choices import InductorChoices
import inspect
src = inspect.getsource(InductorChoices.should_use_persistent_reduction)
if ", 64)" in src:
    print("DEFAULT persistent threshold: 64\n")
else:
    print("DEFAULT persistent threshold: (modified)\n")

for rows, rdim, label in cases:
    bench_one(rows, rdim, label)
