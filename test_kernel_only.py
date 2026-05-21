"""
Measure actual kernel execution time (not Python overhead) using CUDA events.
"""
import torch
import torch._inductor.config as config
import torch._dynamo

config.split_reductions = False
config.force_disable_caches = True


class Model(torch.nn.Module):
    def forward(self, a, b, c):
        r = torch.ops.aten.reshape.default(a, [4, 512, 4096])
        m = torch.ops.aten.mul.Tensor(b, c)
        m2 = torch.ops.aten.mul.Tensor(r, m)
        return torch.ops.aten.sum.dim_IntList(m2, [0, 1], True)


a = torch.randn(2048, 4096, device="cuda")
b = torch.randn(4, 512, 4096, device="cuda")
c = torch.randn(4, 512, 1, device="cuda")

compiled = torch.compile(Model())
with torch.no_grad():
    compiled(a, b, c)

# Now use triton's do_bench to measure just kernel time
from triton.testing import do_bench

def fn():
    return compiled(a, b, c)

with torch.no_grad():
    ms = do_bench(fn, warmup=100, rep=500)
    print(f"Kernel time (do_bench): {ms * 1000:.1f} us")

# Also measure e2e Python time for comparison
import time
with torch.no_grad():
    for _ in range(50):
        compiled(a, b, c)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(500):
        compiled(a, b, c)
    torch.cuda.synchronize()
    e2e = (time.perf_counter() - start) / 500 * 1e6
    print(f"E2E Python time:        {e2e:.1f} us")
    print(f"Python overhead:        {e2e - ms * 1000:.1f} us")
