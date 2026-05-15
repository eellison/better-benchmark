"""Test: what if we ALWAYS use ConcatKernel for cat on innermost dim?

Even when the cat is consumed by a pointwise op, using ConcatKernel means:
1. The cat materializes to a buffer (two writes, each D_HALF)
2. The consumer reads from that buffer (iterates full D)

Cost: extra materialization (write D + read D)
vs current pointwise_cat: no materialization, but 2x iterations with branches
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.metrics as m
import time
cfg.force_disable_caches = True

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2

def bench_graph(fn, inputs, warmup=200, iters=1000):
    for _ in range(warmup):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    for _ in range(warmup): g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

class LlamaRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x2 = x[..., D_HALF:]
        x1 = x[..., :D_HALF]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_e + rotated * sin_e

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')

# Option 1: force pointwise_cat (current behavior when no output cat heuristic)
print("Option 1: force pointwise_cat (8M iterations, branches)")
cfg.prefer_concat_kernel_shared_reads = False
cfg.force_pointwise_cat = True
m.reset(); torch._dynamo.reset()
compiled1 = torch.compile(LlamaRoPE().cuda())
with torch.no_grad(): compiled1(x, cos, sin)
nk1 = m.generated_kernel_count
t1 = bench_graph(compiled1, [x, cos, sin])
print(f"  Kernels: {nk1}, Time: {t1:.1f} us")

# Option 2: force ConcatKernel (materializes intermediate)
print("\nOption 2: force ConcatKernel (materializes cat intermediate)")
cfg.force_pointwise_cat = False
cfg.prefer_concat_kernel_shared_reads = False  
# Hack: disable pointwise_cat eligibility
cfg.max_pointwise_cat_inputs = 0  # This should prevent pointwise_cat
m.reset(); torch._dynamo.reset()
compiled2 = torch.compile(LlamaRoPE().cuda())
with torch.no_grad(): compiled2(x, cos, sin)
nk2 = m.generated_kernel_count
t2 = bench_graph(compiled2, [x, cos, sin])
print(f"  Kernels: {nk2}, Time: {t2:.1f} us")

# Option 3: pointwise_cat with output heuristic (current best)
print("\nOption 3: prefer_concat_kernel_shared_reads=True (heuristic)")
cfg.max_pointwise_cat_inputs = 64
cfg.force_pointwise_cat = False
cfg.prefer_concat_kernel_shared_reads = True
m.reset(); torch._dynamo.reset()
compiled3 = torch.compile(LlamaRoPE().cuda())
with torch.no_grad(): compiled3(x, cos, sin)
nk3 = m.generated_kernel_count
t3 = bench_graph(compiled3, [x, cos, sin])
print(f"  Kernels: {nk3}, Time: {t3:.1f} us")

print(f"\nSummary:")
print(f"  pointwise_cat (branches):    {t1:.1f} us ({nk1} kernels)")
print(f"  ConcatKernel (materialize):  {t2:.1f} us ({nk2} kernels)")
print(f"  Best heuristic:              {t3:.1f} us ({nk3} kernels)")
