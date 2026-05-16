"""Benchmark the split on pattern 4 (the one that fires)."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import time
cfg.force_disable_caches = True

def bench(fn, inputs, warmup=50, iters=500):
    with torch.no_grad():
        for _ in range(warmup): fn(*inputs)
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

def cat_reshape_output(rope_q, proj_q):
    cat = torch.cat([rope_q, proj_q], dim=-1)
    return cat.reshape(-1, 128)

rope_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
proj_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
ref = cat_reshape_output(rope_q, proj_q)

# Baseline (pass off)
torch._dynamo.reset()
cfg.split_cat_pointwise_pass = False
compiled_base = torch.compile(cat_reshape_output, dynamic=True)
with torch.no_grad():
    out_base = compiled_base(rope_q, proj_q)
t_base = bench(compiled_base, [rope_q, proj_q])

# With pass
torch._dynamo.reset()
cfg.split_cat_pointwise_pass = True
compiled_split = torch.compile(cat_reshape_output, dynamic=True)
with torch.no_grad():
    out_split = compiled_split(rope_q, proj_q)
t_split = bench(compiled_split, [rope_q, proj_q])

max_diff = (out_base - out_split).abs().max().item()
print(f"Pattern 4 (cat_4, dynamic): baseline={t_base:.1f}us split={t_split:.1f}us speedup={t_base/t_split:.2f}x diff={max_diff:.6f}")
print(f"  Shape: [128, 64, 64] + [128, 64, 64] -> [128, 64, 128] -> reshape [-1, 128]")
