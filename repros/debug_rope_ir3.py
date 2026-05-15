"""Compare actual codegen for different RoPE forms.

Form A: x*cos + cat(-x2, x1)*sin (Llama with full-D cos/sin) -- intermediate cat, no output cat
Form B: cat([x1*cos - x2*sin, x1*sin + x2*cos]) (two-output with half-D cos/sin) -- output cat
Form C: x*cos + cat(-x2, x1)*sin where result is ALSO sliced/catted downstream
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

# Form A: Llama-style (cat is INTERMEDIATE, consumed by mul+add)
# cos/sin are full D=128
class FormA_Llama(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x2 = x[..., D_HALF:]
        x1 = x[..., :D_HALF]
        rotated = torch.cat([-x2, x1], dim=-1)  # THIS is the intermediate cat
        return x * cos_e + rotated * sin_e  # single output, no cat at end

# Form B: two-output (cat is OUTPUT)
class FormB_TwoOutput(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., :D_HALF]
        x2 = x[..., D_HALF:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        return torch.cat([o1, o2], dim=-1)  # output cat

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos_full = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin_full = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
cos_half = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')
sin_half = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')

print("Form A: Llama x*cos + cat(-x2,x1)*sin (full-D cos/sin, intermediate cat)")
print("-" * 60)
# Test with prefer_concat_kernel_shared_reads = True
cfg.prefer_concat_kernel_shared_reads = True
m.reset(); torch._dynamo.reset()
compiled_a = torch.compile(FormA_Llama().cuda())
with torch.no_grad(): compiled_a(x, cos_full, sin_full)
nk_a = m.generated_kernel_count
t_a = bench_graph(compiled_a, [x, cos_full, sin_full])
print(f"  prefer_concat=True:  {nk_a} kernel(s), {t_a:.1f} us")

cfg.prefer_concat_kernel_shared_reads = False
m.reset(); torch._dynamo.reset()
compiled_a2 = torch.compile(FormA_Llama().cuda())
with torch.no_grad(): compiled_a2(x, cos_full, sin_full)
nk_a2 = m.generated_kernel_count
t_a2 = bench_graph(compiled_a2, [x, cos_full, sin_full])
print(f"  prefer_concat=False: {nk_a2} kernel(s), {t_a2:.1f} us")

print()
print("Form B: Two-output cat([o1,o2]) (half-D cos/sin, output cat)")
print("-" * 60)
cfg.prefer_concat_kernel_shared_reads = True
m.reset(); torch._dynamo.reset()
compiled_b = torch.compile(FormB_TwoOutput().cuda())
with torch.no_grad(): compiled_b(x, cos_half, sin_half)
nk_b = m.generated_kernel_count
t_b = bench_graph(compiled_b, [x, cos_half, sin_half])
print(f"  prefer_concat=True:  {nk_b} kernel(s), {t_b:.1f} us")

cfg.prefer_concat_kernel_shared_reads = False
m.reset(); torch._dynamo.reset()
compiled_b2 = torch.compile(FormB_TwoOutput().cuda())
with torch.no_grad(): compiled_b2(x, cos_half, sin_half)
nk_b2 = m.generated_kernel_count
t_b2 = bench_graph(compiled_b2, [x, cos_half, sin_half])
print(f"  prefer_concat=False: {nk_b2} kernel(s), {t_b2:.1f} us")

print()
print("Summary:")
print(f"  Form A best:  {min(t_a, t_a2):.1f} us (Llama intermediate cat)")
print(f"  Form B best:  {min(t_b, t_b2):.1f} us (two-output cat)")
print(f"  Ratio:        {min(t_a, t_a2)/min(t_b, t_b2):.2f}x")
