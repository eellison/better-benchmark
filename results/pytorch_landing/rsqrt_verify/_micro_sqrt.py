"""
Microbenchmark: isolate `tl.sqrt_rn(x) ; 1.0/that` vs `libdevice.rsqrt(x)`
over a memory-bound pointwise kernel of the SAME shape/access pattern as the
repvgg_a2 7ee473b0 kernel (102M elems, bf16 load/store, scale broadcast over
64-channel modulus). This tests whether the instruction substitution ALONE
explains the ~2.5x.
"""
import torch, triton, triton.language as tl
from triton.testing import do_bench

N = 102760448  # 128*64*112*112
C = 64

@triton.jit
def k_sqrt_div(out_ptr, in_ptr, scale_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x2 = xindex
    x0 = xindex % 64
    a = tl.load(in_ptr + x2, None).to(tl.float32)
    s = tl.load(scale_ptr + x0, None, eviction_policy='evict_last').to(tl.float32)
    eps = tl.full([1], 1e-5, tl.float32)
    t = s + eps
    r = tl.sqrt_rn(t)          # pos22 path
    one = tl.full([1], 1.0, tl.float32)
    inv = one / r
    y = a * inv
    tl.store(out_ptr + x2, y.to(tl.bfloat16), None)

@triton.jit
def k_rsqrt(out_ptr, in_ptr, scale_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x2 = xindex
    x0 = xindex % 64
    a = tl.load(in_ptr + x2, None).to(tl.float32)
    s = tl.load(scale_ptr + x0, None, eviction_policy='evict_last').to(tl.float32)
    eps = tl.full([1], 1e-5, tl.float32)
    t = s + eps
    inv = libdevice.rsqrt(t)   # pos23 path
    y = a * inv
    tl.store(out_ptr + x2, y.to(tl.bfloat16), None)

# libdevice import inside triton namespace
from triton.language.extra import libdevice  # noqa

a = torch.randn(N, dtype=torch.bfloat16, device='cuda')
scale = torch.rand(C, dtype=torch.bfloat16, device='cuda').abs() + 0.1
out = torch.empty_like(a)
XBLOCK = 1024
grid = (triton.cdiv(N, XBLOCK),)

def run_sqrt(): k_sqrt_div[grid](out, a, scale, N, XBLOCK=XBLOCK)
def run_rsqrt(): k_rsqrt[grid](out, a, scale, N, XBLOCK=XBLOCK)

run_sqrt(); run_rsqrt(); torch.cuda.synchronize()
t_sqrt = do_bench(run_sqrt, return_mode="min") * 1000
t_rsqrt = do_bench(run_rsqrt, return_mode="min") * 1000
print(f"sqrt_rn+div : {t_sqrt:8.1f} us")
print(f"libdev.rsqrt: {t_rsqrt:8.1f} us")
print(f"speedup     : {t_sqrt/t_rsqrt:.2f}x")
# pure memcopy SOL for reference
src = torch.empty(N, dtype=torch.bfloat16, device='cuda'); dst=torch.empty_like(src)
t_copy = do_bench(lambda: dst.copy_(src), return_mode="min")*1000
print(f"copy SOL    : {t_copy:8.1f} us  (read+write {2*N*2/1e9:.2f} GB)")
