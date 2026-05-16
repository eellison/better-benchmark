import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def fused_mul_sum_kernel(
    in_ptr0, in_ptr1, in_ptr2, out_ptr,
    xnumel: tl.constexpr, rnumel: tl.constexpr,
    XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]
    acc = tl.zeros([XBLOCK, RBLOCK], dtype=tl.float32)
    for roffset in tl.range(0, rnumel, RBLOCK):
        ridx = roffset + rbase
        rmask = ridx < rnumel
        idx = xindex + xnumel * ridx
        a = tl.load(in_ptr0 + idx, rmask, other=0.0)
        b = tl.load(in_ptr1 + idx, rmask, other=0.0)
        c = tl.load(in_ptr2 + ridx, rmask, other=0.0)
        val = a * (b * c)
        acc += tl.where(rmask, val, tl.zeros_like(val))
    result = tl.sum(acc, axis=1)[:, None]
    tl.store(out_ptr + xindex, result, mask=xindex < xnumel)


xnumel, rnumel = 4096, 2048
in0 = torch.randn(rnumel, xnumel, device="cuda", dtype=torch.float32)
in1 = torch.randn(rnumel, xnumel, device="cuda", dtype=torch.float32)
in2 = torch.randn(rnumel, 1, device="cuda", dtype=torch.float32)
out = torch.empty(xnumel, device="cuda", dtype=torch.float32)
grid = (triton.cdiv(xnumel, 16),)

for nw in [4, 16]:
    def fn(nw=nw):
        fused_mul_sum_kernel[grid](
            in0, in1, in2, out, xnumel, rnumel,
            XBLOCK=16, RBLOCK=256, num_warps=nw,
        )
    ms = do_bench(fn, warmup=100, rep=500)
    print(f"do_bench num_warps={nw:>2}: {ms * 1000:.1f} us")
