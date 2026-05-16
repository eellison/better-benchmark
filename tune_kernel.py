"""
Manual tuning of the 1.99x-from-SOL reduction kernel.

Kernel: reshape + mul + mul + sum(dim=[0,1])
Input shapes: mm_16[2048,4096], add_9[4,512,4096], rsqrt_4[4,512,1]
Output: [1,1,4096]

This is an outer reduction: reducing 2048 rows of width 4096.
The contiguous dimension (4096) is the non-reduced dim.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def fused_mul_sum_kernel(
    in_ptr0,  # [2048, 4096] - mm_16 (contiguous)
    in_ptr1,  # [4, 512, 4096] = [2048, 4096] - add_9 (contiguous)
    in_ptr2,  # [4, 512, 1] = [2048] - rsqrt_4
    out_ptr,  # [4096]
    xnumel: tl.constexpr,  # 4096
    rnumel: tl.constexpr,  # 2048
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]
    acc = tl.zeros([XBLOCK, RBLOCK], dtype=tl.float32)

    for roffset in tl.range(0, rnumel, RBLOCK):
        ridx = roffset + rbase
        rmask = ridx < rnumel

        # Load mm_16[r, x] and add_9[r, x]
        idx = xindex + xnumel * ridx
        a = tl.load(in_ptr0 + idx, rmask, other=0.0)
        b = tl.load(in_ptr1 + idx, rmask, other=0.0)

        # Load rsqrt_4[r] (broadcast over x)
        c = tl.load(in_ptr2 + ridx, rmask, other=0.0)

        # Compute: a * (b * c)
        val = a * (b * c)
        acc += tl.where(rmask, val, tl.zeros_like(val))

    result = tl.sum(acc, axis=1)[:, None]
    tl.store(out_ptr + xindex, result, mask=xindex < xnumel)


def benchmark_config(xblock, rblock, n_warmup=50, n_iter=500):
    xnumel = 4096
    rnumel = 2048

    in0 = torch.randn(rnumel, xnumel, device="cuda", dtype=torch.float32)
    in1 = torch.randn(rnumel, xnumel, device="cuda", dtype=torch.float32)
    in2 = torch.randn(rnumel, 1, device="cuda", dtype=torch.float32)
    out = torch.empty(xnumel, device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(xnumel, xblock),)

    for _ in range(n_warmup):
        fused_mul_sum_kernel[grid](in0, in1, in2, out, xnumel, rnumel, XBLOCK=xblock, RBLOCK=rblock)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        fused_mul_sum_kernel[grid](in0, in1, in2, out, xnumel, rnumel, XBLOCK=xblock, RBLOCK=rblock)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iter * 1e6

    # Verify correctness
    expected = (in0 * in1 * in2).sum(dim=0)
    assert torch.allclose(out, expected, atol=1e-2, rtol=1e-2), \
        f"Max diff: {(out - expected).abs().max().item()}"

    n_programs = triton.cdiv(xnumel, xblock)
    n_iters = triton.cdiv(rnumel, rblock)
    total_bytes = (rnumel * xnumel * 4 * 2 + rnumel * 4 + xnumel * 4)
    bw = total_bytes / (elapsed * 1e-6) / 1e9

    return elapsed, n_programs, n_iters, bw


def main():
    # Measure memcopy SOL for this data size (64 MB)
    total_bytes = 2048 * 4096 * 4 * 2 + 2048 * 4 + 4096 * 4
    n_elems = total_bytes // 4
    src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(50):
        dst.copy_(src)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(500):
        dst.copy_(src)
    torch.cuda.synchronize()
    sol_us = (time.perf_counter() - start) / 500 * 1e6
    sol_bw = total_bytes / (sol_us * 1e-6) / 1e9
    print(f"Memcopy SOL: {sol_us:.1f} us ({sol_bw:.1f} GB/s)")
    print()

    # Also measure inductor's default
    import torch._inductor.config as config
    config.split_reductions = False
    config.force_disable_caches = True

    class Repro(torch.nn.Module):
        def forward(self, mm_16, add_9, rsqrt_4):
            reshape_default = torch.ops.aten.reshape.default(mm_16, [4, 512, 4096])
            mul_tensor = torch.ops.aten.mul.Tensor(add_9, rsqrt_4)
            mul_tensor_1 = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor)
            return torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True)

    mod = Repro()
    inputs = [
        torch.randn([2048, 4096], dtype=torch.float32, device="cuda"),
        torch.randn([4, 512, 4096], dtype=torch.float32, device="cuda"),
        torch.randn([4, 512, 1], dtype=torch.float32, device="cuda"),
    ]
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(50):
            compiled(*inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(500):
            compiled(*inputs)
        torch.cuda.synchronize()
        inductor_us = (time.perf_counter() - start) / 500 * 1e6
    print(f"Inductor default: {inductor_us:.1f} us")
    print()

    # Test various configurations
    configs = [
        # (XBLOCK, RBLOCK)
        (1, 2048),    # 4096 programs, persistent (might OOM registers)
        (1, 1024),    # 4096 programs, 2 iters
        (2, 1024),    # 2048 programs, 2 iters
        (4, 512),     # 1024 programs, 4 iters
        (4, 1024),    # 1024 programs, 2 iters
        (8, 256),     # 512 programs, 8 iters
        (8, 512),     # 512 programs, 4 iters
        (8, 1024),    # 512 programs, 2 iters
        (16, 128),    # 256 programs, 16 iters
        (16, 256),    # 256 programs, 8 iters
        (16, 512),    # 256 programs, 4 iters
        (32, 64),     # 128 programs, 32 iters
        (32, 128),    # 128 programs, 16 iters
        (32, 256),    # 128 programs, 8 iters
        (64, 32),     # 64 programs, 64 iters
        (64, 64),     # 64 programs, 32 iters
        (64, 128),    # 64 programs, 16 iters
        (128, 32),    # 32 programs, 64 iters
        (128, 64),    # 32 programs, 32 iters
    ]

    print(f"{'XBLOCK':>8} {'RBLOCK':>8} {'Programs':>10} {'R_iters':>8} {'Time(us)':>10} {'BW(GB/s)':>10} {'vs SOL':>8}")
    print("-" * 75)

    best_time = float("inf")
    best_config = None

    for xb, rb in configs:
        try:
            t, n_prog, n_iters, bw = benchmark_config(xb, rb)
            gap = t / sol_us
            marker = " *" if t < best_time else ""
            print(f"{xb:>8} {rb:>8} {n_prog:>10} {n_iters:>8} {t:>9.1f} {bw:>9.1f} {gap:>7.2f}x{marker}")
            if t < best_time:
                best_time = t
                best_config = (xb, rb)
        except Exception as e:
            print(f"{xb:>8} {rb:>8} {'FAILED':>10} {str(e)[:40]}")

    print()
    print(f"Best: XBLOCK={best_config[0]}, RBLOCK={best_config[1]}: {best_time:.1f} us")
    print(f"Inductor: {inductor_us:.1f} us")
    print(f"SOL: {sol_us:.1f} us")
    print(f"Best vs SOL: {best_time/sol_us:.2f}x")
    print(f"Inductor vs SOL: {inductor_us/sol_us:.2f}x")
    print(f"Best vs Inductor: {best_time/inductor_us:.2f}x ({(1 - best_time/inductor_us)*100:.1f}% faster)")


if __name__ == "__main__":
    main()
