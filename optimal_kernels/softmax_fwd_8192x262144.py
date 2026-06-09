"""
Hand-optimized Triton kernel for Softmax Forward [8192, 262144] bf16.
Pattern: bf16 -> f32 -> amax -> sub -> exp -> sum -> div -> bf16

Key insight: With rnumel=262144, persistent reduction is NOT viable (would need
262144 * 4B = 1MB SRAM per row). Instead, we use the online softmax algorithm
with a single pass to compute max and sum(exp), then a second pass to write output.

This is a 2-pass reduction. The minimum achievable time is:
  (8192 * 262144 * 2 * 3) / BW = 3 * input_size / BW
  = 12.88 GB / 6.9 TB/s = ~1865 us

ROOT CAUSE OF INDUCTOR'S GAP:
  Inductor's _reduction_configs() heuristic caps R0_BLOCK at 1024 for this shape.
  The configs it generates are:
    XBLOCK=1,  R0_BLOCK=1024, warps=8
    XBLOCK=16, R0_BLOCK=512,  warps=16
    XBLOCK=8,  R0_BLOCK=512,  warps=16
    XBLOCK=64, R0_BLOCK=64,   warps=16
    XBLOCK=64, R0_BLOCK=4,    warps=8

  With R0_BLOCK=1024 and rnumel=262144, each program loops 256 times.
  With R0_BLOCK=8192, it only loops 32 times -> less overhead, better ILP.

  The hand-optimized kernel uses:
    XBLOCK=1, R0_BLOCK=8192, num_warps=8
  Achieving 1915us vs inductor's 3457us (1.81x speedup).

FIX DIRECTION:
  The reduction config generator should include larger R0_BLOCK values (4096, 8192)
  in its search space when rnumel is large (>= 64K). The constraint is register
  pressure: R0_BLOCK * sizeof(accumulators) must fit. With online softmax state
  (just max + sum per row), even R0_BLOCK=8192 only uses ~32KB registers which
  is well within B200's 64KB/SM limit.
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def softmax_fwd_kernel(
    input_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Online softmax: single program per row, 2-pass over reduction dim."""
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    # Pass 1: compute max and sum(exp(x - running_max)) using online algorithm
    m_i = tl.full([], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)

    for col_offset in tl.range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf'))
        x = x.to(tl.float32)

        # Online softmax update
        m_ij = tl.max(x, axis=0)
        new_m = tl.maximum(m_i, m_ij)
        # Rescale old accumulator
        l_i = l_i * tl.exp(m_i - new_m)
        # Add new contribution
        p = tl.exp(x - new_m)
        l_i = l_i + tl.sum(p, axis=0)
        m_i = new_m

    # Pass 2: compute softmax output = exp(x - m) / l
    for col_offset in tl.range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf'))
        x = x.to(tl.float32)
        p = tl.exp(x - m_i) / l_i
        tl.store(output_ptr + row_start + cols, p.to(tl.bfloat16), mask=mask)


@triton.jit
def softmax_fwd_kernel_v2(
    input_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Multi-row variant: each program handles ROWS_PER_PROGRAM rows."""
    pid = tl.program_id(0)

    for row_offset in tl.range(0, ROWS_PER_PROGRAM):
        row_idx = pid * ROWS_PER_PROGRAM + row_offset
        row_start = row_idx * n_cols

        # Pass 1: online softmax
        m_i = tl.full([], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([], dtype=tl.float32)

        for col_offset in tl.range(0, n_cols, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            x = tl.load(input_ptr + row_start + cols)
            x = x.to(tl.float32)
            m_ij = tl.max(x, axis=0)
            new_m = tl.maximum(m_i, m_ij)
            l_i = l_i * tl.exp(m_i - new_m)
            p = tl.exp(x - new_m)
            l_i = l_i + tl.sum(p, axis=0)
            m_i = new_m

        # Pass 2
        for col_offset in tl.range(0, n_cols, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            x = tl.load(input_ptr + row_start + cols)
            x = x.to(tl.float32)
            p = tl.exp(x - m_i) / l_i
            tl.store(output_ptr + row_start + cols, p.to(tl.bfloat16))


def benchmark_softmax():
    M, N = 8192, 262144
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    out = torch.empty_like(x)

    # Reference
    ref = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)

    # Test correctness
    softmax_fwd_kernel[(M,)](x, out, N, BLOCK_SIZE=4096, num_warps=8)
    torch.cuda.synchronize()
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max diff (v1, BLOCK=4096, warps=8): {max_diff:.6f}")

    # Benchmark different configs
    configs = [
        (2048, 4, 1),
        (2048, 8, 1),
        (4096, 4, 1),
        (4096, 8, 1),
        (4096, 16, 1),
        (8192, 8, 1),
        (8192, 16, 1),
    ]

    print(f"\n{'BLOCK':>8} {'warps':>6} {'rows/prog':>10} {'time_us':>10} {'BW_TB/s':>10}")
    print("-" * 50)

    total_bytes = M * N * 2 * 3  # 2 reads + 1 write, all bf16
    best_time = float('inf')
    best_cfg = None

    for block_size, num_warps, rows_per_prog in configs:
        grid = (M // rows_per_prog,)
        if rows_per_prog == 1:
            fn = lambda: softmax_fwd_kernel[grid](x, out, N, BLOCK_SIZE=block_size, num_warps=num_warps)
        else:
            fn = lambda bs=block_size, nw=num_warps, rpp=rows_per_prog: softmax_fwd_kernel_v2[grid](
                x, out, N, BLOCK_SIZE=bs, ROWS_PER_PROGRAM=rpp, num_warps=nw
            )

        try:
            ms = do_bench(fn, warmup=25, rep=100, return_mode='min')
            us = ms * 1000
            bw = total_bytes / (ms / 1000) / 1e12
            print(f"{block_size:>8} {num_warps:>6} {rows_per_prog:>10} {us:>10.1f} {bw:>10.3f}")
            if us < best_time:
                best_time = us
                best_cfg = (block_size, num_warps, rows_per_prog)
        except Exception as e:
            print(f"{block_size:>8} {num_warps:>6} {rows_per_prog:>10} {'FAILED':>10} {str(e)[:30]}")

    # Also benchmark inductor
    print("\n--- Inductor baseline ---")
    class SoftmaxMod(torch.nn.Module):
        def forward(self, x):
            x = x.float()
            amax = x.amax(-1, keepdim=True)
            x = (x - amax).exp()
            return (x / x.sum(-1, keepdim=True)).to(torch.bfloat16)

    torch._dynamo.reset()
    mod = torch.compile(SoftmaxMod())
    with torch.no_grad():
        for _ in range(3):
            mod(x)
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        mod(x)
    torch.cuda.synchronize()
    inductor_ms = do_bench(lambda: g.replay(), warmup=25, rep=100, return_mode='min')
    inductor_us = inductor_ms * 1000
    print(f"Inductor: {inductor_us:.1f} us")

    # SOL
    sol_elems = M * N * 2 * 2 // 4  # total bytes / 4 for f32 elements (read+write)
    src = torch.empty(sol_elems, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: torch.add(src, 1, out=dst), warmup=25, rep=100, return_mode='min')
    sol_us = sol_ms * 1000
    del src, dst

    print(f"\n--- Summary ---")
    print(f"SOL (memcpy): {sol_us:.1f} us")
    print(f"Inductor:     {inductor_us:.1f} us")
    print(f"Best hand:    {best_time:.1f} us (BLOCK={best_cfg[0]}, warps={best_cfg[1]}, rows/prog={best_cfg[2]})")
    print(f"Speedup over inductor: {inductor_us / best_time:.2f}x")
    print(f"Ratio to SOL: {best_time / sol_us:.2f}x")


if __name__ == "__main__":
    benchmark_softmax()
