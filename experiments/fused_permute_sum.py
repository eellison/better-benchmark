"""
Fused permute+sum kernel for BERT-Large backward pass.

Pattern found 145x in sum_sum_sum_6b730b98ce8a (BERT-Large full backward):
    permute_out = X.permute(1, 0)   # [M, N] -> [N, M], i.e. transpose
    col_sum     = X.sum(dim=0)      # [M, N] -> [N]

Both operations read the ENTIRE tensor X. Unfused, that's 2x memory traffic.
Fusing them reads X once and produces both outputs, halving traffic.

Concrete instances from the repro:
  1. view_538: f32[2048, 1024] (line ~28-30) - dense layer backward (modeling_bert.py:481)
     permute_default  = permute(view_538, [1, 0])  -> f32[1024, 2048]
     sum_dim_int_list_3 = sum(view_538, [0], True) -> f32[1, 1024]

  2. view_544: f32[2048, 4096] (line ~43-45) - FFN dense backward (modeling_bert.py:339)
     permute_default_2 = permute(view_544, [1, 0])  -> f32[4096, 2048]
     sum_dim_int_list_7 = sum(view_544, [0], True)  -> f32[1, 4096]

  3. view_547: f32[2048, 1024] (line ~53-55) - attention output backward (modeling_bert.py:294)
     permute_default_3 = permute(view_547, [1, 0])  -> f32[1024, 2048]
     sum_dim_int_list_10 = sum(view_547, [0], True) -> f32[1, 1024]

Key shapes: f32[2048, 1024] (8 MB) and f32[2048, 4096] (32 MB).
145 instances * average ~12 MB read savings = ~1.16 GB total saved.
"""

import torch
import triton
import triton.language as tl
import time


# ============================================================================
# Fused Triton kernel: reads X[M, N] once, writes:
#   - transposed[N, M] = X^T
#   - col_sum[N] = sum(X, dim=0)
#
# Strategy: 2D tiling with square-ish tiles. Each program handles a
# BLOCK_M x BLOCK_N tile of X. The transpose write is coalesced because
# we write BLOCK_N contiguous elements along the M-axis of T (which is the
# fast axis of T[N, M] row-major). The column sum is computed via atomics
# (each tile contributes a partial sum for its columns).
#
# For the sum-only output, we use atomic adds so multiple row-tiles can
# contribute to the same column's sum without requiring a second pass.
# ============================================================================

@triton.jit
def fused_permute_sum_kernel(
    X_ptr,           # input [M, N], row-major
    T_ptr,           # output transposed [N, M], row-major
    S_ptr,           # output col_sum [N]
    M,               # number of rows
    N,               # number of columns
    stride_xm,      # stride of X along dim 0 (= N for contiguous)
    stride_xn,      # stride of X along dim 1 (= 1 for contiguous)
    stride_tn,      # stride of T along dim 0 (= M for contiguous)
    stride_tm,      # stride of T along dim 1 (= 1 for contiguous)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D grid: (num_col_tiles, num_row_tiles)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    col_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]

    col_mask = col_offset < N
    row_mask = row_offset < M

    # Load tile X[row_offset, col_offset] - coalesced read (row-major, contiguous along N)
    x_ptrs = X_ptr + row_offset[:, None] * stride_xm + col_offset[None, :] * stride_xn
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Write transposed tile: T[col_offset, row_offset]
    # T is [N, M] row-major. T[n, m] at T_ptr + n*stride_tn + m*stride_tm
    # Writing along row_offset (the M axis = fast axis of T) is coalesced!
    t_ptrs = T_ptr + col_offset[:, None] * stride_tn + row_offset[None, :] * stride_tm
    t_mask = col_mask[:, None] & row_mask[None, :]
    tl.store(t_ptrs, tl.trans(x), mask=t_mask)

    # Partial column sum for this tile: sum over BLOCK_M rows
    partial_sum = tl.sum(x, axis=0)  # [BLOCK_N]

    # Atomic add to col_sum
    s_ptrs = S_ptr + col_offset
    tl.atomic_add(s_ptrs, partial_sum, mask=col_mask)


def fused_permute_sum(x: torch.Tensor):
    """
    Given x of shape [M, N], returns:
      - transposed: x.permute(1, 0), shape [N, M]
      - col_sum: x.sum(dim=0), shape [N]
    Reading x only once.
    """
    assert x.ndim == 2 and x.is_contiguous(), "Input must be 2D contiguous"
    M, N = x.shape

    transposed = torch.empty(N, M, device=x.device, dtype=x.dtype)
    col_sum = torch.zeros(N, device=x.device, dtype=x.dtype)  # zeros for atomic add

    # Tuning: 64x32 gives good performance across both [2048,1024] and [2048,4096]
    BLOCK_M = 64
    BLOCK_N = 32

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))

    fused_permute_sum_kernel[grid](
        x, transposed, col_sum,
        M, N,
        x.stride(0), x.stride(1),
        transposed.stride(0), transposed.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return transposed, col_sum


# ============================================================================
# Unfused baseline (what inductor does today)
# ============================================================================

def unfused_permute_sum(x: torch.Tensor):
    """Separate permute + separate sum (2 full reads of x)."""
    transposed = x.permute(1, 0).contiguous()  # must materialize
    col_sum = x.sum(dim=0)
    return transposed, col_sum


# ============================================================================
# Correctness check
# ============================================================================

def check_correctness(M, N, dtype=torch.float32):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    t_ref, s_ref = unfused_permute_sum(x)
    t_fused, s_fused = fused_permute_sum(x)

    t_close = torch.allclose(t_ref, t_fused, atol=1e-4, rtol=1e-4)
    # Sum reduction order differs between fused (row-chunks) and unfused (full column).
    # With 2048 f32 additions, expect ~sqrt(2048)*eps ~ 0.002 max diff.
    s_close = torch.allclose(s_ref, s_fused, atol=1e-3, rtol=1e-4)

    if not t_close or not s_close:
        print(f"  FAILED for shape [{M}, {N}]!")
        if not t_close:
            diff = (t_ref - t_fused).abs().max().item()
            print(f"    Transpose max diff: {diff}")
        if not s_close:
            diff = (s_ref - s_fused).abs().max().item()
            print(f"    Sum max diff: {diff}")
        return False
    return True


# ============================================================================
# Benchmark
# ============================================================================

def fused_permute_sum_tuned(x: torch.Tensor, BLOCK_M=64, BLOCK_N=64):
    """
    Tuned variant with configurable tile size.
    """
    assert x.ndim == 2 and x.is_contiguous(), "Input must be 2D contiguous"
    M, N = x.shape

    transposed = torch.empty(N, M, device=x.device, dtype=x.dtype)
    col_sum = torch.zeros(N, device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))

    fused_permute_sum_kernel[grid](
        x, transposed, col_sum,
        M, N,
        x.stride(0), x.stride(1),
        transposed.stride(0), transposed.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return transposed, col_sum


def benchmark_fn(fn, x, warmup=50, rep=200):
    """Benchmark using triton's do_bench for accurate GPU timing."""
    ms = triton.testing.do_bench(lambda: fn(x), warmup=warmup, rep=rep)
    return ms


def compute_bandwidth(M, N, ms, dtype=torch.float32):
    """Compute effective bandwidth in GB/s."""
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    # Read X once: M*N elements
    # Write transposed: M*N elements
    # Write col_sum: N elements
    total_bytes = (M * N + M * N + N) * elem_bytes
    return total_bytes / (ms * 1e-3) / 1e9


def main():
    print("=" * 70)
    print("Fused permute+sum kernel for BERT-Large backward")
    print("Pattern: X[M,N] -> (X.permute(1,0), X.sum(dim=0))")
    print("Unfused reads X twice; fused reads X once.")
    print("=" * 70)
    print()

    shapes = [
        (2048, 1024),   # attention/dense layers (most common, ~120 instances)
        (2048, 4096),   # FFN intermediate layer (less common, ~25 instances)
    ]

    # Correctness
    print("Correctness check:")
    all_correct = True
    for M, N in shapes:
        ok = check_correctness(M, N)
        status = "PASS" if ok else "FAIL"
        print(f"  [{M}, {N}]: {status}")
        all_correct &= ok
    print()

    if not all_correct:
        print("Correctness check failed! Aborting benchmark.")
        return

    # Benchmark
    print("Benchmark results:")
    print(f"{'Shape':>20} | {'Unfused (ms)':>12} | {'Fused (ms)':>12} | {'Speedup':>8} | {'BW Unfused':>12} | {'BW Fused':>12} | {'Traffic saved':>14}")
    print("-" * 110)

    total_unfused_us = 0.0
    total_fused_us = 0.0

    for M, N in shapes:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        elem_bytes = 4  # f32

        ms_unfused = benchmark_fn(unfused_permute_sum, x)
        ms_fused = benchmark_fn(fused_permute_sum, x)

        speedup = ms_unfused / ms_fused

        # Bandwidth calculation
        # Unfused: read X twice (once for permute, once for sum) + write transposed + write sum
        # = 3*M*N + N elements read/written
        unfused_bytes = (3 * M * N + N) * elem_bytes
        bw_unfused = unfused_bytes / (ms_unfused * 1e-3) / 1e9

        # Fused: read X once + write transposed + write sum = 2*M*N + N elements
        fused_bytes = (2 * M * N + N) * elem_bytes
        bw_fused = fused_bytes / (ms_fused * 1e-3) / 1e9

        traffic_saved_mb = (M * N * elem_bytes) / 1e6  # one fewer read of X

        print(f"  [{M:>4}, {N:>4}] | {ms_unfused:>10.4f}ms | {ms_fused:>10.4f}ms | {speedup:>6.2f}x | {bw_unfused:>9.1f} GB/s | {bw_fused:>9.1f} GB/s | {traffic_saved_mb:>10.1f} MB")

        # Accumulate for total-graph estimate
        # In BERT-Large backward: ~120 instances of [2048,1024], ~25 of [2048,4096]
        if N == 1024:
            count = 120
        else:
            count = 25
        total_unfused_us += ms_unfused * 1000 * count  # convert to us
        total_fused_us += ms_fused * 1000 * count

    print()
    print("=" * 70)
    print("Estimated impact on full BERT-Large backward (145 instances):")
    print(f"  Total unfused time: {total_unfused_us/1000:.2f} ms")
    print(f"  Total fused time:   {total_fused_us/1000:.2f} ms")
    print(f"  Time saved:         {(total_unfused_us - total_fused_us)/1000:.2f} ms")
    print(f"  Traffic saved:      {(120 * 2048 * 1024 + 25 * 2048 * 4096) * 4 / 1e9:.2f} GB (one fewer read per instance)")
    print("=" * 70)

    # Tile size sweep
    print()
    print("Tile size sweep (BLOCK_M x BLOCK_N):")
    print(f"{'Shape':>20} | {'Tile':>10} | {'Fused (ms)':>12} | {'Speedup vs unfused':>18}")
    print("-" * 75)
    tile_sizes = [(32, 32), (32, 64), (64, 32), (64, 64), (128, 32), (32, 128)]
    for M, N in shapes:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        ms_base = benchmark_fn(unfused_permute_sum, x)
        for bm, bn in tile_sizes:
            fn = lambda x, bm=bm, bn=bn: fused_permute_sum_tuned(x, BLOCK_M=bm, BLOCK_N=bn)
            ms = benchmark_fn(fn, x)
            speedup = ms_base / ms
            print(f"  [{M:>4}, {N:>4}] | {bm:>3}x{bn:<3} | {ms:>10.4f}ms | {speedup:>14.2f}x")
        print()


if __name__ == "__main__":
    main()
