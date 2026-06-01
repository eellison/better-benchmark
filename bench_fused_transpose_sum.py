"""
Benchmark: fused reduction+transpose kernel vs current 3-kernel approach.

Pattern:
  input: f32[25216, 3072]
  output1: f32[3072, 25216] = permute(input, [1, 0])
  output2: f32[3072] = sum(input, dim=0)

The fused kernel reads the input once, writes both outputs.
Transpose uses shared memory tile trick for coalesced global stores.
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

# Problem dimensions
M = 25216  # rows
N = 3072   # cols


# ─── Fused Triton kernel ───────────────────────────────────────────────────────
@triton.jit
def fused_transpose_sum_kernel(
    input_ptr,
    out_transpose_ptr,
    out_sum_ptr,
    M,  # number of rows in input
    N,  # number of cols in input
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_om: tl.constexpr,  # stride for transpose output (row stride = cols of transposed = M)
    stride_on: tl.constexpr,  # stride for transpose output (col stride = 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each program instance handles a [BLOCK_M, BLOCK_N] tile of the input.
    - Writes transposed tile to out_transpose using shared memory for coalescing.
    - Accumulates partial sums along dim 0, then atomically adds to out_sum.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Mask for boundary
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load tile [BLOCK_M, BLOCK_N] from input
    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    tile = tl.load(input_ptrs, mask=mask, other=0.0)

    # ─── Transpose write via shared memory for coalesced stores ───
    # Output transposed: out[n, m] = input[m, n]
    # out_transpose is [N, M] with strides [stride_om, stride_on]
    # We write tile transposed: for each (bm, bn), write to out[offs_n[bn], offs_m[bm]]
    # Coalesced write means consecutive threads write consecutive addresses.
    # With shared memory trick:
    #   1. Store tile to shared mem in [BLOCK_M, BLOCK_N] layout
    #   2. Read from shared mem in [BLOCK_N, BLOCK_M] layout (transposed)
    #   3. Write to global in the transposed order
    # In Triton, we can just do the index math directly since Triton handles
    # the vectorization. The key insight: we want consecutive threads to write
    # consecutive memory locations in the output.

    # For the transpose output [N, M], row = offs_n, col = offs_m
    # We want stores where consecutive elements in the last dim (cols of output = M dim)
    # are written by consecutive threads. So we iterate over BLOCK_N (rows of output tile)
    # and write BLOCK_M elements per row.
    out_t_ptrs = out_transpose_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    mask_t = mask_n[:, None] & mask_m[None, :]
    # tile is [BLOCK_M, BLOCK_N], we need [BLOCK_N, BLOCK_M]
    tile_t = tl.trans(tile)
    tl.store(out_t_ptrs, tile_t, mask=mask_t)

    # ─── Partial sum along dim 0 (reduce BLOCK_M rows → [BLOCK_N]) ───
    partial_sum = tl.sum(tile, axis=0)  # shape [BLOCK_N]

    # Atomic add to out_sum[n] since multiple pid_m blocks contribute
    sum_ptrs = out_sum_ptr + offs_n
    tl.atomic_add(sum_ptrs, partial_sum, mask=mask_n)


def fused_transpose_sum(input_tensor):
    """Run the fused kernel."""
    M_val, N_val = input_tensor.shape
    out_transpose = torch.empty(N_val, M_val, device=input_tensor.device, dtype=input_tensor.dtype)
    out_sum = torch.zeros(N_val, device=input_tensor.device, dtype=input_tensor.dtype)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(M_val, BLOCK_M), triton.cdiv(N_val, BLOCK_N))

    fused_transpose_sum_kernel[grid](
        input_tensor,
        out_transpose,
        out_sum,
        M_val,
        N_val,
        input_tensor.stride(0),
        input_tensor.stride(1),
        out_transpose.stride(0),
        out_transpose.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out_transpose, out_sum


# ─── Reference (unfused) ──────────────────────────────────────────────────────
def reference_unfused(input_tensor):
    permuted = input_tensor.permute(1, 0).contiguous()
    summed = input_tensor.sum(dim=0)
    return permuted, summed


# ─── torch.compile version (matches the repro pattern) ────────────────────────
@torch.compile
def compiled_pattern(input_tensor):
    permuted = input_tensor.permute(1, 0).contiguous()
    summed = input_tensor.sum(dim=0, keepdim=True).reshape(-1)
    return permuted, summed


# ─── SOL: just memcopy the same bytes ─────────────────────────────────────────
def sol_memcopy(input_tensor, out1, out2):
    """Speed-of-light: just copy the same number of bytes."""
    # Read: M*N*4 bytes (input)
    # Write: N*M*4 bytes (transpose) + N*4 bytes (sum)
    # Total memory moved: 2*M*N*4 + N*4
    # Approximate with a copy of equal size
    out1.copy_(input_tensor.t())  # This is not quite SOL but close
    return out1


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")

    input_tensor = torch.randn(M, N, device=device, dtype=torch.float32)

    # Warm up and verify correctness
    ref_t, ref_s = reference_unfused(input_tensor)
    fused_t, fused_s = fused_transpose_sum(input_tensor)

    assert torch.allclose(ref_t, fused_t, atol=1e-3, rtol=1e-3), \
        f"Transpose mismatch! Max diff: {(ref_t - fused_t).abs().max().item()}"
    assert torch.allclose(ref_s, fused_s, atol=1e-2, rtol=1e-3), \
        f"Sum mismatch! Max diff: {(ref_s - fused_s).abs().max().item()}"
    print("Correctness verified!")

    # ─── Benchmark ─────────────────────────────────────────────────────────────
    # Warm up compiled version
    _ = compiled_pattern(input_tensor)
    _ = compiled_pattern(input_tensor)

    # Current approach: torch.compile
    ms_compiled = do_bench(lambda: compiled_pattern(input_tensor))

    # Fused Triton kernel
    ms_fused = do_bench(lambda: fused_transpose_sum(input_tensor))

    # SOL: simple memcopy of same total bytes
    # Total bytes: read input (M*N*4) + write transpose (M*N*4) + write sum (N*4)
    # ≈ 2 * M * N * 4 bytes
    total_bytes = 2 * M * N * 4 + N * 4
    # For SOL, just time a raw copy of equivalent size
    sol_src = torch.randn(M, N, device=device, dtype=torch.float32)
    sol_dst = torch.empty(M, N, device=device, dtype=torch.float32)
    ms_sol = do_bench(lambda: sol_dst.copy_(sol_src))

    print(f"\n{'='*60}")
    print(f"Benchmark: fused transpose+sum for f32[{M}, {N}]")
    print(f"{'='*60}")
    print(f"Total memory traffic: {total_bytes / 1e9:.3f} GB")
    print(f"  (read {M}x{N}x4 + write {N}x{M}x4 + write {N}x4)")
    print(f"{'='*60}")
    print(f"  torch.compile (current):  {ms_compiled:.3f} ms")
    print(f"  Fused Triton kernel:      {ms_fused:.3f} ms")
    print(f"  SOL (memcopy equiv):      {ms_sol:.3f} ms")
    print(f"{'='*60}")
    print(f"  Speedup (compiled/fused): {ms_compiled / ms_fused:.2f}x")
    print(f"  Fused vs SOL:             {ms_fused / ms_sol:.2f}x of SOL")
    print(f"  Bandwidth (fused):        {total_bytes / (ms_fused * 1e-3) / 1e9:.1f} GB/s")
    print(f"  Bandwidth (SOL copy):     {(2 * M * N * 4) / (ms_sol * 1e-3) / 1e9:.1f} GB/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
