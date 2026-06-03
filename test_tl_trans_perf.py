"""Test: does tl.trans() improve performance for transposed stores?"""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def with_tl_trans(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Load tile, transpose via tl.trans(), store coalesced."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tile = tl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    # Transpose in shared mem, then store coalesced
    out_mask = (offs_n < N)[:, None] & (offs_m < M)[None, :]
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :], tl.trans(tile), mask=out_mask)


@triton.jit
def without_tl_trans(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Load tile, store with non-coalesced transposed indexing (inductor-style)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tile = tl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    # Store without transpose — non-coalesced (stride M between adjacent thread writes)
    tl.store(out_ptr + offs_n[None, :] * M + offs_m[:, None], tile, mask=mask)


def main():
    M, N = 25216, 3072
    inp = torch.randn(M, N, device='cuda')
    out1 = torch.empty(N, M, device='cuda')
    out2 = torch.empty(N, M, device='cuda')

    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Verify correctness
    with_tl_trans[grid](inp, out1, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    without_tl_trans[grid](inp, out2, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    ref = inp.T.contiguous()
    print(f"with_tl_trans correct: {torch.allclose(out1, ref)}")
    print(f"without_tl_trans correct: {torch.allclose(out2, ref)}")

    # Benchmark
    ms_with = do_bench(lambda: with_tl_trans[grid](inp, out1, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N))
    ms_without = do_bench(lambda: without_tl_trans[grid](inp, out2, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N))

    print(f"\nWith tl.trans():    {ms_with:.3f} ms")
    print(f"Without tl.trans(): {ms_without:.3f} ms")
    print(f"Speedup: {ms_without/ms_with:.2f}x")


if __name__ == "__main__":
    main()
