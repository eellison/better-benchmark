"""Test: optimal transpose+sum without atomic_add."""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def transpose_sum_workspace(
    in_ptr, out_transpose_ptr, workspace_ptr, out_sum_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    2D tiled: each CTA handles [BLOCK_M, BLOCK_N] tile.
    - Coalesced load
    - Coalesced transposed store (via tl.trans)
    - Partial sum written to workspace (no atomics)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_m_blocks = tl.cdiv(M, BLOCK_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

    # Load tile [BLOCK_M, BLOCK_N] - coalesced
    tile = tl.load(in_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)

    # Transpose store - coalesced via tl.trans
    out_mask = (offs_n < N)[:, None] & (offs_m < M)[None, :]
    tl.store(out_transpose_ptr + offs_n[:, None] * M + offs_m[None, :], tl.trans(tile), mask=out_mask)

    # Partial sum along M dim -> [BLOCK_N]
    partial = tl.sum(tile, axis=0)
    # Write to workspace[pid_m, offs_n]
    ws_mask = offs_n < N
    tl.store(workspace_ptr + pid_m * N + offs_n, partial, mask=ws_mask)


@triton.jit
def reduce_workspace(workspace_ptr, out_sum_ptr, N, num_m_blocks, BLOCK_N: tl.constexpr):
    """Second kernel: reduce workspace along m-blocks dimension."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(num_m_blocks):
        partial = tl.load(workspace_ptr + i * N + offs_n, mask=mask, other=0.0)
        acc += partial

    tl.store(out_sum_ptr + offs_n, acc, mask=mask)


def transpose_sum_no_atomic(x):
    M, N = x.shape
    BLOCK_M, BLOCK_N = 64, 64
    num_m_blocks = triton.cdiv(M, BLOCK_M)
    num_n_blocks = triton.cdiv(N, BLOCK_N)

    out_t = torch.empty(N, M, device=x.device, dtype=x.dtype)
    workspace = torch.empty(num_m_blocks, N, device=x.device, dtype=x.dtype)
    out_s = torch.empty(N, device=x.device, dtype=x.dtype)

    # Kernel 1: transpose + partial sums
    grid1 = (num_m_blocks, num_n_blocks)
    transpose_sum_workspace[grid1](x, out_t, workspace, out_s, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # Kernel 2: reduce workspace
    grid2 = (num_n_blocks,)
    reduce_workspace[grid2](workspace, out_s, N, num_m_blocks, BLOCK_N=BLOCK_N)

    return out_t, out_s


def main():
    M, N = 25216, 3072
    x = torch.randn(M, N, device='cuda')

    # Verify correctness
    out_t, out_s = transpose_sum_no_atomic(x)
    ref_t = x.T.contiguous()
    ref_s = x.sum(dim=0)
    print(f"Transpose correct: {torch.allclose(out_t, ref_t)}")
    print(f"Sum correct: {torch.allclose(out_s, ref_s, atol=1e-3)}")

    # Benchmark (2 kernels, no atomics)
    transpose_sum_no_atomic(x)
    ms_no_atomic = do_bench(lambda: transpose_sum_no_atomic(x))

    # Compare: SOL
    src = torch.randn(M, N, device='cuda')
    dst = torch.empty(M, N, device='cuda')
    ms_sol = do_bench(lambda: torch.add(src, 1, out=dst))

    print(f"\nNo-atomic (2 kernels): {ms_no_atomic:.3f} ms")
    print(f"SOL: {ms_sol:.3f} ms")
    print(f"vs SOL: {ms_no_atomic/ms_sol:.2f}x")


if __name__ == "__main__":
    main()
