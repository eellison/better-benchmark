"""
Hand-optimized Triton kernel for Cross-Entropy Forward [8192, 262144] bf16.

Pattern: log_softmax + gather (nll_loss)
  - Compute log_softmax = x - max(x) - log(sum(exp(x - max(x))))
  - Gather the log_softmax value at the target index
  - Negate and mask with -100 padding

Key insight: This is similar to softmax but we only need to OUTPUT the gathered
scalar per row, not the full log_softmax output. So:
  - Read: bf16[8192, 262144] (2 passes for online softmax) + i64[8192] (labels)
  - Write: bf16[8192] (loss per sample)

This means we can AVOID the second full pass! Instead:
  Pass 1: online softmax to get max and log(sum(exp))
  Then: just compute log_softmax at the target index = x[target] - max - log_sum_exp
  No second pass needed!

Total bytes: 8192 * 262144 * 2 (one pass) + 8192 * 8 (labels) + 8192 * 2 (output) ~ 4.29 GB
Expected at ~6.9 TB/s: ~625 us

ROOT CAUSE: Inductor ALREADY uses the single-pass approach (online softmax + scalar
gather of x[label] afterward). The gap is purely from R0_BLOCK being capped too low.

Inductor's reduction heuristic generates configs with max R0_BLOCK=1024.
With rnumel=262144: 256 loop iterations per row.
Our kernel uses R0_BLOCK=8192: only 32 loop iterations per row.
Additionally, we use XBLOCK=1 with 4 warps giving 8192 programs for 148 SMs
(55 programs/SM = excellent occupancy).

Result: 689us vs inductor's 2113us (3.07x speedup!).
The hand kernel is within 10% of the theoretical single-pass SOL (624us).

FIX DIRECTION:
1. Increase max R0_BLOCK in _reduction_configs() for large rnumel (>=64K).
   Add configs like (XBLOCK=1, R0_BLOCK=4096, warps=4) and
   (XBLOCK=1, R0_BLOCK=8192, warps=4/8) to the search space.
2. The existing single-pass fusion is already correct algorithmically;
   only the tile size heuristic needs fixing.
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,     # bf16[M, N]
    labels_ptr,     # i64[M]
    output_ptr,     # bf16[M]
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass cross-entropy: compute max, sum(exp), and save x[label] in one pass."""
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load label for this row
    label = tl.load(labels_ptr + row_idx)
    ignore = (label == -100)

    # Single pass: online softmax + capture x[label]
    m_i = tl.full([], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    x_label = tl.full([], 0.0, dtype=tl.float32)

    # Clamp label to valid range for the gather
    safe_label = tl.where(ignore, tl.zeros([], dtype=tl.int64), label)

    for col_offset in tl.range(0, N, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + row_start + cols)
        x_f32 = x.to(tl.float32)

        # Online softmax update
        m_ij = tl.max(x_f32, axis=0)
        new_m = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - new_m)
        p = tl.exp(x_f32 - new_m)
        l_i = l_i + tl.sum(p, axis=0)
        m_i = new_m

        # Check if label falls in this block
        label_local = safe_label - col_offset
        in_block = (label_local >= 0) & (label_local < BLOCK_SIZE)
        if in_block:
            # Gather x[label] from this block
            x_at_label = tl.sum(tl.where(cols == safe_label, x_f32, tl.zeros([BLOCK_SIZE], dtype=tl.float32)), axis=0)
            x_label = x_at_label

    # log_softmax[label] = x[label] - max - log(sum_exp)
    log_softmax_label = x_label - m_i - tl.log(l_i)
    # loss = -log_softmax[label], masked
    loss = tl.where(ignore, tl.zeros([], dtype=tl.float32), -log_softmax_label)
    tl.store(output_ptr + row_idx, loss.to(tl.bfloat16))


@triton.jit
def cross_entropy_fwd_kernel_v2(
    logits_ptr,     # bf16[M, N]
    labels_ptr,     # i64[M]
    output_ptr,     # bf16[M]
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Alternative: track x[label] more efficiently with a simple scalar load."""
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load label for this row
    label = tl.load(labels_ptr + row_idx)
    ignore = (label == -100)

    # Load x[label] directly (single element load - very fast)
    safe_label = tl.where(ignore, tl.zeros([], dtype=tl.int64), label)
    x_label_bf16 = tl.load(logits_ptr + row_start + safe_label)
    x_label = x_label_bf16.to(tl.float32)

    # Single pass: just compute max and log(sum(exp))
    m_i = tl.full([], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)

    for col_offset in tl.range(0, N, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + row_start + cols)
        x_f32 = x.to(tl.float32)

        # Online softmax update
        m_ij = tl.max(x_f32, axis=0)
        new_m = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - new_m)
        p = tl.exp(x_f32 - new_m)
        l_i = l_i + tl.sum(p, axis=0)
        m_i = new_m

    # log_softmax[label] = x[label] - max - log(sum_exp)
    log_softmax_label = x_label - m_i - tl.log(l_i)
    # loss = -log_softmax[label], masked
    loss = tl.where(ignore, tl.zeros([], dtype=tl.float32), -log_softmax_label)
    tl.store(output_ptr + row_idx, loss.to(tl.bfloat16))


def benchmark_cross_entropy():
    M, N = 8192, 262144

    logits = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    labels = torch.randint(0, N, (M,), dtype=torch.int64, device='cuda')
    out = torch.empty(M, dtype=torch.bfloat16, device='cuda')

    # Verify correctness
    ref_log_softmax = torch.log_softmax(logits.float(), dim=-1)
    ref_loss = -ref_log_softmax[torch.arange(M, device='cuda'), labels].to(torch.bfloat16)

    cross_entropy_fwd_kernel_v2[(M,)](logits, labels, out, N, BLOCK_SIZE=4096, num_warps=8)
    torch.cuda.synchronize()
    max_diff = (out.float() - ref_loss.float()).abs().max().item()
    print(f"Max diff v2: {max_diff:.6f}")

    cross_entropy_fwd_kernel[(M,)](logits, labels, out, N, BLOCK_SIZE=4096, num_warps=8)
    torch.cuda.synchronize()
    max_diff = (out.float() - ref_loss.float()).abs().max().item()
    print(f"Max diff v1: {max_diff:.6f}")

    # Benchmark configs
    configs = [
        (2048, 4),
        (2048, 8),
        (4096, 4),
        (4096, 8),
        (8192, 4),
        (8192, 8),
        (8192, 16),
    ]

    print(f"\n{'version':>8} {'BLOCK':>8} {'warps':>6} {'time_us':>10}")
    print("-" * 40)

    best_time = float('inf')
    best_cfg = None

    for block_size, num_warps in configs:
        for version, kernel in [("v1", cross_entropy_fwd_kernel), ("v2", cross_entropy_fwd_kernel_v2)]:
            fn = lambda k=kernel, bs=block_size, nw=num_warps: k[(M,)](
                logits, labels, out, N, BLOCK_SIZE=bs, num_warps=nw
            )
            try:
                ms = do_bench(fn, warmup=25, rep=100, return_mode='min')
                us = ms * 1000
                print(f"{version:>8} {block_size:>8} {num_warps:>6} {us:>10.1f}")
                if us < best_time:
                    best_time = us
                    best_cfg = (version, block_size, num_warps)
            except Exception as e:
                print(f"{version:>8} {block_size:>8} {num_warps:>6} {'FAILED':>10} {str(e)[:30]}")

    # Inductor baseline
    print("\n--- Inductor baseline ---")
    import sys
    sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
    from repros.canonical.amax_sum_f0661488d68c.repro import Repro, make_inputs

    mod = Repro()
    inputs = make_inputs()
    torch._dynamo.reset()
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    inductor_ms = do_bench(lambda: compiled(*inputs), warmup=25, rep=100, return_mode='min')
    inductor_us = inductor_ms * 1000
    print(f"Inductor: {inductor_us:.1f} us")

    # SOL: single pass over bf16[8192, 262144]
    sol_bytes = M * N * 2 * 2  # read + write same size (for memcpy SOL calc)
    sol_elems = sol_bytes // 8
    src = torch.empty(sol_elems, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: torch.add(src, 1, out=dst), warmup=25, rep=100, return_mode='min')
    sol_us = sol_ms * 1000
    del src, dst

    # True SOL for cross-entropy is single-pass read only
    true_sol_bytes = M * N * 2  # just read bf16 input
    true_sol_elems = true_sol_bytes // 8
    src2 = torch.empty(true_sol_elems, dtype=torch.float32, device='cuda')
    dst2 = torch.empty_like(src2)
    true_sol_ms = do_bench(lambda: torch.add(src2, 1, out=dst2), warmup=25, rep=100, return_mode='min')
    true_sol_us = true_sol_ms * 1000
    del src2, dst2

    print(f"\n--- Summary ---")
    print(f"SOL (memcpy r+w, same size as logits): {sol_us:.1f} us")
    print(f"True SOL (single read of logits):      {true_sol_us:.1f} us")
    print(f"Inductor:     {inductor_us:.1f} us")
    print(f"Best hand:    {best_time:.1f} us ({best_cfg[0]}, BLOCK={best_cfg[1]}, warps={best_cfg[2]})")
    print(f"Speedup over inductor: {inductor_us / best_time:.2f}x")


if __name__ == "__main__":
    benchmark_cross_entropy()
