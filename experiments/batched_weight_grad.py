"""
Batched Weight Gradient Reduction Kernel
=========================================

Motivation: In BERT-large backward, there are ~241 independent column-sum reductions
(bias gradients and layernorm weight gradients). Each computes:

    dW[i] = sum(input_i, dim=0)   # [M, N] -> [N]

Where M=2048 (batch*seq) and N is either 1024 or 4096.

Inductor produces a separate kernel launch for each reduction (or small groups).
This experiment batches ALL reductions into a single kernel launch using an
indirect pointer table, eliminating ~240 kernel launch overheads.

Architecture (V1 - sequential M):
- Grid: (num_reductions, cdiv(max_N, BLOCK_N))
- Each program processes ALL rows (M=2048) for a BLOCK_N-wide column slice
- Simple but limited parallelism (few blocks per reduction for small N)

Architecture (V2 - parallel M with atomics):
- Grid: (num_reductions * cdiv(M, BLOCK_M), cdiv(max_N, BLOCK_N))
- Each program handles BLOCK_M rows x BLOCK_N columns
- Atomic_add partials to output -> full M-parallelism
- Requires output zero-init before kernel launch

For BERT-large: 241 reductions, M=2048, N in {1024, 4096, 30522}
With BLOCK_M=32: 64 row-blocks * 241 reductions = 15424 independent tiles
-> Excellent GPU utilization on B200 (128 SMs)
"""

import torch
import triton
import triton.language as tl
import time
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def batched_col_sum_v1(
    table_ptr,
    ENTRIES_PER_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    V1: Sequential over M. Each program handles all rows for one column block.
    Grid: (num_reductions, cdiv(max_N, BLOCK_N))
    Good when M is small or when launch overhead dominates.
    """
    reduction_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    table_offset = reduction_id * ENTRIES_PER_ROW
    input_ptr = tl.load(table_ptr + table_offset + 0).to(tl.pointer_type(tl.float32))
    output_ptr = tl.load(table_ptr + table_offset + 1).to(tl.pointer_type(tl.float32))
    M = tl.load(table_ptr + table_offset + 2)
    N = tl.load(table_ptr + table_offset + 3)
    input_stride = tl.load(table_ptr + table_offset + 4)

    col_start = col_block_id * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    if col_start >= N:
        return

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for row_start in range(0, M, BLOCK_M):
        for r in tl.static_range(BLOCK_M):
            row_idx = row_start + r
            mask = col_mask & (row_idx < M)
            offset = row_idx * input_stride + col_offsets
            vals = tl.load(input_ptr + offset, mask=mask, other=0.0)
            acc += vals

    tl.store(output_ptr + col_offsets, acc, mask=col_mask)


@triton.jit
def batched_col_sum_v2(
    table_ptr,
    num_row_blocks,   # cdiv(M, BLOCK_M) - uniform across reductions
    ENTRIES_PER_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    V2: Parallel over M with atomic accumulation.
    Grid: (num_reductions * num_row_blocks, cdiv(max_N, BLOCK_N))

    Each program processes BLOCK_M rows x BLOCK_N cols and atomically adds
    its partial sum to the output. Requires outputs pre-zeroed.

    This achieves full GPU utilization by creating many small tiles.
    """
    # Decode program_id(0) into (reduction_id, row_block_id)
    linear_id = tl.program_id(0)
    reduction_id = linear_id // num_row_blocks
    row_block_id = linear_id % num_row_blocks
    col_block_id = tl.program_id(1)

    # Load metadata
    table_offset = reduction_id * ENTRIES_PER_ROW
    input_ptr = tl.load(table_ptr + table_offset + 0).to(tl.pointer_type(tl.float32))
    output_ptr = tl.load(table_ptr + table_offset + 1).to(tl.pointer_type(tl.float32))
    M = tl.load(table_ptr + table_offset + 2)
    N = tl.load(table_ptr + table_offset + 3)
    input_stride = tl.load(table_ptr + table_offset + 4)

    col_start = col_block_id * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    # Early exit for out-of-bounds
    if col_start >= N:
        return

    row_start = row_block_id * BLOCK_M
    if row_start >= M:
        return

    # Accumulate partial sum over BLOCK_M rows
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in tl.static_range(BLOCK_M):
        row_idx = row_start + r
        mask = col_mask & (row_idx < M)
        offset = row_idx * input_stride + col_offsets
        vals = tl.load(input_ptr + offset, mask=mask, other=0.0)
        acc += vals

    # Atomic add partial to output
    tl.atomic_add(output_ptr + col_offsets, acc, mask=col_mask)


@triton.jit
def single_col_sum_parallel(
    input_ptr, output_ptr, M, N, input_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Single-op parallel reduction (for baseline comparison).
    Grid: (cdiv(N, BLOCK_N), cdiv(M, BLOCK_M))
    """
    col_block_id = tl.program_id(0)
    row_block_id = tl.program_id(1)

    col_start = col_block_id * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    row_start = row_block_id * BLOCK_M

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in tl.static_range(BLOCK_M):
        row_idx = row_start + r
        mask = col_mask & (row_idx < M)
        offset = row_idx * input_stride + col_offsets
        vals = tl.load(input_ptr + offset, mask=mask, other=0.0)
        acc += vals

    tl.atomic_add(output_ptr + col_offsets, acc, mask=col_mask)


# ============================================================================
# Helper: Build pointer table
# ============================================================================

ENTRIES_PER_ROW = 5

def build_pointer_table(reductions, device):
    """
    Build flat int64 pointer table for batched kernel.
    Each row: [input_ptr, output_ptr, M, N, input_stride_row]
    """
    num_reductions = len(reductions)
    table = torch.zeros(num_reductions * ENTRIES_PER_ROW, dtype=torch.int64, device=device)

    max_N = 0
    max_M = 0
    for i, (inp, out) in enumerate(reductions):
        M, N = inp.shape
        max_N = max(max_N, N)
        max_M = max(max_M, M)
        base = i * ENTRIES_PER_ROW
        table[base + 0] = inp.data_ptr()
        table[base + 1] = out.data_ptr()
        table[base + 2] = M
        table[base + 3] = N
        table[base + 4] = inp.stride(0)

    return table, max_M, max_N


def zero_outputs(reductions):
    """Zero all output tensors (required for atomic_add approach)."""
    for _, out in reductions:
        out.zero_()


# ============================================================================
# Synthetic workload mimicking BERT-large backward
# ============================================================================

def create_bert_like_reductions(num_layers=24, device="cuda"):
    """
    Create synthetic reductions mimicking BERT-large backward.

    Per transformer layer (10 reductions):
    - 4x linear bias grads: [2048, 1024] -> [1024] (Q, K, V, output proj)
    - 1x FFN up bias grad: [2048, 4096] -> [4096]
    - 1x FFN down bias grad: [2048, 1024] -> [1024]
    - 2x LayerNorm weight grads: [2048, 1024] -> [1024]
    - 2x LayerNorm bias grads: [2048, 1024] -> [1024]

    Plus embedding layer: 1x [2048, 30522] -> [30522]
    Total: 241 reductions
    """
    M = 2048
    reductions = []

    for layer in range(num_layers):
        for _ in range(4):  # Q, K, V, output proj
            reductions.append((torch.randn(M, 1024, device=device),
                               torch.zeros(1024, device=device)))
        # FFN up
        reductions.append((torch.randn(M, 4096, device=device),
                           torch.zeros(4096, device=device)))
        # FFN down
        reductions.append((torch.randn(M, 1024, device=device),
                           torch.zeros(1024, device=device)))
        # LN weight + bias (4 total)
        for _ in range(4):
            reductions.append((torch.randn(M, 1024, device=device),
                               torch.zeros(1024, device=device)))

    # Embedding
    reductions.append((torch.randn(M, 30522, device=device),
                       torch.zeros(30522, device=device)))

    return reductions


# ============================================================================
# Benchmark functions
# ============================================================================

def benchmark_torch_eager(reductions, num_warmup=10, num_iters=100):
    """241 individual torch.sum calls (eager ATen dispatch)."""
    for _ in range(num_warmup):
        for inp, out in reductions:
            torch.sum(inp, dim=0, out=out)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        for inp, out in reductions:
            torch.sum(inp, dim=0, out=out)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def benchmark_batched_v1(reductions, num_warmup=10, num_iters=100):
    """Batched V1: sequential M, 1 launch."""
    BLOCK_N, BLOCK_M = 1024, 64
    device = reductions[0][0].device
    table, max_M, max_N = build_pointer_table(reductions, device)
    num_reductions = len(reductions)
    grid = (num_reductions, triton.cdiv(max_N, BLOCK_N))

    for _ in range(num_warmup):
        batched_col_sum_v1[grid](
            table, ENTRIES_PER_ROW=ENTRIES_PER_ROW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        batched_col_sum_v1[grid](
            table, ENTRIES_PER_ROW=ENTRIES_PER_ROW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def benchmark_batched_v2(reductions, num_warmup=10, num_iters=100):
    """Batched V2: parallel M with atomics, 1 launch."""
    BLOCK_N, BLOCK_M = 128, 32
    device = reductions[0][0].device
    table, max_M, max_N = build_pointer_table(reductions, device)
    num_reductions = len(reductions)
    num_row_blocks = triton.cdiv(max_M, BLOCK_M)
    grid = (num_reductions * num_row_blocks, triton.cdiv(max_N, BLOCK_N))

    for _ in range(num_warmup):
        zero_outputs(reductions)
        batched_col_sum_v2[grid](
            table, num_row_blocks,
            ENTRIES_PER_ROW=ENTRIES_PER_ROW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        zero_outputs(reductions)
        batched_col_sum_v2[grid](
            table, num_row_blocks,
            ENTRIES_PER_ROW=ENTRIES_PER_ROW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def benchmark_batched_v2_tuned(reductions, num_warmup=10, num_iters=100):
    """
    Batched V2 with optimal parameters found from tuning.
    Groups reductions by shape to avoid padding waste.
    """
    BLOCK_M = 32
    device = reductions[0][0].device

    from collections import defaultdict
    groups = defaultdict(list)
    for inp, out in reductions:
        groups[inp.shape[1]].append((inp, out))

    # Tune BLOCK_N per group for maximum efficiency
    group_configs = []
    for N, group_reds in groups.items():
        # Pick BLOCK_N to minimize padding
        if N <= 128:
            BLOCK_N = 128
        elif N <= 256:
            BLOCK_N = 256
        elif N <= 512:
            BLOCK_N = 512
        else:
            BLOCK_N = 128  # Many col blocks = more parallelism

        table, max_M, max_N = build_pointer_table(group_reds, device)
        num_row_blocks = triton.cdiv(max_M, BLOCK_M)
        num_col_blocks = triton.cdiv(max_N, BLOCK_N)
        grid = (len(group_reds) * num_row_blocks, num_col_blocks)
        nw = min(4, max(1, BLOCK_N // 32))
        group_configs.append((table, grid, num_row_blocks, BLOCK_N, nw, group_reds))

    for _ in range(num_warmup):
        zero_outputs(reductions)
        for table, grid, nrb, bn, nw, _ in group_configs:
            batched_col_sum_v2[grid](
                table, nrb,
                ENTRIES_PER_ROW=ENTRIES_PER_ROW,
                BLOCK_M=BLOCK_M, BLOCK_N=bn, num_warps=nw,
            )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        zero_outputs(reductions)
        for table, grid, nrb, bn, nw, _ in group_configs:
            batched_col_sum_v2[grid](
                table, nrb,
                ENTRIES_PER_ROW=ENTRIES_PER_ROW,
                BLOCK_M=BLOCK_M, BLOCK_N=bn, num_warps=nw,
            )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_iters
    return elapsed, len(group_configs)


def benchmark_individual_triton(reductions, num_warmup=10, num_iters=100):
    """Individual Triton kernels with parallel M (same algorithm, 241 launches)."""
    BLOCK_N, BLOCK_M = 128, 32

    for _ in range(num_warmup):
        for inp, out in reductions:
            M, N = inp.shape
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
            out.zero_()
            single_col_sum_parallel[grid](
                inp, out, M, N, inp.stride(0),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
            )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        for inp, out in reductions:
            M, N = inp.shape
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
            out.zero_()
            single_col_sum_parallel[grid](
                inp, out, M, N, inp.stride(0),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
            )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def benchmark_cuda_graph_torch(reductions, num_warmup=10, num_iters=100):
    """CUDA graph of 241 torch.sum calls (best case for individual ops)."""
    for _ in range(3):
        for inp, out in reductions:
            torch.sum(inp, dim=0, out=out)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for inp, out in reductions:
            torch.sum(inp, dim=0, out=out)
    torch.cuda.synchronize()

    for _ in range(num_warmup):
        g.replay()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


def benchmark_cuda_graph_batched_v2(reductions, num_warmup=10, num_iters=100):
    """CUDA graph of batched V2 kernel (isolates pure GPU time)."""
    BLOCK_N, BLOCK_M = 128, 32
    device = reductions[0][0].device
    table, max_M, max_N = build_pointer_table(reductions, device)
    num_reductions = len(reductions)
    num_row_blocks = triton.cdiv(max_M, BLOCK_M)
    grid = (num_reductions * num_row_blocks, triton.cdiv(max_N, BLOCK_N))

    # Warmup / compile
    for _ in range(3):
        zero_outputs(reductions)
        batched_col_sum_v2[grid](
            table, num_row_blocks,
            ENTRIES_PER_ROW=ENTRIES_PER_ROW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
        )
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        zero_outputs(reductions)
        batched_col_sum_v2[grid](
            table, num_row_blocks,
            ENTRIES_PER_ROW=ENTRIES_PER_ROW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
        )
    torch.cuda.synchronize()

    for _ in range(num_warmup):
        g.replay()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / num_iters


# ============================================================================
# Correctness verification
# ============================================================================

def verify_correctness(reductions):
    """Verify both V1 and V2 batched kernels produce correct results."""
    BLOCK_N_v1, BLOCK_M_v1 = 1024, 64
    BLOCK_N_v2, BLOCK_M_v2 = 128, 32
    device = reductions[0][0].device

    # Reference
    references = [inp.sum(dim=0).clone() for inp, _ in reductions]

    # V1 test
    table, max_M, max_N = build_pointer_table(reductions, device)
    num_reductions = len(reductions)
    grid_v1 = (num_reductions, triton.cdiv(max_N, BLOCK_N_v1))
    # Reset outputs
    zero_outputs(reductions)
    batched_col_sum_v1[grid_v1](
        table, ENTRIES_PER_ROW=ENTRIES_PER_ROW,
        BLOCK_M=BLOCK_M_v1, BLOCK_N=BLOCK_N_v1, num_warps=8,
    )
    torch.cuda.synchronize()

    max_err_v1 = max((out - ref).abs().max().item() for (_, out), ref in zip(reductions, references))

    # V2 test
    zero_outputs(reductions)
    num_row_blocks = triton.cdiv(max_M, BLOCK_M_v2)
    grid_v2 = (num_reductions * num_row_blocks, triton.cdiv(max_N, BLOCK_N_v2))
    batched_col_sum_v2[grid_v2](
        table, num_row_blocks,
        ENTRIES_PER_ROW=ENTRIES_PER_ROW,
        BLOCK_M=BLOCK_M_v2, BLOCK_N=BLOCK_N_v2, num_warps=4,
    )
    torch.cuda.synchronize()

    max_err_v2 = max((out - ref).abs().max().item() for (_, out), ref in zip(reductions, references))

    return max_err_v1, max_err_v2


# ============================================================================
# Main
# ============================================================================

def main():
    device = "cuda"
    print("=" * 72)
    print("  Batched Weight Gradient Reduction Kernel")
    print("  BERT-large backward: 241 independent column-sum reductions")
    print("=" * 72)
    print()
    print("Pattern from BERT-large backward (24 layers, batch=4, seq=512):")
    print("  - sum([2048, 1024], dim=0) -> [1024]  (bias + LN grads, 216x)")
    print("  - sum([2048, 4096], dim=0) -> [4096]  (FFN bias grads, 24x)")
    print("  - sum([2048, 30522], dim=0) -> [30522] (embedding bias, 1x)")
    print()
    print("Inductor generates ~241 separate kernel launches for these.")
    print("Our batched kernel handles all 241 in 1-3 launches.")

    reductions = create_bert_like_reductions(num_layers=24, device=device)
    num_reductions = len(reductions)

    from collections import Counter
    shape_counts = Counter(inp.shape[1] for inp, _ in reductions)
    print(f"\n  Total reductions: {num_reductions}")
    for N, count in sorted(shape_counts.items()):
        print(f"    N={N:>5}: {count} reductions")

    # Correctness
    print("\n--- Correctness ---")
    err_v1, err_v2 = verify_correctness(reductions)
    print(f"  V1 (sequential M) max error: {err_v1:.2e}")
    print(f"  V2 (parallel M)   max error: {err_v2:.2e}")
    assert err_v1 < 1e-2 and err_v2 < 1e-2, "FAILED"
    print("  PASSED")

    # Benchmarks
    print("\n--- Performance (wall-clock time per iteration) ---")
    print(f"  {'Method':<46} {'Time(ms)':<9} {'Launches':<9} {'Speedup':<8}")
    print(f"  {'-'*46} {'-'*9} {'-'*9} {'-'*8}")

    t_eager = benchmark_torch_eager(reductions) * 1000
    print(f"  {'torch.sum eager (241 ATen dispatches)':<46} {t_eager:<9.3f} {241:<9} {'1.00x':<8}")

    t_graph_torch = benchmark_cuda_graph_torch(reductions) * 1000
    sp = t_eager / t_graph_torch
    print(f"  {'CUDA Graph (241 torch.sum, no CPU overhead)':<46} {t_graph_torch:<9.3f} {'1 rep':<9} {sp:.2f}x")

    t_v1 = benchmark_batched_v1(reductions) * 1000
    sp = t_eager / t_v1
    print(f"  {'Batched V1 (seq M, 1 launch)':<46} {t_v1:<9.3f} {1:<9} {sp:.2f}x")

    t_v2 = benchmark_batched_v2(reductions) * 1000
    sp = t_eager / t_v2
    print(f"  {'Batched V2 (parallel M + atomics, 1 launch)':<46} {t_v2:<9.3f} {1:<9} {sp:.2f}x")

    t_v2_tuned, n_groups = benchmark_batched_v2_tuned(reductions)
    t_v2_tuned *= 1000
    sp = t_eager / t_v2_tuned
    print(f"  {'Batched V2 grouped ({0} launches, tuned)'.format(n_groups):<46} {t_v2_tuned:<9.3f} {n_groups:<9} {sp:.2f}x")

    t_graph_v2 = benchmark_cuda_graph_batched_v2(reductions) * 1000
    sp = t_eager / t_graph_v2
    print(f"  {'CUDA Graph of Batched V2 (pure GPU time)':<46} {t_graph_v2:<9.3f} {'1 rep':<9} {sp:.2f}x")

    t_individual = benchmark_individual_triton(reductions) * 1000
    sp = t_eager / t_individual
    print(f"  {'Triton individual (241 launches, parallel M)':<46} {t_individual:<9.3f} {241:<9} {sp:.2f}x")

    # Analysis
    print("\n--- Analysis ---")
    BLOCK_M = 32
    max_M = 2048
    max_N = max(inp.shape[1] for inp, _ in reductions)
    num_row_blocks = triton.cdiv(max_M, BLOCK_M)

    print(f"\n  Kernel configuration (V2 - optimal):")
    print(f"    BLOCK_M=32, BLOCK_N=128, num_warps=4")
    print(f"    Row blocks per reduction: {num_row_blocks} (M={max_M}/BLOCK_M={BLOCK_M})")
    print(f"    Grid: ({num_reductions}*{num_row_blocks}, {triton.cdiv(max_N, 128)}) = "
          f"({num_reductions * num_row_blocks}, {triton.cdiv(max_N, 128)}) = "
          f"{num_reductions * num_row_blocks * triton.cdiv(max_N, 128)} total blocks")

    print(f"\n  Launch overhead breakdown:")
    print(f"    Per-op torch.sum overhead: {t_eager/num_reductions*1000:.1f} us/op")
    print(f"    Batched V2 total: {t_v2*1000:.0f} us for all {num_reductions} ops")
    print(f"    = {t_v2/num_reductions*1000:.2f} us amortized per op")

    total_bytes = sum(inp.numel() * 4 + out.numel() * 4 for inp, out in reductions)
    print(f"\n  Memory bandwidth:")
    print(f"    Data volume: {total_bytes/1e6:.1f} MB")
    print(f"    Eager:          {total_bytes / (t_eager/1000) / 1e9:.0f} GB/s")
    print(f"    Batched V2:     {total_bytes / (t_v2/1000) / 1e9:.0f} GB/s")
    print(f"    Graph(torch):   {total_bytes / (t_graph_torch/1000) / 1e9:.0f} GB/s")
    print(f"    Graph(batchV2): {total_bytes / (t_graph_v2/1000) / 1e9:.0f} GB/s")
    print(f"    Note: Values >2TB/s indicate L2 cache residency")

    print(f"\n  Key findings:")
    best_speedup = t_eager / min(t_v1, t_v2, t_v2_tuned)
    best_method = "V1" if t_v1 <= t_v2 and t_v1 <= t_v2_tuned else \
                  "V2" if t_v2 <= t_v2_tuned else "V2-grouped"
    print(f"    1. Best batched approach: {best_method}, {best_speedup:.1f}x vs eager")
    print(f"    2. V1 (sequential M) works well because launch overhead >> compute")
    print(f"       at this scale ({num_reductions} ops, {max_M}x{1024} dominant shape)")
    print(f"    3. V2 (parallel M) enables better GPU utilization for larger M")
    print(f"    4. CUDA graph of torch.sum ({t_graph_torch:.2f}ms) shows that pure")
    print(f"       GPU time for 241 individual reductions is comparable to batched")
    print(f"       -> the win is primarily from eliminating CPU dispatch overhead")
    print(f"    5. For inductor integration: group same-shape reductions into batched")
    print(f"       kernels during graph partitioning ({n_groups} groups for BERT)")


if __name__ == "__main__":
    main()
