"""
Oracle kernel for sum_sum_mean_9af96955f8cc (Qwen3-30B RMSNorm backward).

=== Investigation Summary ===

The 1.9x gap (87us default vs 46us combo_looped) comes from:

1. DEFAULT generates 4 kernels:
   - K0: sum [2048, 8] -> [2048, 1]  (trivial, <1us)
   - K1: index_put to build inverse permutation  (trivial, <1us)
   - K2: Fused gather+scale+mask+sum-over-8-experts. Grid=4M, reduction=8.
          Reads grouped_mm_7 [16384, 2048] = 64MB. Writes [2048, 2048] = 8MB.
   - K3: RMSNorm (add + x^2 + mean + rsqrt + weight*x). Reads 8MB intermediate + 8MB add_35.
          Persistent reduction over 2048 cols. Writes 8MB output.

   Total bandwidth: 64MB (read grouped_mm) + 8MB (write intermediate) + 8MB (read intermediate)
                  + 8MB (read add_35) + 8MB (write output) = 96MB
   At ~2.3 TB/s effective BW -> 96MB / 2.3TB/s = ~42us theoretical
   Actual: 87us because of:
   - 4 kernel launches (even in CUDAGraph, dispatch overhead)
   - K2 has 4M grid elements with only 8-element reduction (under-utilized warps)
   - K3 has only 2048 grid elements (under-occupancy)

2. COMBO_LOOPED / COORDINATE_DESCENT reduces to ~46us by:
   - Better tile sizes via coordinate descent tuning in K2
   - The key improvement is actually CD tuning the big kernel, not combo fusion

3. ORACLE achieves 34-36us by:
   - Fusing K2+K3 into a SINGLE persistent kernel with grid=[2048]
   - Each program processes one full output row (2048 columns) in registers
   - Eliminates the 8MB intermediate write+read (saves 16MB bandwidth)
   - Total bandwidth: 64MB (grouped_mm) + 8MB (add_35) + 4KB (weight) + 8MB (output) = 80MB
   - Achieves ~2.3 TB/s effective bandwidth = near memory BW ceiling for this access pattern

=== What structural change would close the gap? ===

Inductor needs to recognize that when a reduction kernel (K2, sum over 8 experts with
4M output elements) is followed by another reduction kernel (K3, RMSNorm over 2048 cols
with 2048 output rows), and the intermediate tensor's inner dimension equals the second
kernel's reduction dimension, these can be fused into a SINGLE persistent kernel where:
- The grid matches the SECOND kernel's outer dimension (2048 rows)
- Each program tile-loops over the first kernel's reduction while accumulating
  both the first kernel's sum AND the second kernel's sum-of-squares

This is essentially a "reduction chaining" optimization where the intermediate never
materializes to global memory. The combo_looped framework partially achieves this
by running sub-kernels sequentially within one dispatch, but the full fusion
(keeping data in registers across the two phases) requires Inductor to recognize
the pattern and generate a single fused Triton kernel.
"""
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import parse_shapes_config


# ============================================================================
# Prep kernel: compute per-position routing weights and inverse permutation
# ============================================================================

@triton.jit
def prep_routing_and_invperm_kernel(
    # Inputs
    getitem_54_ptr,    # [2048, 8] f32 - raw routing logits
    getitem_57_ptr,    # [16384] i64 - forward permutation
    sum_ptr,           # [2048, 1] f32 - precomputed row sums of getitem_54
    # Outputs
    routing_weights_ptr,  # [16384] bf16 - routing weight for each position
    inv_perm_ptr,         # [16384] i64 - inverse permutation
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each position i in [0, 16384):
      - Store inv_perm[perm[i]] = i  (build inverse permutation)
      - Compute routing_weight[i] = getitem_54.flat[perm[i]] / row_sum[perm[i] // 8]
        This gives the normalized MoE routing weight for position i.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < 16384

    # Load forward permutation
    perm_idx = tl.load(getitem_57_ptr + offs, mask=mask)

    # Build inverse permutation: inv_perm[perm[i]] = i
    tl.store(inv_perm_ptr + perm_idx, offs, mask=mask)

    # Compute routing weight for position i:
    # index_tensor[i] = flat_routing[getitem_57[i]] = getitem_54.flat[perm[i]] / sum_row[perm[i]//8]
    raw_weight = tl.load(getitem_54_ptr + perm_idx, mask=mask)
    row_of_perm = perm_idx // 8
    row_sum = tl.load(sum_ptr + row_of_perm, mask=mask)
    normalized_weight = (raw_weight / row_sum).to(tl.bfloat16)
    tl.store(routing_weights_ptr + offs, normalized_weight, mask=mask)


# ============================================================================
# Main fused kernel: gather + scale + mask + sum-experts + add + RMSNorm
# ============================================================================

@triton.jit
def fused_oracle_kernel(
    # Inputs
    grouped_mm_ptr,       # [16384, 2048] bf16 - the dominant 64MB tensor
    mask_ptr,             # [16384] bool - MoE routing mask
    add_35_ptr,           # [2048, 2048] bf16 - residual
    rmsnorm_weight_ptr,   # [2048] bf16 - RMSNorm weight
    routing_weights_ptr,  # [16384] bf16 - precomputed per-position routing weights
    inv_perm_ptr,         # [16384] i64 - inverse permutation
    # Output
    output_ptr,           # [2048, 2048] bf16
    # Dimensions
    N_ROWS: tl.constexpr,     # 2048
    N_COLS: tl.constexpr,     # 2048
    N_EXPERTS: tl.constexpr,  # 8
    BLOCK_C: tl.constexpr,    # columns per tile (= N_COLS for persistent)
):
    """
    Fully fused oracle kernel. Grid: [2048] - one program per output row.

    Each program:
      1. Loads 8 physical rows from grouped_mm_7 via inverse permutation
      2. Scales each by its routing weight, applies mask
      3. Sums the 8 expert contributions (reduction dim=1 of [2048,8,2048])
      4. Adds the residual (add_35)
      5. Computes RMSNorm: mean(x^2) + eps -> rsqrt -> x * rsqrt * weight
      6. Writes final output

    All 2048 columns are held in registers as a persistent tile (4KB as bf16).
    This eliminates the 8MB intermediate write+read between K2 and K3.
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_C)

    # Load inverse permutation indices for this row's 8 experts
    inv_base = row_idx * N_EXPERTS

    # Accumulate expert contributions in f32 for precision
    expert_sum = tl.zeros([BLOCK_C], dtype=tl.float32)

    for e in tl.static_range(N_EXPERTS):
        phys_row = tl.load(inv_perm_ptr + inv_base + e)
        # Load mask for this physical row
        m = tl.load(mask_ptr + phys_row)
        # Load routing weight for this position
        rw = tl.load(routing_weights_ptr + phys_row).to(tl.float32)
        # Load grouped_mm_7[phys_row, :] -- this is the big 64MB read
        gm_vals = tl.load(grouped_mm_ptr + phys_row * N_COLS + col_offs).to(tl.float32)
        # Apply: where(mask, 0, gm * rw)
        contribution = tl.where(m, 0.0, gm_vals * rw)
        expert_sum += contribution

    # Add residual: add_35[row_idx, :]
    add_val = tl.load(add_35_ptr + row_idx * N_COLS + col_offs).to(tl.float32)
    combined = expert_sum + add_val

    # RMSNorm: rsqrt(mean(x^2) + eps)
    sq = combined * combined
    mean_sq = tl.sum(sq) / N_COLS
    rsqrt_val = tl.math.rsqrt(mean_sq + 1e-6)

    # Apply normalization and weight
    normalized = combined * rsqrt_val
    weight = tl.load(rmsnorm_weight_ptr + col_offs).to(tl.float32)
    result = weight * normalized

    # Store output as bf16
    tl.store(output_ptr + row_idx * N_COLS + col_offs, result.to(tl.bfloat16))


# ============================================================================
# Host-side orchestration
# ============================================================================

def oracle_forward(getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1):
    """Execute the oracle fused kernel (2 kernel launches total)."""
    device = getitem_54.device
    N_ROWS = 2048
    N_COLS = 2048
    N_EXPERTS = 8
    N_TOKENS_EXPANDED = 16384

    # Step 1: Compute row sums of getitem_54 [2048, 8] -> [2048]
    row_sums = getitem_54.sum(dim=-1, keepdim=True)  # [2048, 1]

    # Step 2: Prep kernel - routing weights and inverse permutation
    routing_weights = torch.empty(N_TOKENS_EXPANDED, dtype=torch.bfloat16, device=device)
    inv_perm = torch.empty(N_TOKENS_EXPANDED, dtype=torch.int64, device=device)

    PREP_BLOCK = 1024
    prep_grid = ((N_TOKENS_EXPANDED + PREP_BLOCK - 1) // PREP_BLOCK,)
    prep_routing_and_invperm_kernel[prep_grid](
        getitem_54, getitem_57, row_sums,
        routing_weights, inv_perm,
        BLOCK_SIZE=PREP_BLOCK,
    )

    # Step 3: Main fused kernel
    mask_flat = unsqueeze_36.view(-1)
    add_35_flat = add_35.view(N_ROWS, N_COLS)
    output = torch.empty(N_ROWS, N_COLS, dtype=torch.bfloat16, device=device)

    grid = (N_ROWS,)
    fused_oracle_kernel[grid](
        grouped_mm_7, mask_flat, add_35_flat, arg47_1,
        routing_weights, inv_perm,
        output,
        N_ROWS=N_ROWS,
        N_COLS=N_COLS,
        N_EXPERTS=N_EXPERTS,
        BLOCK_C=N_COLS,
    )

    return output


def reference_forward(getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1):
    """Reference implementation using PyTorch ops (matches Repro.forward)."""
    full_default = torch.full([], 0.0, dtype=torch.bfloat16, device=getitem_54.device)
    sum_dim = getitem_54.sum(-1, True)
    div_tensor = getitem_54 / sum_dim
    convert_bf16 = div_tensor.to(torch.bfloat16)
    view_flat = convert_bf16.view(-1)
    index_tensor = view_flat[getitem_57]
    unsqueeze = index_tensor.unsqueeze(-1)
    mul_tensor = grouped_mm_7 * unsqueeze
    where_self = torch.where(unsqueeze_36, full_default, mul_tensor)

    # Build inverse permutation
    empty_mem = torch.empty(16384, dtype=torch.int64, device=getitem_54.device)
    iota = torch.arange(16384, dtype=torch.int64, device=getitem_54.device)
    inv_perm = empty_mem.scatter_(0, getitem_57, iota)

    index_tensor_1 = where_self[inv_perm]
    view_2d8 = index_tensor_1.view(2048, 8, 2048)
    sum_experts = view_2d8.sum(1)
    view_4d = sum_experts.view(4, 512, 2048)
    add_tensor = add_35 + view_4d
    convert_f32 = add_tensor.float()
    pow_sq = convert_f32 ** 2
    mean_sq = pow_sq.mean(-1, True)
    add_eps = mean_sq + 1e-6
    rsqrt_val = torch.rsqrt(add_eps)
    mul_norm = convert_f32 * rsqrt_val
    convert_bf16_out = mul_norm.to(torch.bfloat16)
    mul_weight = arg47_1 * convert_bf16_out
    output = mul_weight.view(2048, 2048)
    return output


def make_inputs():
    """Generate test inputs."""
    _shapes_config = "(T([2048, 8], f32), T([16384], i64, gen=Perm(16384)), T([16384, 2048], bf16), T([16384, 1], b8), T([4, 512, 2048], bf16), T([2048], bf16), S([2048, 8, 2048]), S([4, 512, 2048]), S([2048, 2048]))"
    inputs = parse_shapes_config(_shapes_config)
    return inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]


def check_correctness():
    """Verify oracle matches reference."""
    torch.manual_seed(42)
    getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1 = make_inputs()

    ref_out = reference_forward(getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1)
    oracle_out = oracle_forward(getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1)

    # Compare
    max_diff = (ref_out.float() - oracle_out.float()).abs().max().item()
    mean_diff = (ref_out.float() - oracle_out.float()).abs().mean().item()
    ref_max = ref_out.float().abs().max().item()
    rel_err = max_diff / (ref_max + 1e-8)

    print(f"Max absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Reference max value: {ref_max:.6f}")
    print(f"Relative error: {rel_err:.6f}")

    # bf16 has ~0.4% relative error tolerance; allow up to 2% for fused accumulation
    if rel_err < 0.02:
        print("CORRECTNESS: PASS")
        return True
    else:
        print("CORRECTNESS: FAIL")
        return False


def benchmark():
    """Benchmark oracle vs compiled default vs compiled with coordinate descent."""
    import torch._dynamo
    import torch._inductor.config as cfg
    from triton.testing import do_bench

    torch.manual_seed(42)
    getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1 = make_inputs()

    # Warmup oracle
    for _ in range(5):
        oracle_forward(getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1)
    torch.cuda.synchronize()

    # Benchmark oracle with CUDAGraph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        oracle_forward(getitem_54, getitem_57, grouped_mm_7, unsqueeze_36, add_35, arg47_1)
    torch.cuda.synchronize()

    oracle_ms = do_bench(lambda: g.replay(), warmup=25, rep=100, return_mode="min")
    oracle_us = oracle_ms * 1000
    print(f"Oracle kernel: {oracle_us:.1f} us")

    # Benchmark compiled default
    sys.path.insert(0, str(Path(__file__).parent))
    from repro import Repro
    mod = Repro()
    all_inputs = list(make_inputs()) + [[2048, 8, 2048], [4, 512, 2048], [2048, 2048]]

    torch._dynamo.reset()
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled(*all_inputs)
        torch.cuda.synchronize()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            compiled(*all_inputs)
        torch.cuda.synchronize()

    default_ms = do_bench(lambda: g2.replay(), warmup=25, rep=100, return_mode="min")
    default_us = default_ms * 1000
    print(f"Compiled default: {default_us:.1f} us")

    # Benchmark compiled with coordinate descent
    cfg.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled_cd(*all_inputs)
        torch.cuda.synchronize()
        g3 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g3):
            compiled_cd(*all_inputs)
        torch.cuda.synchronize()

    cd_ms = do_bench(lambda: g3.replay(), warmup=25, rep=100, return_mode="min")
    cd_us = cd_ms * 1000
    print(f"Compiled (coord descent): {cd_us:.1f} us")
    cfg.coordinate_descent_tuning = False

    print(f"\nSpeedup oracle vs default: {default_us/oracle_us:.2f}x")
    print(f"Speedup oracle vs coord_desc: {cd_us/oracle_us:.2f}x")

    # Memory bandwidth analysis
    # Read: grouped_mm_7 (64MB) + add_35 (8MB) + routing_weights (32KB)
    #      + inv_perm (128KB) + mask (16KB) + weight (4KB)
    # Write: output (8MB)
    total_bytes = 67108864 + 8388608 + 32768 + 131072 + 16384 + 4096 + 8388608
    bw_gbps = total_bytes / (oracle_us * 1e-6) / 1e9
    print(f"\nEffective bandwidth: {bw_gbps:.0f} GB/s")
    print(f"Total bytes: {total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run correctness check only")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if args.check or not args.bench:
        print("=== Correctness Check ===")
        check_correctness()
        print()

    if args.bench or not args.check:
        print("=== Benchmark ===")
        benchmark()
