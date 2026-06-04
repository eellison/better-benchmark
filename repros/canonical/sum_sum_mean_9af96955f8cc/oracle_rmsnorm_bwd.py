"""
Oracle for sum_sum_mean_9af96955f8cc (Qwen3-30B RMSNorm backward).

Gap diagnosis:
  Classification: SCHEDULER_FUSION (reduction chaining)
  What oracle does differently: Fuses the expert-sum reduction (K2) and RMSNorm
    reduction (K3) into a single persistent Triton kernel, eliminating an 8MB
    intermediate write+read between the two phases.
  What Inductor change would fix: Inductor needs to recognize that when a
    reduction kernel followed by another reduction share the same intermediate
    dimension, they can be chained into a single persistent kernel that keeps
    data in registers (reduction chaining optimization).

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
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

# --- Configuration ---
REPRO_ID = "sum_sum_mean_9af96955f8cc"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


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
# Oracle forward
# ============================================================================

def oracle_forward(inputs):
    """Run the oracle fused kernel (2 kernel launches total).

    SCOPE INVARIANT: Accepts the same inputs as Repro.forward() and returns
    the same output (same shape, dtype, strides).

    Args:
        inputs: list of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    # Unpack inputs (first 6 are tensors, last 3 are shape params)
    getitem_54 = inputs[0]   # [2048, 8] f32
    getitem_57 = inputs[1]   # [16384] i64
    grouped_mm_7 = inputs[2] # [16384, 2048] bf16
    unsqueeze_36 = inputs[3] # [16384, 1] b8
    add_35 = inputs[4]       # [4, 512, 2048] bf16
    arg47_1 = inputs[5]      # [2048] bf16

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


# --- CLI entry point ---
def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=2e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=2e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    args = parser.parse_args()

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        result = bench_oracle(
            oracle_forward,
            instance,
            inputs,
            REPRO_ID,
            warmup=args.warmup,
            rep=args.rep,
        )
        if result["status"] == "BAD_ORACLE":
            print(f"WARNING: oracle is slower than compile "
                  f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
