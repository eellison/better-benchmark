# pointwise_5d7c3bd1f499 — BART Causal Mask

## Summary
- **Ratio**: 1.093x (oracle: 8.99us, compile: 9.82us)
- **Classification**: BANDWIDTH_BOUND (marginal codegen overhead)
- **Oracle kernels**: 1 (2D-tiled causal mask kernel)
- **Inductor kernels**: 1

## Root Cause

Both the oracle and Inductor emit a single kernel that computes the BART causal mask: for each position (batch, q, k), it checks `k <= q` and `source_mask[batch, k] != 0`, writing a bool result to the `[8, 1, 1024, 1024]` output.

The oracle uses a 2D tiled approach (BLOCK_Q=16, BLOCK_K=128) with grid=(8, 64, 8), loading the source mask once per k-block and broadcasting across query positions. Inductor uses a flat 1D pointwise kernel that iterates over all 8M elements, loading the source mask with `evict_last` per-element.

The 9.3% gap is small and comes from:
1. The oracle's 2D tiling amortizes source mask loads (one load serves 16 query positions)
2. Inductor's flat indexing requires decomposing the linear index into (batch, q, k) coordinates at every element

The oracle docstring itself classifies this as BANDWIDTH_BOUND. At 8.99us for 8MB of bool output, the kernel is already near the bandwidth floor.

## Config Exploration

Standard configs active. coordinate_descent_tuning helps Inductor pick a good XBLOCK but cannot change the fundamental 1D vs 2D tiling approach. The gap is marginal (< 10% typically considered noise-adjacent for this scale).

## Design Doc

**Why the gap is marginal**: The output is 8MB of bool data. At ~2TB/s HBM bandwidth, the floor is ~4us. Both implementations are 2-2.5x above floor due to write amplification and occupancy. The 9.3% gap comes from Inductor's simpler 1D tiling vs the oracle's structured 2D blocking.

**Potential improvement**: A codegen enhancement that detects "broadcast source + structured predicate" patterns and emits 2D-tiled kernels. However, the absolute savings (~0.83us) may not justify the complexity.

**Affected files**: `torch/_inductor/codegen/triton.py` (tiling strategy for broadcast patterns)
