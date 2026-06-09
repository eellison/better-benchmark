# pointwise_eae6c9133626

## Classification: BORDERLINE_AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/pointwise_eae6c9133626/oracle_split_swiglu.py`
- Model: vllm_Qwen_Qwen3-30B-A3B_000
- Correctness: PASS
- Oracle: 17.89 us
- Compile (cd=True, combo=True): 18.98 us
- Ratio: 1.061 (marginal gap, within noise)
- Status: GOOD (barely above 1.05 threshold)

## Config Exploration

| Config | Compile (us) | Ratio | Notes |
|--------|-------------|-------|-------|
| Default (cd=True, combo=True) | 18.98 | 1.061 | Baseline |
| multi_kernel=2 | ~64.52 (no CUDA graph) | - | Slower without graph capture |
| multi_kernel=3 | ~31.25 (no CUDA graph) | - | Slower without graph capture |
| use_fast_math=True | ~30.30 (no CUDA graph) | - | N/A (both use libdevice.exp) |

Note: Config exploration without CUDA graph replay shows higher absolute times but the gap is only significant with graph replay (harness measurement).

## Root Cause Analysis

The repro computes split-SwiGLU: splits bf16 [16384, 1536] into two [16384, 768] halves, applies silu(x) = x / (exp(-x) + 1) on the first half in fp32, casts back to bf16, multiplies by second half.

**Oracle approach**: 2D autotuned kernel (BLOCK_M rows x BLOCK_N cols) that reads both halves of each row using direct offset arithmetic (lhs at row*1536+col, rhs at row*1536+768+col). No division/modulo needed.

**Inductor approach**: Single fused kernel with flat 1D indexing. Uses `x0 = xindex % 768` and `x1 = xindex // 768` to recover row/col from flat index, then loads from `in_ptr0 + (x0 + 1536*x1)` and `in_ptr0 + (768 + x0 + 1536*x1)`. Both use `libdevice.exp` for precision.

**Why the gap is marginal (1.06x)**: The div/mod overhead for split-dimension indexing adds a small cost per element. On B200 with CUDA graph replay, this manifests as ~1 us difference (17.89 vs 18.98 us). This gap is within run-to-run noise (previous measurement showed oracle 20.58us vs compile 18.66us = 0.907x BAD_ORACLE).

## Kernel Count

- Inductor: 1 kernel (triton_poi_fused_add_convert_element_type_div_exp_mul_neg_split_0)
- Oracle: 1 kernel (_split_swiglu_kernel)

## Design Issue

Same root cause as BROADCAST_AFFINE_FLAT_TILING family: Inductor's 1D flat pointwise codegen uses div/mod to recover multi-dimensional indices from split-view patterns. A 2D tiling approach that processes rows directly would avoid this overhead. However, at 1.06x this is borderline - measurement noise across runs.

## Status

Borderline - gap fluctuates around 1.0x across measurements. Previous run showed compile winning (0.907x). Not actionable.
