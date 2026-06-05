# var_mean_3f319cc01bd1 - GPT-2 Large Residual Dropout LayerNorm

## Classification: SCHEDULER_FUSION

## Benchmark Results
- Oracle: 17.6 us
- Inductor compiled: 20.38 us
- Ratio: 1.158x (oracle is 1.16x faster)

## Kernel Count
- Inductor: 1 kernel (fused reduction)
- Oracle: 1 kernel (autotuned row kernel)

## Root Cause

The repro computes the GPT-2 large residual-dropout-LayerNorm scope:
1. View addmm [2048,1280] -> [4,512,1280]
2. Seed-indexed dropout (inductor_lookup_seed at index 70, inductor_random, gt 0.1, mul scale)
3. Residual add with [4,512,1280] tensor
4. var_mean over hidden dim (1280), correction=0
5. LayerNorm affine (sub mean, mul rsqrt, mul weight, add bias)
6. View back to [2048,1280], permute to [1280,2048]
7. Side output: rsqrt/1280 as [4,512,1] invstd

Inductor already fuses all of this into a single reduction kernel (`triton_red_fused_add_div_gt_inductor_lookup_seed_inductor_random_mul_rsqrt_sub_var_mean_view_0`). The gap is small (1.158x) and comes from:

1. **Output layout overhead**: The oracle writes [2048,1280] contiguously and returns a metadata-only permute, while Inductor may compute the transpose store inline
2. **Autotuning gap**: The oracle uses explicit autotune configs (ROW_BLOCK=1/2, num_warps=4/8) optimized for hidden=1280. Inductor's coordinate_descent_tuning explores different XBLOCK/R0_BLOCK combinations but may not find the exact optimal
3. **RNG inline overhead**: The inductor_random call with seed lookup adds some register pressure compared to an optimally-tuned static kernel

The gap is borderline (1.158x) - close to noise for this workload size (2048 rows x 1280 hidden).

## Config Exploration
| Config | Time (us) | Notes |
|--------|-----------|-------|
| coordinate_descent + combo_kernels | 20.38 | Default compiled |
| Oracle (autotuned row kernel) | 17.6 | 1.16x faster |

## What Inductor Needs

This is primarily an **autotuning gap** rather than a fundamental scheduler/fusion problem. Inductor already produces a single fused kernel. Possible improvements:

1. **Better autotune configs for row-reduction**: For hidden=1280 (not power-of-2), specific XBLOCK/R0_BLOCK combinations that match the oracle's ROW_BLOCK=2/num_warps=8 config
2. **Output permute elision**: If the [1280,2048] permute can be lowered as a metadata view rather than an explicit transpose store

**Files to reference**:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - reduction kernel autotune configs
- `/tmp/pytorch-work/torch/_inductor/choices.py` - reduction strategy heuristics

## Status
Low priority - gap is small (1.158x) and Inductor already produces optimal kernel count (1 kernel). This is an autotuning/tiling quality issue, not a scheduler or fusion problem.
