# var_mean_var_mean_aabd7680d8e0 - oracle_chained_bn_recompute

## Classification
RECOMPUTE_FUSION

## Benchmark Results
- Oracle: 20.54 us (2 kernels)
- Inductor: 31.87 us (1 kernel)
- Ratio: 1.551x

## Config Exploration
| Config | Time (us) |
|--------|-----------|
| combo_kernels + CDT | 31.87 |
| combo + multi_kernel=3 + per_subkernel_blocks | 31.13 |
| combo + multi3 + cooperative_reductions | 29.40 |

cooperative_reductions helps slightly (31.87 -> 29.40), but doesn't close the gap.

## Root Cause

The repro (from timm visformer_small train) computes chained batch norms:
1. BN1: var_mean over [128, 768, 7, 7], affine, running stats -> output [128, 768, 7, 7]
2. Residual add: bn1_output + arg142_1 [1, 768, 7, 7]
3. BN2: var_mean over the result [128, 768, 7, 7], affine, running stats -> output [128, 768, 7, 7]

The oracle uses 2 kernels:
1. Kernel 1: BN1 reduction (var_mean + running stats + rsqrt output)
2. Kernel 2: **Recomputes** BN1's affine epilogue (sub_mean * rsqrt * weight + bias + residual) directly inside BN2's reduction, avoiding materialization of the intermediate [128, 768, 7, 7] tensor

Inductor uses 1 kernel that fuses everything but is slower because:
- It must materialize the intermediate BN1 output to feed into BN2's reduction
- Even though it's "1 kernel", it does TWO passes over the data (the BN template reads the normalized output for the second reduction)
- The single large kernel has register pressure issues with 768 channels and two full BN passes

The oracle's 2-kernel approach is FASTER than Inductor's 1-kernel approach because:
- Kernel 2 recomputes the cheap BN1 affine epilogue (mul + add, ~5 FLOPs/element) instead of reading the 128*768*7*7*4 = 19.2 MB intermediate
- This saves one full buffer read (19.2 MB) at the cost of a few extra FLOPs
- The recomputation is register-cheap and L2-friendly

## Kernel Count
- Oracle: 2 kernels (recompute strategy)
- Inductor: 1 kernel (but materializes intermediate internally)

## Fix Assessment
**Design doc** - Requires recomputation-aware fusion in the norm template.

### What's needed:
The scheduler/codegen should recognize when:
1. BN1 -> pointwise -> BN2 forms a chain
2. The pointwise intermediate (BN1 affine + residual) is cheap to recompute
3. Recomputing it inside BN2's reduction is cheaper than materializing and reading back

This is the classic "recompute vs materialize" trade-off. The oracle demonstrates that for chained BNs with moderate channel count (768) and spatial dims (7x7), recomputation wins.

### Implementation approach:
In the scheduler, when a reduction node (BN2) consumes the output of another reduction node (BN1) through a cheap pointwise chain:
1. Estimate cost of materializing intermediate: N*C*H*W*4 bytes read
2. Estimate cost of recomputing: (FLOPs per element) * elements / peak_flops
3. If recompute is cheaper, emit a 2-kernel plan where kernel 2 recomputes BN1's epilogue

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - recompute vs materialize decision
- `/tmp/pytorch-work/torch/_inductor/ir.py` - should_realize_on_reuse decides when to materialize
- `/tmp/pytorch-work/torch/_inductor/choices.py` - reduction strategy for chained norms

### Difficulty: Hard
This requires the scheduler to reason about recomputation profitability, which is a more sophisticated optimization than simple fusion. The current scheduler either fuses (1 kernel) or materializes (2 kernels reading from buffer), but doesn't support the "2 kernels with recomputation" strategy.

### Affected repro count:
Chained BN patterns are common in vision models. This likely affects multiple repros in the var_mean_var_mean family.
