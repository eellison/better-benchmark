# sum_sum_sum_766837223ded

## Summary

- Model: Layernorm-backward (row + transpose + column reductions)
- Oracle: `oracle_multi_output_reduction.py`
- Classification: MULTI_OUTPUT_SHARED_REDUCTION
- Ratio: 1.109x (oracle 10.24us, compile 11.36us)
- Kernel count: Inductor multiple kernels, Oracle 1 fused full-scope multi-output reduction

## Root Cause

The oracle fuses the full row-wise layernorm-backward algebra with the side-output materialization and then computes the three column reductions in one Triton reduction kernel.

Inductor emits a less direct multi-kernel schedule for the dependent row reductions, side transpose, and independent column reductions. The scheduler does not form a custom full-scope multi-output reduction plan that reuses the row-local scalars across the materialized transpose and the later reductions.

The 1.109x gap is modest but comes from:
1. Extra kernel launch overhead
2. Materializing the transposed intermediate
3. Cannot keep row-local scalars live for column accumulation

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 11.36 | 1.109x |
| multi_kernel=2 | 19.55 | 1.909x (WORSE) |
| multi_kernel=3 | 22.24 | 2.172x (WORSE) |

multi_kernel=2/3 significantly worsen performance. Default CDT is the best config.

## Fix Assessment

**Design doc** -- requires SCHEDULER_FUSION enhancement (same pattern as sum_sum_sum_728df9734dbe).

### What's needed:
Teach the scheduler/codegen to group the dependent row-reduction plus transpose plus column-reduction pattern into a small coordinated kernel set. This is the same fundamental issue as other layernorm-backward multi-output repros.

### Files:
- `torch/_inductor/scheduler.py` (multi-output reduction plan)
- `torch/_inductor/codegen/triton.py` (full-scope reduction codegen)
