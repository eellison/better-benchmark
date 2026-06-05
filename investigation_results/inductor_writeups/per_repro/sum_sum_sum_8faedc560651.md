# sum_sum_sum_8faedc560651

## Summary

- Model: Grouped multi-output reduction (BN-backward style, 32 groups x 16 channels)
- Oracle: `oracle_multi_output.py`
- Classification: MULTI_OUTPUT_SHARED_REDUCTION
- Ratio: 3.15x (oracle 6.816us, compile 21.472us)
- Kernel count: Inductor multiple kernels, Oracle 1 kernel (32 programs, one per group)

## Root Cause

The oracle computes the full forward return tuple in one Triton program family with 32 groups (one program per group). Each program:
1. Loads `mm` masked with `where(mask, 0, mm)` semantics
2. Computes per-`[64, 16]` grouped 16-wide summaries (weighted sums)
3. Derives dependent BN-backward-style scalars (delta, grouped_scale, grouped_bias)
4. Writes the full `[64, 512, 1, 1]` pointwise output
5. Writes both returned `[512]` batch reductions (out1, out2)

Inductor decomposes this into separate generic kernels: singleton reductions, grouped reductions, dependent pointwise epilogue, and sibling batch reductions -- each materializing intermediates.

The 3.15x gap comes from the scheduler not forming a single grouped multi-output reduction node whose row-local summaries feed both a full tensor side output and two compatible column reductions.

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 21.472 | 3.15x |
| combo_kernels + CDT + mk=3 | 22.432 | 3.29x |
| multi_kernel=2 | 19.584 | 2.87x |

multi_kernel=2 provides slight improvement but doesn't close the 3x gap. The fundamental issue is the scheduler cannot form the grouped multi-output reduction.

## Fix Assessment

**Design doc** -- requires SCHEDULER_FUSION enhancement for dependent multi-output reduction scheduling.

### What's needed:
Add dependent multi-output reduction scheduling for fixed small grouped reductions. The scheduler needs to recognize that the shared masked producer can be read once and all three outputs (full tensor + two column reductions) can be finalized in the same generated kernel.

### Files:
- `torch/_inductor/scheduler.py` (multi-output reduction node formation)
- `torch/_inductor/codegen/triton.py` (grouped reduction codegen)
