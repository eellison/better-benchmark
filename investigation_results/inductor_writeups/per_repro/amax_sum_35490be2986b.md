# amax_sum_35490be2986b

## Compile: 88.8us, Oracle: 63.0us, Gap: 1.43x

## Diagnosis: NEW_PATTERN

## Root cause: Inductor fuses the Swin relative-position-bias attention softmax into a single kernel that replays the indirect index gather (from [169,16] bias table via [49,49] index tensor) for every one of the 401408 rows. The oracle splits into two kernels: (1) materialize the bias table into a contiguous [16, 49, 49] buffer (38K elements), and (2) run the softmax reading the pre-materialized bias directly. This avoids repeated indirect indexing overhead (negative-index check, non-coalesced table access with stride `head + 16*gathered_index`) across 8192 batch-window elements that all share the same [49,49] index pattern.

## Kernel count: Inductor 1 kernel, Oracle 2 kernels

## Config exploration

- `coordinate_descent_tuning = True`: already enabled, no additional improvement
- `combo_kernels = True`: already enabled
- `assert_indirect_indexing = False`: removes device_assert but does not close gap (different autotune path selected, actually slower)
- `triton.multi_kernel = 3`: no improvement (still 1 kernel)

## Key observations

1. The Inductor kernel is structurally correct -- it fuses everything including the gather into a single `triton_per` kernel with the softmax reduction.
2. The overhead comes from: (a) negative-index wraparound check (`tmp1 < 0; tl.where(...)`) on every row, (b) non-coalesced access to the bias table (`in_ptr2 + (x1 + 16*tmp5)`), (c) L1/L2 thrashing from 401408 rows each doing 49 indirect loads through the same small table.
3. The oracle's 2-kernel approach trades a tiny kernel launch for contiguous bias reads in the hot softmax kernel, winning net.

## Fix path

The scheduler should recognize when a fused reduction kernel contains an indirect-indexed producer that is:
- Small relative to the consumer (38K elements vs 19.7M output elements)
- Invariant across the batch dimension (shared [49,49] index across 8192 windows)
- Accessed non-coalesced within the reduction kernel

In such cases, the scheduler should "hoist" the producer by materializing it into a small buffer before the reduction kernel, rather than fusing it inline. This is a scheduler-level decision in `torch/_inductor/scheduler.py` (fusion scoring / `can_fuse` logic).

Alternatively, the codegen could detect that the index tensor has no batch dependence and hoist the gather computation outside the row loop within the same kernel (register-level hoisting).

## File references

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion decisions (where producer hoisting would be decided)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`: indirect indexing codegen (lines 617-629, 3298-3347)
- `/tmp/pytorch-work/torch/_inductor/ir.py`: `should_realize_on_reuse` (line 10144) -- could be extended to account for indirect-indexing cost

## Related repros

- `amax_sum_17ab35828f89`: same pattern, [4096,49,49], 32 heads, 1.64x gap
- `amax_sum_1a6624550d06`: same pattern + window mask, [8192,49,49], 16 heads, 1.34x gap
- All are Swin-transformer relative-position-bias attention softmax variants

## Status: design_todo
