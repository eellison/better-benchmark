Gap diagnosis (classification: `SCHEDULER_FUSION`): Inductor fails to fuse two sibling masked reductions (where+sum over different channel slices of the same input tensor) because the `score_fusion_memory` heuristic computes a shared_data_score of 0 for sibling reductions that read the same buffer at different indices, and `InductorChoices.can_fuse` unconditionally blocks fusion of reductions with score 0. Even after forcing fusion via `score_fusion_memory_threshold=0`, a secondary performance gap remains due to an overly conservative split-reduction factor (10 splits yielding 640 blocks vs oracle's ~756 splits yielding ~48k blocks).

## Artifact

- Oracle path: `repros/canonical/sum_sum_ced249279c9d/oracle_dual_where_sum.py`
- Scope: SqueezeNet1.1 training subgraph -- two independent `where(mask, fill, slice(input)) -> sum(dim=[0,2,3])` branches producing two `f32[64]` outputs.
- Input shapes: `f32[512, 128, 55, 55]`, `b8[512, 64, 55, 55]`, `f32[]`, `b8[512, 64, 55, 55]`
- Correctness: oracle `--check` passes versus eager repro.

## Computation Graph Structure

```
getitem_63: f32[512, 128, 55, 55]
  |
  +-- slice(dim=1, 64:128) --> where(arg63_1, full, ...) --> sum(dim=[0,2,3]) --> f32[64]  (sum_dim_int_list)
  |
  +-- slice(dim=1, 0:64)   --> where(arg64_1, full, ...) --> sum(dim=[0,2,3]) --> f32[64]  (sum_dim_int_list_1)
```

Both branches:
- Read from the SAME f32[512,128,55,55] tensor (different channel halves)
- Share the SAME scalar fill value (arg2_1)
- Have IDENTICAL reduction dimensions: reduce [512, 64, 55, 55] over [0, 2, 3] -> [64]
- Inductor group: `(numel=640, rnumel=154880)` after 10x split reduction

## Kernel Count

| Configuration | Kernel launches | Median latency |
|---|---|---|
| Inductor default | 4 (2 reductions + 2 final sums, serial) | ~437 us |
| Inductor + `score_fusion_memory_threshold=0` | 2 (1 fused reduction + 1 fused final sum) | ~354 us |
| Oracle (dual_where_sum) | 2 (partial + final) | ~179 us |

## Root Cause Analysis

### Issue 1: Fusion Blocked by Heuristic (primary cause of kernel count)

Location: `torch/_inductor/choices.py`, line 584, `InductorChoices.can_fuse()`:

```python
if shared_data_score == 0 and (
    not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
):
    ...
    return False  # heuristic not needed for correctness
```

The two reductions (op0, op2) are grouped together by `get_possible_fusions` because they share buffer names (`arg0_1` for the input tensor, `arg2_1` for the scalar fill). However, `score_fusion_memory` at line 8541 computes common deps via `MemoryDep.__eq__` which requires both the buffer name AND the index expression to match. Since:
- op0 reads `arg0_1` at offset `193600 + ...` (channels 64-128 slice)
- op2 reads `arg0_1` at offset `...` (channels 0-64 slice)
- op0 reads `arg1_1`, op2 reads `arg3_1` (different mask buffers)

The MemoryDeps are not equal, so `common_memory_deps` is empty, giving `score = 0`.

The buffer-overlap fallback scoring (`_can_use_buffer_overlap_scoring`) explicitly excludes reductions at line 8582:
```python
if node1.is_reduction() or node2.is_reduction():
    return False
```

With `score = 0`, `InductorChoices.can_fuse` returns False unconditionally for reductions (the `or node1.is_reduction() or node2.is_reduction()` clause ensures this even with `aggressive_fusion=True`).

### Issue 2: Suboptimal Split Factor (cause of remaining 2x gap)

Location: `torch/_inductor/choices.py`, line 472, `InductorChoices.reduction_split_factor()`:

For our workload (numel=64, reduction_numel=1548800, num_sm=148):
- `target_blocks = 148 * 2048 / (2 * 256) = 592`
- `blocks_per_output = ceil(592 / 64) = 10`
- `split_size = 605` (a divisor of 1548800)
- `num_splits = 10`
- Result: 640 blocks, each processing 154880 elements in a loop

The oracle uses:
- `BLOCK_R = 2048`, giving `ceil(1548800/2048) = 756` partial blocks per channel
- Total: `64 * 756 = 48384` blocks, each processing only 2048 elements

With 148 SMs, Inductor's 640 blocks give ~4.3 blocks/SM (very low occupancy utilization). The oracle's approach is better for memory-bandwidth-bound reductions because it maximizes parallelism and minimizes per-block work.

## Config That Partially Closes the Gap

Setting `config.score_fusion_memory_threshold = 0` enables the fusion (2 kernels instead of 4), reducing latency from ~437us to ~354us (1.24x speedup). However, this does NOT close the gap to the oracle (179us) because the split factor remains at 10.

No existing config combination closes the full gap:
- `max_autotune=True`: no improvement (same split factor selected)
- `aggressive_fusion=True`: still blocked by the reduction-specific clause in `can_fuse`
- `combo_kernels=True`: helps with the final sum (uses combo kernel) but not the main reduction

## Fix Approach

### Fix 1: Allow buffer-overlap scoring for reductions (enables fusion)

**File**: `torch/_inductor/scheduler.py`, method `_can_use_buffer_overlap_scoring` (line ~8582)

Remove or relax the `if node1.is_reduction() or node2.is_reduction(): return False` guard to allow same-buffer-different-index reductions to get a positive overlap score. This would need careful gating to avoid regressing cases where horizontal reduction fusion increases register pressure.

Alternatively, in `score_fusion_memory` (line ~8541), add name-based matching for reductions: if two reduction nodes read from the same buffer name at different indices AND have the same reduction axes, count the buffer size towards the shared_data_score.

### Fix 2: Improve split-reduction heuristic for small-numel large-rnumel cases

**File**: `torch/_inductor/choices.py`, method `reduction_split_factor` (line ~472)

The current heuristic targets `target_blocks = num_sm * threads_per_sm / (2 * num_threads)` which for B200 gives 592 target blocks. With 64 output elements, this means 10 splits. But for large reductions (1.5M elements), more splits would be beneficial to increase GPU utilization and reduce per-block work.

The oracle demonstrates that using ~756 splits (one per 2048 elements per channel) is optimal. A better heuristic might target a per-block reduction size of 2048-4096 elements for large reductions, rather than limiting to num_sm-based targets.

### Fix 3 (alternative): Extend `InductorChoices.can_fuse` to not block sibling reduction fusion

**File**: `torch/_inductor/choices.py`, line 584

The clause `or node1.is_reduction() or node2.is_reduction()` in the `shared_data_score == 0` check means reductions can NEVER be fused horizontally unless they have exact dep equality. This is overly conservative. If two reductions have the same group (numel, rnumel) and share at least one buffer name, they should be fusible.

## Summary

The 2.29x gap has two components:
1. **Kernel count (4 vs 2)**: Fixed by allowing fusion of sibling reductions reading same buffer at different indices. Achievable by relaxing the buffer-overlap scoring exclusion for reductions, or by relaxing the `can_fuse` heuristic in choices.py.
2. **Split factor (10 vs ~756)**: The remaining ~2x gap after fusion requires improving the split-reduction heuristic to use more parallelism for small-numel large-rnumel cases.

Both fixes are in the scheduler/heuristic layer; no custom Triton kernels needed.
