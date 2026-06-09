# pointwise_3cfc4c59af74


## Measured Timings
- Oracle: 5.82 us
- Compile (CDT): 7.07 us
- Ratio: 1.22x

- **Gap**: 1.16x (confirmed with CUDAGraph measurement)
- **Classification**: INDEX_ASSERT
- **Root cause**: DLRM inference with index + cat on constant index tensors. Approximately 60% of the gap comes from `tl.device_assert` on indices that are provably in-bounds, and 25% from suboptimal 1D vs 2D tiling in the fused kernel.
- **Fix approach**: Elide device_assert at lowering time for constant index tensors whose values are verified to be in `[0, dim_size)`. Precedent exists in `misc_patterns.py` lines 36-69 where `randperm` is converted to `_unsafe_index`.
- **Status**: DESIGN_TODO
- **Affected repros**: 41 repros with constant index tensors could benefit
