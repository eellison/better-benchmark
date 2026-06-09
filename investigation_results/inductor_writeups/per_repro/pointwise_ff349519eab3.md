# pointwise_ff349519eab3


## Measured Timings
- Oracle: 16.74 us
- Compile (CDT): 19.36 us
- Ratio: 1.16x

## Classification: EXPAND_CLONE_SOURCE_DRIVEN_STORE

## Current Result

- Family: `expand_clone_view`
- Oracle path: `repros/canonical/pointwise_ff349519eab3/oracle_expand_clone_view.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `real_gap`

## Diagnosis

The oracle materializes the expand-clone-view by loading each source element once and storing it into both duplicated positions of the final contiguous view. Inductor lowers the clone of the zero-stride expanded tensor as a generic output-driven pointwise copy that reloads the same source value for each expanded output element. This means Inductor reads N elements for the first copy and N elements for the second copy (2N loads total), while the oracle reads N elements once and writes 2N elements (N loads, 2N stores via dual tl.store).

## Root cause

Inductor's layout materialization path does not recognize expand-to-clone as a source-driven duplicate-store copy pattern. When a tensor is expanded along a zero-stride dimension (e.g., unsqueeze + expand([..., 2, ...])), the clone materializes a contiguous copy. Inductor iterates over the output elements and loads the input for each one, while the oracle iterates over the source elements and stores each value to both output positions.

For an expand factor of 2, the oracle saves 50% of load traffic: N loads vs 2N loads, with identical 2N stores.

## Kernel count

- Oracle: 1 kernel (_duplicate_expanded_dim_kernel with autotune over BLOCK_N)
- Inductor: 1 kernel (output-driven pointwise copy, same store count but double loads)

## Config exploration

Standard configs (cd=True, combo_kernels, multi_kernel) do not address this pattern since it is a codegen strategy issue, not a fusion or scheduling issue.

## Recommendation

Add a zero-stride expand clone lowering that emits one source load feeding multiple contiguous stores when the requested memory format requires materialization. This is a new lowering pattern in `torch/_inductor/ir.py` or `torch/_inductor/codegen/triton.py`.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/ir.py` (ReinterpretView, expand detection)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (pointwise store codegen)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (clone lowering for expanded tensors)
